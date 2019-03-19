/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2019                                Université de Bordeaux
 * Copyright (C) 2011-2013,2016,2017                      Inria
 * Copyright (C) 2010,2011,2013,2015-2018                 CNRS
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/sched_policy.h>
#include <datawizard/datastats.h>
#include <datawizard/memory_nodes.h>
#include <drivers/disk/driver_disk.h>
#include <drivers/mpi/driver_mpi_sink.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_common.h>
#include <common/fxt.h>
#include "copy_driver.h"
#include "memalloc.h"
#include <starpu_opencl.h>
#include <starpu_cuda.h>
#include <profiling/profiling.h>
#include <core/disk.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

void _starpu_wake_all_blocked_workers_on_node(unsigned nodeid)
{
	/* wake up all workers on that memory node */
	struct _starpu_memory_node_descr * const descr = _starpu_memory_node_get_description();
	const int cur_workerid = starpu_worker_get_id();
	struct _starpu_worker *cur_worker = cur_workerid>=0?_starpu_get_worker_struct(cur_workerid):NULL;

	STARPU_PTHREAD_RWLOCK_RDLOCK(&descr->conditions_rwlock);

	unsigned nconds = descr->condition_count[nodeid];
	unsigned cond_id;
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		struct _starpu_cond_and_worker *condition;
		condition = &descr->conditions_attached_to_node[nodeid][cond_id];

		if (condition->worker == cur_worker)
		{
			if (condition->cond == &condition->worker->sched_cond)
			{
				condition->worker->state_keep_awake = 1;
			}

			/* No need to wake myself, and I might be called from
			 * the scheduler with mutex locked, through
			 * starpu_prefetch_task_input_on_node */
			continue;
		}

		/* wake anybody waiting on that condition */
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&condition->worker->sched_mutex);
		if (condition->cond == &condition->worker->sched_cond)
		{
			condition->worker->state_keep_awake = 1;
		}
		STARPU_PTHREAD_COND_BROADCAST(condition->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&condition->worker->sched_mutex);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&descr->conditions_rwlock);

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_broadcast(&_starpu_simgrid_transfer_queue[nodeid]);
#endif
}

void starpu_wake_all_blocked_workers(void)
{
	/* workers may be blocked on the various queues' conditions */
	struct _starpu_memory_node_descr * const descr = _starpu_memory_node_get_description();
	const int cur_workerid = starpu_worker_get_id();
	struct _starpu_worker *cur_worker = cur_workerid>=0?_starpu_get_worker_struct(cur_workerid):NULL;

	STARPU_PTHREAD_RWLOCK_RDLOCK(&descr->conditions_rwlock);

	unsigned nconds = descr->total_condition_count;
	unsigned cond_id;
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		struct _starpu_cond_and_worker *condition;
		condition = &descr->conditions_all[cond_id];

		if (condition->worker == cur_worker)
		{
			if (condition->cond == &condition->worker->sched_cond)
			{
				condition->worker->state_keep_awake = 1;
			}

			/* No need to wake myself, and I might be called from
			 * the scheduler with mutex locked, through
			 * starpu_prefetch_task_input_on_node */
			continue;
		}

		/* wake anybody waiting on that condition */
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&condition->worker->sched_mutex);
		if (condition->cond == &condition->worker->sched_cond)
		{
			condition->worker->state_keep_awake = 1;
		}
		STARPU_PTHREAD_COND_BROADCAST(condition->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&condition->worker->sched_mutex);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&descr->conditions_rwlock);

#ifdef STARPU_SIMGRID
	unsigned workerid, nodeid;
	for (workerid = 0; workerid < starpu_worker_get_count(); workerid++)
		starpu_pthread_queue_broadcast(&_starpu_simgrid_task_queue[workerid]);
	for (nodeid = 0; nodeid < starpu_memory_nodes_get_count(); nodeid++)
		starpu_pthread_queue_broadcast(&_starpu_simgrid_transfer_queue[nodeid]);
#endif
}

#ifdef STARPU_USE_FXT
/* we need to identify each communication so that we can match the beginning
 * and the end of a communication in the trace, so we use a unique identifier
 * per communication */
static unsigned long communication_cnt = 0;
#endif

static int copy_data_1_to_1_generic(starpu_data_handle_t handle,
				    struct _starpu_data_replicate *src_replicate,
				    struct _starpu_data_replicate *dst_replicate,
				    struct _starpu_data_request *req)
{
	unsigned src_node = src_replicate->memory_node;
	unsigned dst_node = dst_replicate->memory_node;

	STARPU_ASSERT(src_replicate->refcnt);
	STARPU_ASSERT(dst_replicate->refcnt);

	STARPU_ASSERT(src_replicate->allocated);
	STARPU_ASSERT(dst_replicate->allocated);

#ifdef STARPU_SIMGRID
	if (src_node == STARPU_MAIN_RAM || dst_node == STARPU_MAIN_RAM)
		_starpu_simgrid_data_transfer(handle->ops->get_size(handle), src_node, dst_node);

	return _starpu_simgrid_transfer(handle->ops->get_size(handle), src_node, dst_node, req);
#else /* !SIMGRID */

	int ret = 0;

	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);

#ifdef STARPU_USE_CUDA
	cudaError_t cures;
	cudaStream_t stream;
#endif

	void *src_interface = src_replicate->data_interface;
	void *dst_interface = dst_replicate->data_interface;

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
	if ((src_kind == STARPU_CUDA_RAM) || (dst_kind == STARPU_CUDA_RAM))
	{
		unsigned devid;
		if ((src_kind == STARPU_CUDA_RAM) && (dst_kind == STARPU_CUDA_RAM))
		{
			/* GPU-GPU transfer, issue it from the destination */
			devid = _starpu_memory_node_get_devid(dst_node);
		}
		else
		{
			unsigned node = (dst_kind == STARPU_CUDA_RAM)?dst_node:src_node;
			devid = _starpu_memory_node_get_devid(node);
		}
		starpu_cuda_set_device(devid);
	}
#endif

	switch (_STARPU_MEMORY_NODE_TUPLE(src_kind,dst_kind))
	{
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CPU_RAM):
		/* STARPU_CPU_RAM -> STARPU_CPU_RAM */
		if (copy_methods->ram_to_ram)
			copy_methods->ram_to_ram(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req ? &req->async_channel : NULL);
		break;
#ifdef STARPU_USE_CUDA
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CPU_RAM):
		/* only the proper CUBLAS thread can initiate this directly ! */
#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
		STARPU_ASSERT(_starpu_memory_node_get_local_key() == src_node);
#endif
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() ||
				!(copy_methods->cuda_to_ram_async || copy_methods->any_to_any))
		{
			/* this is not associated to a request so it's synchronous */
			STARPU_ASSERT(copy_methods->cuda_to_ram || copy_methods->any_to_any);
			if (copy_methods->cuda_to_ram)
				copy_methods->cuda_to_ram(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_CUDA_RAM;
			cures = cudaEventCreateWithFlags(&req->async_channel.event.cuda_event, cudaEventDisableTiming);
			if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

			stream = starpu_cuda_get_out_transfer_stream(src_node);
			if (copy_methods->cuda_to_ram_async)
				ret = copy_methods->cuda_to_ram_async(src_interface, src_node, dst_interface, dst_node, stream);
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}

			cures = cudaEventRecord(req->async_channel.event.cuda_event, stream);
			if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);
		}
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CUDA_RAM):
		/* STARPU_CPU_RAM -> CUBLAS_RAM */
		/* only the proper CUBLAS thread can initiate this ! */
#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
		STARPU_ASSERT(_starpu_memory_node_get_local_key() == dst_node);
#endif
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() ||
				!(copy_methods->ram_to_cuda_async || copy_methods->any_to_any))
		{
			/* this is not associated to a request so it's synchronous */
			STARPU_ASSERT(copy_methods->ram_to_cuda || copy_methods->any_to_any);
			if (copy_methods->ram_to_cuda)
				copy_methods->ram_to_cuda(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_CUDA_RAM;
			cures = cudaEventCreateWithFlags(&req->async_channel.event.cuda_event, cudaEventDisableTiming);
			if (STARPU_UNLIKELY(cures != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(cures);

			stream = starpu_cuda_get_in_transfer_stream(dst_node);
			if (copy_methods->ram_to_cuda_async)
				ret = copy_methods->ram_to_cuda_async(src_interface, src_node, dst_interface, dst_node, stream);
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}

			cures = cudaEventRecord(req->async_channel.event.cuda_event, stream);
			if (STARPU_UNLIKELY(cures != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(cures);
		}
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CUDA_RAM):
		/* CUDA - CUDA transfer */
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() ||
				!(copy_methods->cuda_to_cuda_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->cuda_to_cuda || copy_methods->any_to_any);
			/* this is not associated to a request so it's synchronous */
			if (copy_methods->cuda_to_cuda)
				copy_methods->cuda_to_cuda(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_CUDA_RAM;
			cures = cudaEventCreateWithFlags(&req->async_channel.event.cuda_event, cudaEventDisableTiming);
			if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

			stream = starpu_cuda_get_peer_transfer_stream(src_node, dst_node);
			if (copy_methods->cuda_to_cuda_async)
				ret = copy_methods->cuda_to_cuda_async(src_interface, src_node, dst_interface, dst_node, stream);
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}

			cures = cudaEventRecord(req->async_channel.event.cuda_event, stream);
			if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);
		}
		break;
#endif
#ifdef STARPU_USE_OPENCL
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_OPENCL_RAM,STARPU_CPU_RAM):
		/* OpenCL -> RAM */
		STARPU_ASSERT(_starpu_memory_node_get_local_key() == src_node);
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_opencl_copy_disabled() ||
				!(copy_methods->opencl_to_ram_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->opencl_to_ram || copy_methods->any_to_any);
			/* this is not associated to a request so it's synchronous */
			if (copy_methods->opencl_to_ram)
				copy_methods->opencl_to_ram(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_OPENCL_RAM;
			if (copy_methods->opencl_to_ram_async)
				ret = copy_methods->opencl_to_ram_async(src_interface, src_node, dst_interface, dst_node, &(req->async_channel.event.opencl_event));
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}
		}
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_OPENCL_RAM):
		/* STARPU_CPU_RAM -> STARPU_OPENCL_RAM */
		STARPU_ASSERT(_starpu_memory_node_get_local_key() == dst_node);
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_opencl_copy_disabled() ||
				!(copy_methods->ram_to_opencl_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->ram_to_opencl || copy_methods->any_to_any);
			/* this is not associated to a request so it's synchronous */
			if (copy_methods->ram_to_opencl)
				copy_methods->ram_to_opencl(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_OPENCL_RAM;
			if (copy_methods->ram_to_opencl_async)
				ret = copy_methods->ram_to_opencl_async(src_interface, src_node, dst_interface, dst_node, &(req->async_channel.event.opencl_event));
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}
		}
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_OPENCL_RAM,STARPU_OPENCL_RAM):
		/* STARPU_OPENCL_RAM -> STARPU_OPENCL_RAM */
		STARPU_ASSERT(_starpu_memory_node_get_local_key() == dst_node || _starpu_memory_node_get_local_key() == src_node);
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_opencl_copy_disabled() ||
				!(copy_methods->opencl_to_opencl_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->opencl_to_opencl || copy_methods->any_to_any);
			/* this is not associated to a request so it's synchronous */
			if (copy_methods->opencl_to_opencl)
				copy_methods->opencl_to_opencl(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_OPENCL_RAM;
			if (copy_methods->opencl_to_opencl_async)
				ret = copy_methods->opencl_to_opencl_async(src_interface, src_node, dst_interface, dst_node, &(req->async_channel.event.opencl_event));
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}
		}
		break;
#endif
#ifdef STARPU_USE_MIC
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_MIC_RAM):
		/* RAM -> MIC */
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mic_copy_disabled() ||
				!(copy_methods->ram_to_mic_async || copy_methods->any_to_any))
		{
			/* this is not associated to a request so it's synchronous */
			STARPU_ASSERT(copy_methods->ram_to_mic || copy_methods->any_to_any);
			if (copy_methods->ram_to_mic)
				copy_methods->ram_to_mic(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_MIC_RAM;
			if (copy_methods->ram_to_mic_async)
				ret = copy_methods->ram_to_mic_async(src_interface, src_node, dst_interface, dst_node);
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}
			_starpu_mic_init_event(&(req->async_channel.event.mic_event), dst_node);
		}
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_MIC_RAM,STARPU_CPU_RAM):
		/* MIC -> RAM */
		if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mic_copy_disabled() ||
				!(copy_methods->mic_to_ram_async || copy_methods->any_to_any))
		{
			/* this is not associated to a request so it's synchronous */
			STARPU_ASSERT(copy_methods->mic_to_ram || copy_methods->any_to_any);
			if (copy_methods->mic_to_ram)
				copy_methods->mic_to_ram(src_interface, src_node, dst_interface, dst_node);
			else
				copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		}
		else
		{
			req->async_channel.type = STARPU_MIC_RAM;
			if (copy_methods->mic_to_ram_async)
				ret = copy_methods->mic_to_ram_async(src_interface, src_node, dst_interface, dst_node);
			else
			{
				STARPU_ASSERT(copy_methods->any_to_any);
				ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
			}
			_starpu_mic_init_event(&(req->async_channel.event.mic_event), src_node);
		}
		break;
	/* TODO: MIC -> MIC */
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
        case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_MPI_MS_RAM):
                if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mpi_ms_copy_disabled() ||
                                !(copy_methods->ram_to_mpi_ms_async || copy_methods->any_to_any))
                {
                        /* this is not associated to a request so it's synchronous */
                        STARPU_ASSERT(copy_methods->ram_to_mpi_ms || copy_methods->any_to_any);
                        if (copy_methods->ram_to_mpi_ms)
                                copy_methods->ram_to_mpi_ms(src_interface, src_node, dst_interface, dst_node);
                        else
                                copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
                }
                else
                {
                        req->async_channel.type = STARPU_MPI_MS_RAM;
                        if(copy_methods->ram_to_mpi_ms_async)
                                ret = copy_methods->ram_to_mpi_ms_async(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
                        else
                        {
                                STARPU_ASSERT(copy_methods->any_to_any);
                                ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
                        }
                }
                break;

        case _STARPU_MEMORY_NODE_TUPLE(STARPU_MPI_MS_RAM,STARPU_CPU_RAM):
                if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mpi_ms_copy_disabled() ||
                                !(copy_methods->mpi_ms_to_ram_async || copy_methods->any_to_any))
                {
                        /* this is not associated to a request so it's synchronous */
                        STARPU_ASSERT(copy_methods->mpi_ms_to_ram || copy_methods->any_to_any);
                        if (copy_methods->mpi_ms_to_ram)
                                copy_methods->mpi_ms_to_ram(src_interface, src_node, dst_interface, dst_node);
                        else
                                copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
                }
                else
                {
                        req->async_channel.type = STARPU_MPI_MS_RAM;
                        if(copy_methods->mpi_ms_to_ram_async)
                                ret = copy_methods->mpi_ms_to_ram_async(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
                        else
                        {
                                STARPU_ASSERT(copy_methods->any_to_any);
                                ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
                        }
                }
                break;

        case _STARPU_MEMORY_NODE_TUPLE(STARPU_MPI_MS_RAM,STARPU_MPI_MS_RAM):
                if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mpi_ms_copy_disabled() ||
                                !(copy_methods->mpi_ms_to_mpi_ms_async || copy_methods->any_to_any))
                {
                        /* this is not associated to a request so it's synchronous */
                        STARPU_ASSERT(copy_methods->mpi_ms_to_mpi_ms || copy_methods->any_to_any);
                        if (copy_methods->mpi_ms_to_mpi_ms)
                                copy_methods->mpi_ms_to_mpi_ms(src_interface, src_node, dst_interface, dst_node);
                        else
                                copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
                }
                else
                {
                        req->async_channel.type = STARPU_MPI_MS_RAM;
                        if(copy_methods->mpi_ms_to_mpi_ms_async)
                                ret = copy_methods->mpi_ms_to_mpi_ms_async(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
                        else
                        {
                                STARPU_ASSERT(copy_methods->any_to_any);
                                ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
                        }
                }
                break;
#endif
#ifdef STARPU_USE_SCC
		/* SCC RAM associated to the master process is considered as
		 * the main memory node. */
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_SCC_RAM):
		/* master private SCC RAM -> slave private SCC RAM */
		if (copy_methods->scc_src_to_sink)
			copy_methods->scc_src_to_sink(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_SCC_RAM,STARPU_CPU_RAM):
		/* slave private SCC RAM -> master private SCC RAM */
		if (copy_methods->scc_sink_to_src)
			copy_methods->scc_sink_to_src(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		break;
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_SCC_RAM,STARPU_SCC_RAM):
		/* slave private SCC RAM -> slave private SCC RAM */
		if (copy_methods->scc_sink_to_sink)
			copy_methods->scc_sink_to_sink(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
		break;
#endif

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_DISK_RAM):
                if (req && !starpu_asynchronous_copy_disabled())
                {
                        req->async_channel.type = STARPU_DISK_RAM;
                        req->async_channel.event.disk_event.requests = NULL;
                        req->async_channel.event.disk_event.ptr = NULL;
                        req->async_channel.event.disk_event.handle = NULL;
                }
		if(copy_methods->any_to_any)
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);

		else
		{
			void *obj = starpu_data_handle_to_pointer(handle, dst_node);
			void * ptr = NULL;
			starpu_ssize_t size = 0;
			handle->ops->pack_data(handle, src_node, &ptr, &size);
			ret = _starpu_disk_full_write(src_node, dst_node, obj, ptr, size, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
			if (ret == 0)
			{
				/* write is already finished, ptr was allocated in pack_data */
				_starpu_free_flags_on_node(src_node, ptr, size, 0);
			}
			else if (ret == -EAGAIN)
			{
				STARPU_ASSERT(req);
				req->async_channel.event.disk_event.ptr = ptr;
				req->async_channel.event.disk_event.node = src_node;
				req->async_channel.event.disk_event.size = size;
			}

			STARPU_ASSERT(ret == 0 || ret == -EAGAIN);
		}
		break;

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM,STARPU_CPU_RAM):
                if (req && !starpu_asynchronous_copy_disabled())
                {
                        req->async_channel.type = STARPU_DISK_RAM;
                        req->async_channel.event.disk_event.requests = NULL;
                        req->async_channel.event.disk_event.ptr = NULL;
                        req->async_channel.event.disk_event.handle = NULL;
                }
		if(copy_methods->any_to_any)
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled()  ? &req->async_channel : NULL);
		else
		{
			void *obj = starpu_data_handle_to_pointer(handle, src_node);
			void * ptr = NULL;
			size_t size = 0;
			ret = _starpu_disk_full_read(src_node, dst_node, obj, &ptr, &size, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
			if (ret == 0)
			{
				/* read is already finished, we can already unpack */
				handle->ops->unpack_data(handle, dst_node, ptr, size);
				/* ptr is allocated in full_read */
				_starpu_free_flags_on_node(dst_node, ptr, size, 0);
			}
			else if (ret == -EAGAIN)
			{
				STARPU_ASSERT(req);
				req->async_channel.event.disk_event.ptr = ptr;
				req->async_channel.event.disk_event.node = dst_node;
				req->async_channel.event.disk_event.size = size;
				req->async_channel.event.disk_event.handle = handle;
			}

			STARPU_ASSERT(ret == 0 || ret == -EAGAIN);
		}
		break;

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM,STARPU_DISK_RAM):
                if (req && !starpu_asynchronous_copy_disabled())
                {
                        req->async_channel.type = STARPU_DISK_RAM;
                        req->async_channel.event.disk_event.requests = NULL;
                        req->async_channel.event.disk_event.ptr = NULL;
                        req->async_channel.event.disk_event.handle = NULL;
                }
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
		break;

	default:
		STARPU_ABORT();
		break;
	}

	return ret;
#endif /* !SIMGRID */
}

static int update_map_generic(starpu_data_handle_t handle,
				    struct _starpu_data_replicate *src_replicate,
				    struct _starpu_data_replicate *dst_replicate,
				    struct _starpu_data_request *req STARPU_ATTRIBUTE_UNUSED)
{
	unsigned src_node = src_replicate->memory_node;
	unsigned dst_node = dst_replicate->memory_node;

	STARPU_ASSERT(src_replicate->refcnt);
	STARPU_ASSERT(dst_replicate->refcnt);

	STARPU_ASSERT((src_replicate->mapped && dst_replicate->allocated)
			||(src_replicate->allocated && dst_replicate->mapped));

	void *src_interface = src_replicate->data_interface;
	void *dst_interface = dst_replicate->data_interface;

	handle->ops->update_map(src_interface, src_node, dst_interface, dst_node);

	return 0;
}

int STARPU_ATTRIBUTE_WARN_UNUSED_RESULT _starpu_driver_copy_data_1_to_1(starpu_data_handle_t handle,
									struct _starpu_data_replicate *src_replicate,
									struct _starpu_data_replicate *dst_replicate,
									unsigned donotread,
									struct _starpu_data_request *req,
									unsigned may_alloc,
									unsigned prefetch STARPU_ATTRIBUTE_UNUSED)
{
	if (!donotread)
	{
		STARPU_ASSERT(src_replicate->allocated || src_replicate->mapped);
		STARPU_ASSERT(src_replicate->refcnt);
	}

	unsigned src_node = src_replicate->memory_node;
	unsigned dst_node = dst_replicate->memory_node;

	if (!dst_replicate->allocated && !dst_replicate->mapped
			&& handle->ops->map_data
			&& (_starpu_memory_node_get_mapped(dst_replicate->memory_node) /* || handle wants it */))
	{
		/* Memory node which can just map the main memory, try to map.  */
		STARPU_ASSERT(starpu_node_get_kind(src_replicate->memory_node) == STARPU_CPU_RAM);
		if (!handle->ops->map_data(
				src_replicate->data_interface, src_replicate->memory_node,
				dst_replicate->data_interface, dst_replicate->memory_node))
		{
			dst_replicate->mapped = 1;

			if (_starpu_node_needs_map_update(dst_node))
			{
				switch (starpu_node_get_kind(dst_node))
				{
					case STARPU_OPENCL_RAM:
					/* OpenCL mappings write access defaults to the device */
						dst_replicate->map_write = 1;
						break;
					case STARPU_CUDA_RAM:
						dst_replicate->map_write = 0;
						break;
					case STARPU_CPU_RAM:
					default:
						/* Should not happen */
						STARPU_ABORT();
						break;
				}
			}
		}
	}

	/* first make sure the destination has an allocated buffer */
	if (!dst_replicate->allocated && !dst_replicate->mapped)
	{
		if (!may_alloc || _starpu_is_reclaiming(dst_node))
			/* We're not supposed to allocate there at the moment */
			return -ENOMEM;

		int ret_alloc = _starpu_allocate_memory_on_node(handle, dst_replicate, req ? req->prefetch : 0);
		if (ret_alloc)
			return -ENOMEM;
	}

	STARPU_ASSERT(dst_replicate->allocated || dst_replicate->mapped);
	STARPU_ASSERT(dst_replicate->refcnt);

	/* In the case of a mapped data, we are here requested either
	 * - because the destination will write to it, and thus needs write
	 *   access.
	 * - because the source was modified, and the destination needs to get
	 *   updated.
	 * All in all, any data change will actually trigger both.
	 */
	if (dst_replicate->mapped)
	{
		STARPU_ASSERT(src_replicate->memory_node == 0);
		if (_starpu_node_needs_map_update(dst_node))
		{
			/* We need to flush from RAM to the device */
			if (!dst_replicate->map_write)
			{
				update_map_generic(handle, src_replicate, dst_replicate, req);
				dst_replicate->map_write = 1;
			}
		}
	}

	else if (src_replicate->mapped)
	{
		STARPU_ASSERT(dst_replicate->memory_node == 0);
		if (_starpu_node_needs_map_update(src_node))
		{
			/* We need to flush from the device to the RAM */
			if (src_replicate->map_write)
			{
				update_map_generic(handle, src_replicate, dst_replicate, req);
				src_replicate->map_write = 0;
			}
		}
	}

	/* if there is no need to actually read the data,
	 * we do not perform any transfer */
	else if (!donotread)
	{
		unsigned long STARPU_ATTRIBUTE_UNUSED com_id = 0;
		size_t size = _starpu_data_get_size(handle);
		_starpu_bus_update_profiling_info((int)src_node, (int)dst_node, size);

#ifdef STARPU_USE_FXT
		com_id = STARPU_ATOMIC_ADDL(&communication_cnt, 1);

		if (req)
			req->com_id = com_id;
#endif

		dst_replicate->initialized = 1;

		_STARPU_TRACE_START_DRIVER_COPY(src_node, dst_node, size, com_id, prefetch, handle);
		int ret_copy = copy_data_1_to_1_generic(handle, src_replicate, dst_replicate, req);
		if (!req)
			/* Synchronous, this is already finished */
			_STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, com_id, prefetch);

		return ret_copy;
	}

	return 0;
}

void starpu_interface_start_driver_copy_async(unsigned src_node, unsigned dst_node, double *start)
{
	*start = starpu_timing_now();
	_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
}

void starpu_interface_end_driver_copy_async(unsigned src_node, unsigned dst_node, double start)
{
	double end = starpu_timing_now();
	double elapsed = end - start;
	if (elapsed > 300)
	{
		static int warned = 0;
		if (!warned)
		{
			char src_name[16], dst_name[16];
			warned = 1;
			starpu_memory_node_get_name(src_node, src_name, sizeof(src_name));
			starpu_memory_node_get_name(dst_node, dst_name, sizeof(dst_name));

			_STARPU_DISP("Warning: the submission of asynchronous transfer from %s to %s took a very long time (%f ms)\nFor proper asynchronous transfer overlapping, data registered to StarPU must be allocated with starpu_malloc() or pinned with starpu_memory_pin()\n", src_name, dst_name, elapsed / 1000.);
		}
	}
	_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);
}

/* This can be used by interfaces to easily transfer a piece of data without
 * caring about the particular transfer methods.  */

/* This should either return 0 if the transfer is complete, or -EAGAIN if the
 * transfer is still pending, and will have to be waited for by
 * _starpu_driver_test_request_completion/_starpu_driver_wait_request_completion
 */
int starpu_interface_copy(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, void *async_data)
{
	struct _starpu_async_channel *async_channel = async_data;

	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);

	switch (_STARPU_MEMORY_NODE_TUPLE(src_kind,dst_kind))
	{
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CPU_RAM):
		memcpy((void *) (dst + dst_offset), (void *) (src + src_offset), size);
		return 0;

#ifdef STARPU_USE_CUDA
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CPU_RAM):
		return starpu_cuda_copy_async_sync(
				(void*) (src + src_offset), src_node,
				(void*) (dst + dst_offset), dst_node,
				size,
				async_channel?starpu_cuda_get_out_transfer_stream(src_node):NULL,
				cudaMemcpyDeviceToHost);

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CUDA_RAM):
		return starpu_cuda_copy_async_sync(
				(void*) (src + src_offset), src_node,
				(void*) (dst + dst_offset), dst_node,
				size,
				async_channel?starpu_cuda_get_in_transfer_stream(dst_node):NULL,
				cudaMemcpyHostToDevice);

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CUDA_RAM):
		return starpu_cuda_copy_async_sync(
				(void*) (src + src_offset), src_node,
				(void*) (dst + dst_offset), dst_node,
				size,
				async_channel?starpu_cuda_get_peer_transfer_stream(src_node, dst_node):NULL,
				cudaMemcpyDeviceToDevice);

#endif
#ifdef STARPU_USE_OPENCL
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_OPENCL_RAM,STARPU_CPU_RAM):
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_OPENCL_RAM):
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_OPENCL_RAM,STARPU_OPENCL_RAM):
		return starpu_opencl_copy_async_sync(
				src, src_offset, src_node,
				dst, dst_offset, dst_node,
				size,
				&async_channel->event.opencl_event);
#endif
#ifdef STARPU_USE_MIC
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_MIC_RAM,STARPU_CPU_RAM):
		if (async_data)
			return _starpu_mic_copy_mic_to_ram_async(
					(void*) (src + src_offset), src_node,
					(void*) (dst + dst_offset), dst_node,
					size);
		else
			return _starpu_mic_copy_mic_to_ram(
					(void*) (src + src_offset), src_node,
					(void*) (dst + dst_offset), dst_node,
					size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_MIC_RAM):
		if (async_data)
			return _starpu_mic_copy_ram_to_mic_async(
					(void*) (src + src_offset), src_node,
					(void*) (dst + dst_offset), dst_node,
					size);
		else
			return _starpu_mic_copy_ram_to_mic(
					(void*) (src + src_offset), src_node,
					(void*) (dst + dst_offset), dst_node,
					size);
	/* TODO: MIC->MIC */
#endif
#ifdef STARPU_USE_SCC
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_SCC_RAM,STARPU_CPU_RAM):
		return _starpu_scc_copy_sink_to_src(
				(void*) (src + src_offset), src_node,
				(void*) (dst + dst_offset), dst_node,
				size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_SCC_RAM):
		return _starpu_scc_copy_src_to_sink(
				(void*) (src + src_offset), src_node,
				(void*) (dst + dst_offset), dst_node,
				size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_SCC_RAM,STARPU_SCC_RAM):
		return _starpu_scc_copy_sink_to_sink(
				(void*) (src + src_offset), src_node,
				(void*) (dst + dst_offset), dst_node,
				size);
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
        case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM, STARPU_MPI_MS_RAM):
                if (async_data)
                        return _starpu_mpi_copy_ram_to_mpi_async(
                                        (void*) (src + src_offset), src_node,
                                        (void*) (dst + dst_offset), dst_node,
                                        size, async_data);
                else
                        return _starpu_mpi_copy_ram_to_mpi_sync(
                                        (void*) (src + src_offset), src_node,
                                        (void*) (dst + dst_offset), dst_node,
                                        size);
        case _STARPU_MEMORY_NODE_TUPLE(STARPU_MPI_MS_RAM, STARPU_CPU_RAM):
                if (async_data)
                        return _starpu_mpi_copy_mpi_to_ram_async(
                                        (void*) (src + src_offset), src_node,
                                        (void*) (dst + dst_offset), dst_node,
                                        size, async_data);
                else
                        return _starpu_mpi_copy_mpi_to_ram_sync(
                                        (void*) (src + src_offset), src_node,
                                        (void*) (dst + dst_offset), dst_node,
                                        size);

        case _STARPU_MEMORY_NODE_TUPLE(STARPU_MPI_MS_RAM, STARPU_MPI_MS_RAM):
                if (async_data)
                        return _starpu_mpi_copy_sink_to_sink_async(
                                        (void*) (src + src_offset), src_node,
                                        (void*) (dst + dst_offset), dst_node,
                                        size, async_data);
                else
                        return _starpu_mpi_copy_sink_to_sink_sync(
                                        (void*) (src + src_offset), src_node,
                                        (void*) (dst + dst_offset), dst_node,
                                        size);
#endif

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM, STARPU_DISK_RAM):
	{
		return _starpu_disk_copy_src_to_disk(
			(void*) (src + src_offset), src_node,
			(void*) dst, dst_offset, dst_node,
			size, async_channel);
	}
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM, STARPU_CPU_RAM):
		return _starpu_disk_copy_disk_to_src(
			(void*) src, src_offset, src_node,
			(void*) (dst + dst_offset), dst_node,
			size, async_channel);

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM, STARPU_DISK_RAM):
		return _starpu_disk_copy_disk_to_disk(
			(void*) src, src_offset, src_node,
			(void*) dst, dst_offset, dst_node,
			size, async_channel);

	default:
		STARPU_ABORT();
		return -1;
	}
	return 0;
}

uintptr_t starpu_interface_map(uintptr_t src, size_t src_offset, unsigned src_node, unsigned dst_node, size_t size, int *ret)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);

	switch (_STARPU_MEMORY_NODE_TUPLE(src_kind,dst_kind))
	{
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CPU_RAM):
		return src + src_offset;
#if defined(STARPU_USE_CUDA) && defined(STARPU_USE_CUDA_MAP)
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CUDA_RAM):
		return _starpu_cuda_map_ram(
				(void*) (src + src_offset), src_node,
				dst_node,
				size, ret);
#endif
#ifdef STARPU_USE_OPENCL
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_OPENCL_RAM):
		return _starpu_opencl_map_ram(
				src, src_offset, src_node,
				dst_node,
				size, ret);
#endif
	default:
		STARPU_ABORT();
		return -1;
	}
	*ret = -EIO;
	return 0;
}

int starpu_interface_unmap(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, unsigned dst_node, size_t size)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);

	switch (_STARPU_MEMORY_NODE_TUPLE(src_kind,dst_kind))
	{
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CPU_RAM):
		return 0;
#if defined(STARPU_USE_CUDA) && defined(STARPU_USE_CUDA_MAP)
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CUDA_RAM):
		return _starpu_cuda_unmap_ram(
				(void*) (src + src_offset), src_node,
				(void*) dst, dst_node,
				size);
#endif
#ifdef STARPU_USE_OPENCL
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_OPENCL_RAM):
		return _starpu_opencl_unmap_ram(
				src, src_offset, src_node,
				dst, dst_node,
				size);
#endif
	default:
		STARPU_ABORT();
		return -1;
	}
	return -EIO;
}

int starpu_interface_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);

	switch (_STARPU_MEMORY_NODE_TUPLE(src_kind,dst_kind))
	{
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CPU_RAM):
		/* Memory mappings are cache-coherent */
		/* FIXME: not on SCC */
		return 0;

#if defined(STARPU_USE_CUDA) && defined(STARPU_USE_CUDA_MAP)
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CUDA_RAM):
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CPU_RAM):
		/* CUDA mappings are coherent */
		/* FIXME: not necessarily, depends on board capabilities */
		return 0;
#endif
#ifdef STARPU_USE_OPENCL
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_OPENCL_RAM):
		STARPU_ASSERT(dst_offset == 0);
		return _starpu_opencl_update_opencl_map(
				src, src_offset, src_node,
				dst, dst_node,
				size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_OPENCL_RAM,STARPU_CPU_RAM):
		STARPU_ASSERT(src_offset == 0);
		return _starpu_opencl_update_cpu_map(
				src, src_node,
				dst, src_offset, dst_node,
				size);
#endif
	default:
		STARPU_ABORT();
		return -1;
	}
	return -EIO;
}

void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	_starpu_simgrid_wait_transfer_event(&async_channel->event);
#else /* !SIMGRID */
	enum starpu_node_kind kind = async_channel->type;
#ifdef STARPU_USE_CUDA
	cudaEvent_t event;
	cudaError_t cures;
#endif

	switch (kind)
	{
#ifdef STARPU_USE_CUDA
	case STARPU_CUDA_RAM:
		event = (*async_channel).event.cuda_event;

		cures = cudaEventSynchronize(event);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		cures = cudaEventDestroy(event);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		break;
#endif
#ifdef STARPU_USE_OPENCL
	case STARPU_OPENCL_RAM:
	{
		cl_int err;
		if ((*async_channel).event.opencl_event == NULL)
			STARPU_ABORT();
		err = clWaitForEvents(1, &((*async_channel).event.opencl_event));
		if (STARPU_UNLIKELY(err != CL_SUCCESS))
			STARPU_OPENCL_REPORT_ERROR(err);
		err = clReleaseEvent((*async_channel).event.opencl_event);
		if (STARPU_UNLIKELY(err != CL_SUCCESS))
			STARPU_OPENCL_REPORT_ERROR(err);
	      break;
	}
#endif
#ifdef STARPU_USE_MIC
	case STARPU_MIC_RAM:
		_starpu_mic_wait_request_completion(&(async_channel->event.mic_event));
		break;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
        case STARPU_MPI_MS_RAM:
                _starpu_mpi_common_wait_event(async_channel);
                break;
#endif
	case STARPU_DISK_RAM:
		starpu_disk_wait_request(async_channel);
		if (async_channel->event.disk_event.ptr != NULL)
		{
			if (async_channel->event.disk_event.handle != NULL)
			{
				/* read is finished, we can already unpack */
				async_channel->event.disk_event.handle->ops->unpack_data(async_channel->event.disk_event.handle, async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size);
				/* ptr is allocated in full_read */
				_starpu_free_flags_on_node(async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size, 0);
			}
			else
			{
				/* write is finished, ptr was allocated in pack_data */
				_starpu_free_flags_on_node(async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size, 0);
			}
		}
		break;
	case STARPU_CPU_RAM:
	default:
		STARPU_ABORT();
	}
#endif /* !SIMGRID */
}

unsigned _starpu_driver_test_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	return _starpu_simgrid_test_transfer_event(&async_channel->event);
#else /* !SIMGRID */
	enum starpu_node_kind kind = async_channel->type;
	unsigned success = 0;
#ifdef STARPU_USE_CUDA
	cudaEvent_t event;
#endif

	switch (kind)
	{
#ifdef STARPU_USE_CUDA
	case STARPU_CUDA_RAM:
		event = (*async_channel).event.cuda_event;
		cudaError_t cures = cudaEventQuery(event);

		success = (cures == cudaSuccess);
		if (success)
			cudaEventDestroy(event);
		else if (cures != cudaErrorNotReady)
			STARPU_CUDA_REPORT_ERROR(cures);
		break;
#endif
#ifdef STARPU_USE_OPENCL
	case STARPU_OPENCL_RAM:
	{
		cl_int event_status;
		cl_event opencl_event = (*async_channel).event.opencl_event;
		if (opencl_event == NULL) STARPU_ABORT();
		cl_int err = clGetEventInfo(opencl_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
		if (STARPU_UNLIKELY(err != CL_SUCCESS))
			STARPU_OPENCL_REPORT_ERROR(err);
		if (event_status < 0)
			STARPU_OPENCL_REPORT_ERROR(event_status);
		if (event_status == CL_COMPLETE)
		{
			err = clReleaseEvent(opencl_event);
			if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		}
		success = (event_status == CL_COMPLETE);
		break;
	}
#endif
#ifdef STARPU_USE_MIC
	case STARPU_MIC_RAM:
		success = _starpu_mic_request_is_complete(&(async_channel->event.mic_event));
		break;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
        case STARPU_MPI_MS_RAM:
                success = _starpu_mpi_common_test_event(async_channel);
                break;
#endif
	case STARPU_DISK_RAM:
		success = starpu_disk_test_request(async_channel);
		if (async_channel->event.disk_event.ptr != NULL && success)
		{
			if (async_channel->event.disk_event.handle != NULL)
			{
				/* read is finished, we can already unpack */
				async_channel->event.disk_event.handle->ops->unpack_data(async_channel->event.disk_event.handle, async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size);
				/* ptr is allocated in full_read */
				_starpu_free_flags_on_node(async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size, 0);
			}
			else
			{
				/* write is finished, ptr was allocated in pack_data */
				_starpu_free_flags_on_node(async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size, 0);
			}
		}
		break;
	case STARPU_CPU_RAM:
	default:
		STARPU_ABORT_MSG("Memory is not recognized (kind %d) \n", kind);
	}

	return success;
#endif /* !SIMGRID */
}
