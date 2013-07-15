/* StarPU --- Runtime system for heterogeneous multicore architectures.  * * Copyright (C) 2010-2013  Université de Bordeaux 1 * Copyright (C) 2010, 2011, 2013  Centre National de la Recherche Scientifique
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
#include <drivers/disk/driver_disk.h>
#include <common/fxt.h>
#include "copy_driver.h"
#include "memalloc.h"
#include <starpu_opencl.h>
#include <starpu_cuda.h>
#include <profiling/profiling.h>
#include <core/disk.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#include <msg/msg.h>
#endif

void _starpu_wake_all_blocked_workers_on_node(unsigned nodeid)
{
	/* wake up all workers on that memory node */
	unsigned cond_id;

	struct _starpu_memory_node_descr * const descr = _starpu_memory_node_get_description();

	STARPU_PTHREAD_RWLOCK_RDLOCK(&descr->conditions_rwlock);

	unsigned nconds = descr->condition_count[nodeid];
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		struct _starpu_cond_and_mutex *condition;
		condition  = &descr->conditions_attached_to_node[nodeid][cond_id];

		/* wake anybody waiting on that condition */
		STARPU_PTHREAD_MUTEX_LOCK(condition->mutex);
		STARPU_PTHREAD_COND_BROADCAST(condition->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(condition->mutex);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&descr->conditions_rwlock);
}

void starpu_wake_all_blocked_workers(void)
{
	/* workers may be blocked on the various queues' conditions */
	unsigned cond_id;

	struct _starpu_memory_node_descr * const descr = _starpu_memory_node_get_description();

	STARPU_PTHREAD_RWLOCK_RDLOCK(&descr->conditions_rwlock);

	unsigned nconds = descr->total_condition_count;
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		struct _starpu_cond_and_mutex *condition;
		condition  = &descr->conditions_all[cond_id];

		/* wake anybody waiting on that condition */
		STARPU_PTHREAD_MUTEX_LOCK(condition->mutex);
		STARPU_PTHREAD_COND_BROADCAST(condition->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(condition->mutex);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&descr->conditions_rwlock);
}

#ifdef STARPU_USE_FXT
/* we need to identify each communication so that we can match the beginning
 * and the end of a communication in the trace, so we use a unique identifier
 * per communication */
static unsigned communication_cnt = 0;
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

	_starpu_comm_amounts_inc(src_node, dst_node, handle->ops->get_size(handle));

#ifdef STARPU_SIMGRID
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

#if defined(STARPU_USE_CUDA) && defined(HAVE_CUDA_MEMCPY_PEER)
	if ((src_kind == STARPU_CUDA_RAM) || (dst_kind == STARPU_CUDA_RAM))
	{
		int node = (dst_kind == STARPU_CUDA_RAM)?dst_node:src_node;
		starpu_cuda_set_device(_starpu_memory_node_get_devid(node));
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
#if !defined(HAVE_CUDA_MEMCPY_PEER)
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
			cures = cudaEventCreate(&req->async_channel.event.cuda_event);
			if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

			stream = starpu_cuda_get_local_out_transfer_stream();
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
#if !defined(HAVE_CUDA_MEMCPY_PEER)
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
			cures = cudaEventCreate(&req->async_channel.event.cuda_event);
			if (STARPU_UNLIKELY(cures != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(cures);

			stream = starpu_cuda_get_local_in_transfer_stream();
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
			cures = cudaEventCreate(&req->async_channel.event.cuda_event);
			if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

			stream = starpu_cuda_get_local_peer_transfer_stream();
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
		if(copy_methods->any_to_any)
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req ? &req->async_channel : NULL);

		else
		{
			//TODO OBJ
			void * obj;
			void * ptr = NULL;
			starpu_ssize_t size = 0;
			handle->ops->pack_data(handle, src_node, &ptr, &size);
			ret = _starpu_disk_full_write(src_node, dst_node, obj, ptr, size);
		}
		break;
		
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM,STARPU_CPU_RAM):
		if(copy_methods->any_to_any) 
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req ? &req->async_channel : NULL);
		else
		{
			//TODO OBJ
			void * obj;
			void * ptr = NULL;
			size_t size = 0;
			ret = _starpu_disk_full_read(src_node, dst_node, obj, &ptr, &size);
			handle->ops->unpack_data(handle, dst_node, ptr, size); 
		}
		break;

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM,STARPU_DISK_RAM):	
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req ? &req->async_channel : NULL);
		break;
		
	default:
		STARPU_ABORT();
		break;
	}
	
	return ret;
#endif /* !SIMGRID */
}

int STARPU_ATTRIBUTE_WARN_UNUSED_RESULT _starpu_driver_copy_data_1_to_1(starpu_data_handle_t handle,
									struct _starpu_data_replicate *src_replicate,
									struct _starpu_data_replicate *dst_replicate,
									unsigned donotread,
									struct _starpu_data_request *req,
									unsigned may_alloc)
{
	if (!donotread)
	{
		STARPU_ASSERT(src_replicate->allocated);
		STARPU_ASSERT(src_replicate->refcnt);
	}

	int ret_alloc, ret_copy;
	unsigned STARPU_ATTRIBUTE_UNUSED com_id = 0;

	unsigned src_node = src_replicate->memory_node;
	unsigned dst_node = dst_replicate->memory_node;

	/* first make sure the destination has an allocated buffer */
	if (!dst_replicate->allocated)
	{
		if (!may_alloc)
			return -ENOMEM;

		ret_alloc = _starpu_allocate_memory_on_node(handle, dst_replicate, req ? req->prefetch : 0);
		if (ret_alloc)
			return -ENOMEM;
	}

	STARPU_ASSERT(dst_replicate->allocated);
	STARPU_ASSERT(dst_replicate->refcnt);

	/* if there is no need to actually read the data,
	 * we do not perform any transfer */
	if (!donotread)
	{
		size_t size = _starpu_data_get_size(handle);
		_starpu_bus_update_profiling_info((int)src_node, (int)dst_node, size);

#ifdef STARPU_USE_FXT
		com_id = STARPU_ATOMIC_ADD(&communication_cnt, 1);

		if (req)
			req->com_id = com_id;
#endif

		_STARPU_TRACE_START_DRIVER_COPY(src_node, dst_node, size, com_id);
		ret_copy = copy_data_1_to_1_generic(handle, src_replicate, dst_replicate, req);

		return ret_copy;
	}

	return 0;
}

/* This can be used by interfaces to easily transfer a piece of data without
 * caring about the particular CUDA/OpenCL methods.  */

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
		memcpy((void *) dst + dst_offset, (void *) src + src_offset, size);
		return 0;

#ifdef STARPU_USE_CUDA
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CPU_RAM):
		return starpu_cuda_copy_async_sync(
				(void*) src + src_offset, src_node,
				(void*) dst + dst_offset, dst_node,
				size,
				async_channel?starpu_cuda_get_local_out_transfer_stream():NULL,
				cudaMemcpyDeviceToHost);

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_CUDA_RAM):
		return starpu_cuda_copy_async_sync(
				(void*) src + src_offset, src_node,
				(void*) dst + dst_offset, dst_node,
				size,
				async_channel?starpu_cuda_get_local_in_transfer_stream():NULL,
				cudaMemcpyHostToDevice);

	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CUDA_RAM):
		return starpu_cuda_copy_async_sync(
				(void*) src + src_offset, src_node,
				(void*) dst + dst_offset, dst_node,
				size,
				async_channel?starpu_cuda_get_local_peer_transfer_stream():NULL,
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
					(void*) src + src_offset, src_node,
					(void*) dst + dst_offset, dst_node,
					size);
		else
			return _starpu_mic_copy_mic_to_ram(
					(void*) src + src_offset, src_node,
					(void*) dst + dst_offset, dst_node,
					size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_MIC_RAM):
		if (async_data)
			return _starpu_mic_copy_ram_to_mic_async(
					(void*) src + src_offset, src_node,
					(void*) dst + dst_offset, dst_node,
					size);
		else
			return _starpu_mic_copy_ram_to_mic(
					(void*) src + src_offset, src_node,
					(void*) dst + dst_offset, dst_node,
					size);
#endif
#ifdef STARPU_USE_SCC
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_SCC_RAM,STARPU_CPU_RAM):
		return _starpu_scc_copy_sink_to_src(
				(void*) src + src_offset, src_node,
				(void*) dst + dst_offset, dst_node,
				size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM,STARPU_SCC_RAM):
		return _starpu_scc_copy_src_to_sink(
				(void*) src + src_offset, src_node,
				(void*) dst + dst_offset, dst_node,
				size);
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_SCC_RAM,STARPU_SCC_RAM):
		return _starpu_scc_copy_sink_to_sink(
				(void*) src + src_offset, src_node,
				(void*) dst + dst_offset, dst_node,
				size);
#endif
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CPU_RAM, STARPU_DISK_RAM):
	{
		return _starpu_disk_copy_src_to_disk(
			(void*) src + src_offset, src_node,
			(void*) dst, dst_offset, dst_node,
			size, async_channel);
	}
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_DISK_RAM, STARPU_CPU_RAM):
		return _starpu_disk_copy_disk_to_src(
			(void*) src, src_offset, src_node,
			(void*) dst + dst_offset, dst_node,
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

void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_LOCK(&async_channel->event.mutex);
	while (!async_channel->event.finished)
		STARPU_PTHREAD_COND_WAIT(&async_channel->event.cond, &async_channel->event.mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&async_channel->event.mutex);
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
	case STARPU_MAIN_RAM:
		starpu_disk_wait_request(async_channel);
	case STARPU_CPU_RAM:
	default:
		STARPU_ABORT();
	}
#endif /* !SIMGRID */
}

unsigned _starpu_driver_test_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	unsigned ret;
	STARPU_PTHREAD_MUTEX_LOCK(&async_channel->event.mutex);
	ret = async_channel->event.finished;
	STARPU_PTHREAD_MUTEX_UNLOCK(&async_channel->event.mutex);
	return ret;
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
		success = (event_status == CL_COMPLETE);
		break;
	}
#endif
#ifdef STARPU_USE_MIC
	case STARPU_MIC_RAM:
		success = _starpu_mic_request_is_complete(&(async_channel->event.mic_event));
		break;
#endif
	case STARPU_DISK_RAM:
		success = starpu_disk_test_request(async_channel);
		break;
	case STARPU_CPU_RAM:
	default:
		STARPU_ABORT();
	}

	return success;
#endif /* !SIMGRID */
}
