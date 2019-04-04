/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2019                                Université de Bordeaux
 * Copyright (C) 2011-2013,2016,2017                      Inria
 * Copyright (C) 2010,2011,2013,2015-2019                 CNRS
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
#include <drivers/mic/driver_mic_source.h>
#include <common/fxt.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memalloc.h>
#include <starpu_opencl.h>
#include <starpu_cuda.h>
#include <profiling/profiling.h>
#include <core/disk.h>
#include <datawizard/node_ops.h>

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
	unsigned src_node = (unsigned)src_replicate->memory_node;
	unsigned dst_node = (unsigned)dst_replicate->memory_node;

	STARPU_ASSERT(src_replicate->refcnt);
	STARPU_ASSERT(dst_replicate->refcnt);

	STARPU_ASSERT(src_replicate->allocated);
	STARPU_ASSERT(dst_replicate->allocated);

#ifdef STARPU_SIMGRID
	if (src_node == STARPU_MAIN_RAM || dst_node == STARPU_MAIN_RAM)
		_starpu_simgrid_data_transfer(handle->ops->get_size(handle), src_node, dst_node);

	return _starpu_simgrid_transfer(handle->ops->get_size(handle), src_node, dst_node, req);
#else /* !SIMGRID */
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
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

	if (_node_ops[src_kind].copy_data_to[dst_kind])
	{
		return _node_ops[src_kind].copy_data_to[dst_kind](handle, src_interface, src_node, dst_interface, dst_node, req);
	}
	else
	{
		STARPU_ABORT();
	}
#endif /* !SIMGRID */
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
		STARPU_ASSERT(src_replicate->allocated);
		STARPU_ASSERT(src_replicate->refcnt);
	}

	unsigned src_node = src_replicate->memory_node;
	unsigned dst_node = dst_replicate->memory_node;

	/* first make sure the destination has an allocated buffer */
	if (!dst_replicate->allocated)
	{
		if (!may_alloc || _starpu_is_reclaiming(dst_node))
			/* We're not supposed to allocate there at the moment */
			return -ENOMEM;

		int ret_alloc = _starpu_allocate_memory_on_node(handle, dst_replicate, req ? req->prefetch : 0);
		if (ret_alloc)
			return -ENOMEM;
	}

	STARPU_ASSERT(dst_replicate->allocated);
	STARPU_ASSERT(dst_replicate->refcnt);

	/* if there is no need to actually read the data,
	 * we do not perform any transfer */
	if (!donotread)
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

	if (_node_ops[src_kind].copy_interface_to[dst_kind])
	{
		return _node_ops[src_kind].copy_interface_to[dst_kind](src, src_offset, src_node,
								       dst, dst_offset, dst_node,
								       size,
								       async_channel);
	}
	else
	{
		STARPU_ABORT();
		return -1;
	}
}

void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	_starpu_simgrid_wait_transfer_event(&async_channel->event);
#else /* !SIMGRID */
	enum starpu_node_kind kind = async_channel->type;

	if (_node_ops[kind].wait_request_completion != NULL)
	{
		_node_ops[kind].wait_request_completion(async_channel);
	}
	else
	{
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

	if (_node_ops[kind].test_request_completion != NULL)
	{
		return _node_ops[kind].test_request_completion(async_channel);
	}
	else
	{
		STARPU_ABORT_MSG("Memory is not recognized (kind %d) \n", kind);
	}
#endif /* !SIMGRID */
}
