/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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
#include <drivers/tcpip/driver_tcpip_sink.h>
#include <drivers/tcpip/driver_tcpip_source.h>
#include <drivers/tcpip/driver_tcpip_common.h>
#include <common/fxt.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memalloc.h>
#include <profiling/profiling.h>

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

int _starpu_copy_interface_any_to_any(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (!req || starpu_asynchronous_copy_disabled() ||
		    starpu_asynchronous_copy_disabled_for(src_kind) ||
		    starpu_asynchronous_copy_disabled_for(dst_kind) ||
		    !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		if (dst_kind == STARPU_CPU_RAM)
			req->async_channel.node_ops = starpu_memory_driver_info[src_kind].ops;
		else
			req->async_channel.node_ops = starpu_memory_driver_info[dst_kind].ops;
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
	}
	return ret;
}

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

	const struct _starpu_node_ops *src_node_ops = _starpu_memory_node_get_node_ops(src_node);
	const struct _starpu_node_ops *dst_node_ops = _starpu_memory_node_get_node_ops(dst_node);
	if (src_node_ops && src_node_ops->copy_interface_to[dst_kind])
	{
		return src_node_ops->copy_interface_to[dst_kind](handle, src_interface, src_node, dst_interface, dst_node, req);
	}
	else if (dst_node_ops && dst_node_ops->copy_interface_from[src_kind])
	{
		return dst_node_ops->copy_interface_from[src_kind](handle, src_interface, src_node, dst_interface, dst_node, req);
	}
	else
	{
		STARPU_ABORT_MSG("No copy_interface_to function defined from node %s to node %s\n", _starpu_node_get_prefix(starpu_node_get_kind(src_node)), _starpu_node_get_prefix(starpu_node_get_kind(dst_node)));
	}
#endif /* !SIMGRID */
}

static int update_map_generic(starpu_data_handle_t handle,
				    struct _starpu_data_replicate *src_replicate,
				    struct _starpu_data_replicate *dst_replicate,
				    struct _starpu_data_request *req STARPU_ATTRIBUTE_UNUSED)
{
	int src_node = src_replicate->memory_node;
	int dst_node = dst_replicate->memory_node;

	STARPU_ASSERT(src_replicate->refcnt);
	STARPU_ASSERT(dst_replicate->refcnt);

	STARPU_ASSERT((src_replicate->mapped == dst_node && dst_replicate->allocated)
			||(src_replicate->allocated && dst_replicate->mapped == src_node));

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
									enum _starpu_may_alloc may_alloc,
									enum starpu_is_prefetch prefetch STARPU_ATTRIBUTE_UNUSED)
{
	if (!donotread)
	{
		STARPU_ASSERT(src_replicate->allocated || src_replicate->mapped != STARPU_UNMAPPED);
		STARPU_ASSERT(src_replicate->refcnt);
	}

	unsigned src_node = src_replicate->memory_node;
	unsigned dst_node = dst_replicate->memory_node;

	if (!dst_replicate->allocated && dst_replicate->mapped == STARPU_UNMAPPED && dst_node != src_node
			&& handle->ops->map_data
			&& (_starpu_memory_node_get_mapped(dst_replicate->memory_node) /* || handle wants it */))
	{
		/* Memory node which can just map the main memory, try to map.  */
		if (!handle->ops->map_data(
				src_replicate->data_interface, src_replicate->memory_node,
				dst_replicate->data_interface, dst_replicate->memory_node))
		{
			dst_replicate->mapped = src_node;

			if (_starpu_node_needs_map_update(dst_node))
			{
				/* Driver porters: adding your driver here is
				   optional, it is only needed when implementing
				   support for memory mapping */
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
	if (!dst_replicate->allocated && dst_replicate->mapped == STARPU_UNMAPPED)
	{
		if (may_alloc==_STARPU_DATAWIZARD_DO_NOT_ALLOC || _starpu_is_reclaiming(dst_node))
			/* We're not supposed to allocate there at the moment */
			return -ENOMEM;

		int ret_alloc = _starpu_allocate_memory_on_node(handle, dst_replicate, prefetch, may_alloc==_STARPU_DATAWIZARD_ONLY_FAST_ALLOC);
		if (ret_alloc)
			return -ENOMEM;
	}

	STARPU_ASSERT(dst_replicate->allocated || dst_replicate->mapped != STARPU_UNMAPPED);
	STARPU_ASSERT(dst_replicate->refcnt);

	/* In the case of a mapped data, we are here requested either
	 * - because the destination will write to it, and thus needs write
	 *   access.
	 * - because the source was modified, and the destination needs to get
	 *   updated.
	 * All in all, any data change will actually trigger both.
	 */
	if (!donotread && dst_replicate->mapped != STARPU_UNMAPPED)
	{
		STARPU_ASSERT(src_replicate->memory_node == dst_replicate->mapped);
		if (_starpu_node_needs_map_update(dst_node))
		{
			/* We need to flush from RAM to the device */
			if (!dst_replicate->map_write)
			{
				update_map_generic(handle, src_replicate, dst_replicate, req);
				dst_replicate->map_write = 1;
			}
		}

		dst_replicate->initialized = 1;
	}

	else if (!donotread && src_replicate->mapped != STARPU_UNMAPPED)
	{
		STARPU_ASSERT(dst_replicate->memory_node == src_replicate->mapped);
		if (_starpu_node_needs_map_update(src_node))
		{
			/* We need to flush from the device to the RAM */
			if (src_replicate->map_write)
			{
				update_map_generic(handle, src_replicate, dst_replicate, req);
				src_replicate->map_write = 0;
			}
		}

		dst_replicate->initialized = 1;
	}

	/* if there is no need to actually read the data,
	 * we do not perform any transfer */
	else if (!donotread)
	{
		unsigned long STARPU_ATTRIBUTE_UNUSED com_id = 0;
		size_t size = _starpu_data_get_size(handle);
		_starpu_bus_update_profiling_info((int)src_node, (int)dst_node, size);

#ifdef STARPU_USE_FXT
		if (fut_active)
		{
			com_id = STARPU_ATOMIC_ADDL(&communication_cnt, 1);

			if (req)
				req->com_id = com_id;
		}
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

void starpu_interface_data_copy(unsigned src_node, unsigned dst_node, size_t size)
{
	_STARPU_TRACE_DATA_COPY(src_node, dst_node, size);
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
		STARPU_HG_DISABLE_CHECKING(warned);
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
	const struct _starpu_node_ops *src_node_ops = _starpu_memory_node_get_node_ops(src_node);
	const struct _starpu_node_ops *dst_node_ops = _starpu_memory_node_get_node_ops(dst_node);

	if (src_node_ops && src_node_ops->copy_data_to[dst_kind])
	{
		return src_node_ops->copy_data_to[dst_kind](src, src_offset, src_node,
							     dst, dst_offset, dst_node,
							     size,
							     async_channel);
	}
	else if (dst_node_ops && dst_node_ops->copy_data_from[src_kind])
	{
		return dst_node_ops->copy_data_from[src_kind](src, src_offset, src_node,
							     dst, dst_offset, dst_node,
							     size,
							     async_channel);
	}
	else
	{
		STARPU_ABORT_MSG("No copy_data_to function defined from node %s to node %s\n", _starpu_node_get_prefix(starpu_node_get_kind(src_node)), _starpu_node_get_prefix(starpu_node_get_kind(dst_node)));
		return -1;
	}
}

int starpu_interface_copy2d(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t blocksize,
			    size_t numblocks, size_t ld_src, size_t ld_dst,
			    void *async_data)
{
	int ret = 0;
	unsigned i;
	struct _starpu_async_channel *async_channel = async_data;
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	const struct _starpu_node_ops *src_node_ops = _starpu_memory_node_get_node_ops(src_node);
	const struct _starpu_node_ops *dst_node_ops = _starpu_memory_node_get_node_ops(dst_node);

	STARPU_ASSERT_MSG(ld_src >= blocksize, "block size %lu is bigger than ld %lu in source", (unsigned long) blocksize, (unsigned long) ld_src);
	STARPU_ASSERT_MSG(ld_dst >= blocksize, "block size %lu is bigger than ld %lu in destination", (unsigned long) blocksize, (unsigned long) ld_dst);

	if (ld_src == blocksize && ld_dst == blocksize)
		/* Optimize contiguous case */
		return starpu_interface_copy(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     blocksize * numblocks, async_data);

	if (src_node_ops && src_node_ops->copy2d_data_to[dst_kind])
		/* Hardware-optimized non-contiguous case */
		return src_node_ops->copy2d_data_to[dst_kind](src, src_offset, src_node,
							     dst, dst_offset, dst_node,
							     blocksize,
							     numblocks, ld_src, ld_dst,
							     async_channel);

	if (dst_node_ops && dst_node_ops->copy2d_data_from[src_kind])
		/* Hardware-optimized non-contiguous case */
		return dst_node_ops->copy2d_data_from[src_kind](src, src_offset, src_node,
							     dst, dst_offset, dst_node,
							     blocksize,
							     numblocks, ld_src, ld_dst,
							     async_channel);

	for (i = 0; i < numblocks; i++)
	{
		if (starpu_interface_copy(src, src_offset + i*ld_src, src_node,
					  dst, dst_offset + i*ld_dst, dst_node,
					  blocksize, async_data))
			ret = -EAGAIN;
	}

	return ret;
}

int starpu_interface_copy3d(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t blocksize,
			    size_t numblocks_1, size_t ld1_src, size_t ld1_dst,
			    size_t numblocks_2, size_t ld2_src, size_t ld2_dst,
			    void *async_data)
{
	int ret = 0;
	unsigned i;
	struct _starpu_async_channel *async_channel = async_data;
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	const struct _starpu_node_ops *src_node_ops = _starpu_memory_node_get_node_ops(src_node);
	const struct _starpu_node_ops *dst_node_ops = _starpu_memory_node_get_node_ops(dst_node);

	STARPU_ASSERT_MSG(ld1_src >= blocksize, "block size %lu is bigger than ld %lu in source", (unsigned long) blocksize, (unsigned long) ld1_src);
	STARPU_ASSERT_MSG(ld1_dst >= blocksize, "block size %lu is bigger than ld %lu in destination", (unsigned long) blocksize, (unsigned long) ld1_dst);

	STARPU_ASSERT_MSG(ld2_src >= numblocks_1 * ld1_src, "block group size %lu is bigger than group ld %lu in source", (unsigned long) (numblocks_1 * ld1_src), (unsigned long) ld2_src);
	STARPU_ASSERT_MSG(ld2_dst >= numblocks_1 * ld1_dst, "block group size %lu is bigger than group ld %lu in destination", (unsigned long) (numblocks_1 * ld1_dst), (unsigned long) ld2_dst);

	if (ld2_src == blocksize * numblocks_1 &&
	    ld2_dst == blocksize * numblocks_1)
		/* Optimize contiguous case */
		return starpu_interface_copy(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     blocksize * numblocks_1 * numblocks_2,
					     async_data);

	if (src_node_ops && src_node_ops->copy3d_data_to[dst_kind])
		/* Hardware-optimized non-contiguous case */
		return src_node_ops->copy3d_data_to[dst_kind](src, src_offset, src_node,
							     dst, dst_offset, dst_node,
							     blocksize,
							     numblocks_1, ld1_src, ld1_dst,
							     numblocks_2, ld2_src, ld2_dst,
							     async_channel);

	if (dst_node_ops && dst_node_ops->copy3d_data_from[src_kind])
		/* Hardware-optimized non-contiguous case */
		return dst_node_ops->copy3d_data_from[src_kind](src, src_offset, src_node,
							     dst, dst_offset, dst_node,
							     blocksize,
							     numblocks_1, ld1_src, ld1_dst,
							     numblocks_2, ld2_src, ld2_dst,
							     async_channel);


	for (i = 0; i < numblocks_2; i++)
	{
		if (starpu_interface_copy2d(src, src_offset + i*ld2_src, src_node,
					    dst, dst_offset + i*ld2_dst, dst_node,
					    blocksize, numblocks_1, ld1_src, ld1_dst,
					    async_data))
			ret = -EAGAIN;
	}

	return ret;
}

int starpu_interface_copy4d(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t blocksize,
			    size_t numblocks_1, size_t ld1_src, size_t ld1_dst,
			    size_t numblocks_2, size_t ld2_src, size_t ld2_dst,
			    size_t numblocks_3, size_t ld3_src, size_t ld3_dst,
			    void *async_data)
{
	int ret = 0;
	unsigned i;

	STARPU_ASSERT_MSG(ld1_src >= blocksize, "block size %lu is bigger than ld %lu in source", (unsigned long) blocksize, (unsigned long) ld1_src);
	STARPU_ASSERT_MSG(ld1_dst >= blocksize, "block size %lu is bigger than ld %lu in destination", (unsigned long) blocksize, (unsigned long) ld1_dst);

	STARPU_ASSERT_MSG(ld2_src >= numblocks_1 * ld1_src, "block group size %lu is bigger than group ld %lu in source", (unsigned long) (numblocks_1 * ld1_src), (unsigned long) ld2_src);
	STARPU_ASSERT_MSG(ld2_dst >= numblocks_1 * ld1_dst, "block group size %lu is bigger than group ld %lu in destination", (unsigned long) (numblocks_1 * ld1_dst), (unsigned long) ld2_dst);

	STARPU_ASSERT_MSG(ld3_src >= numblocks_2 * ld2_src, "block group group size %lu is bigger than group group ld %lu in source", (unsigned long) (numblocks_2 * ld2_src), (unsigned long) ld3_src);
	STARPU_ASSERT_MSG(ld3_dst >= numblocks_2 * ld2_dst, "block group group size %lu is bigger than group group ld %lu in destination", (unsigned long) (numblocks_2 * ld2_dst), (unsigned long) ld3_dst);

	if (ld3_src == blocksize * numblocks_1 * numblocks_2 &&
	    ld3_dst == blocksize * numblocks_1 * numblocks_2)
		/* Optimize contiguous case */
		return starpu_interface_copy(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     blocksize * numblocks_1 * numblocks_2 * numblocks_3,
					     async_data);

	/* Probably won't ever have a 4D interface in drivers :) */

	for (i = 0; i < numblocks_3; i++)
	{
		if (starpu_interface_copy3d(src, src_offset + i*ld3_src, src_node,
					    dst, dst_offset + i*ld3_dst, dst_node,
					    blocksize,
					    numblocks_1, ld1_src, ld1_dst,
					    numblocks_2, ld2_src, ld2_dst,
					    async_data))
			ret = -EAGAIN;
	}

	return ret;
}

uintptr_t starpu_interface_map(uintptr_t src, size_t src_offset, unsigned src_node, unsigned dst_node, size_t size, int *ret)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	const struct _starpu_node_ops *node_ops = _starpu_memory_node_get_node_ops(dst_node);

	if (node_ops && node_ops->map[src_kind])
	{
		return node_ops->map[src_kind](src, src_offset, src_node, dst_node, size, ret);
	}
	else
	{
		*ret = -EIO;
		return 0;
	}
}

int starpu_interface_unmap(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, unsigned dst_node, size_t size)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	const struct _starpu_node_ops *node_ops = _starpu_memory_node_get_node_ops(dst_node);

	if (node_ops && node_ops->unmap[src_kind])
	{
		return node_ops->unmap[src_kind](src, src_offset, src_node, dst, dst_node, size);
	}
	else
	{
		STARPU_ABORT_MSG("No unmap function defined from node %s to node %s\n", _starpu_node_get_prefix(starpu_node_get_kind(src_node)), _starpu_node_get_prefix(starpu_node_get_kind(dst_node)));
		return -1;
	}
}

int starpu_interface_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	const struct _starpu_node_ops *src_node_ops = _starpu_memory_node_get_node_ops(src_node);
	const struct _starpu_node_ops *dst_node_ops = _starpu_memory_node_get_node_ops(dst_node);

	if (src_node_ops && src_node_ops->update_map[dst_kind])
	{
		return src_node_ops->update_map[dst_kind](src, src_offset, src_node, dst, dst_offset, dst_node, size);
	}
	else if (dst_node_ops && dst_node_ops->update_map[src_kind])
	{
		return dst_node_ops->update_map[src_kind](src, src_offset, src_node, dst, dst_offset, dst_node, size);
	}
	else
	{
		STARPU_ABORT_MSG("No unmap function defined from node %s to node %s\n", _starpu_node_get_prefix(starpu_node_get_kind(src_node)), _starpu_node_get_prefix(starpu_node_get_kind(dst_node)));
		return -1;
	}
}

static size_t _get_size(uint32_t* nn, size_t ndim)
{
	size_t size = 1;
	unsigned i;
	for (i=0; i<ndim; i++)
		size *= nn[i];

	return size;
}

int starpu_interface_copynd(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t elemsize, size_t ndim,
			    uint32_t* nn, uint32_t* ldn_src, uint32_t* ldn_dst,
			    void *async_data)
{
	int ret = 0;
	unsigned i;

	if (ndim > 0)
	{
		for (i = 0; i < ndim-1; i++)
		{
			STARPU_ASSERT_MSG(ldn_src[i+1] >= nn[i] * ldn_src[i], "block size %lu is bigger than ld %lu in source", (unsigned long) nn[i] * ldn_src[i], (unsigned long) ldn_src[i+1]);
			STARPU_ASSERT_MSG(ldn_dst[i+1] >= nn[i] * ldn_dst[i], "block size %lu is bigger than ld %lu in destination", (unsigned long) nn[i] * ldn_dst[i], (unsigned long) ldn_dst[i+1]);
		}

		if (ldn_src[ndim-1] == _get_size(nn, ndim-1) &&
		    ldn_dst[ndim-1] == _get_size(nn, ndim-1))
			/* Optimize contiguous case */
			return starpu_interface_copy(src, src_offset, src_node,
						     dst, dst_offset, dst_node,
						     _get_size(nn, ndim) * elemsize,
						     async_data);
	}

	if(ndim > 4)
	{
		for (i = 0; i < nn[ndim-1]; i++)
		{
			if (starpu_interface_copynd(src, src_offset + i*ldn_src[ndim-1]*elemsize, src_node,
						    dst, dst_offset + i*ldn_dst[ndim-1]*elemsize, dst_node,
						    elemsize, ndim-1,
						    nn, ldn_src, ldn_dst,
						    async_data))
				ret = -EAGAIN;
		}
	}
	else if(ndim == 4)
	{
		return starpu_interface_copy4d(src, src_offset, src_node,
				    dst, dst_offset, dst_node,
				    nn[0] * elemsize,
				    nn[1], ldn_src[1] * elemsize, ldn_dst[1] * elemsize,
				    nn[2], ldn_src[2] * elemsize, ldn_dst[2] * elemsize,
				    nn[3], ldn_src[3] * elemsize, ldn_dst[3] * elemsize,
				    async_data);
	}
	else if(ndim == 3)
	{
		return starpu_interface_copy3d(src, src_offset, src_node,
				    dst, dst_offset, dst_node,
				    nn[0] * elemsize,
				    nn[1], ldn_src[1] * elemsize, ldn_dst[1] * elemsize,
				    nn[2], ldn_src[2] * elemsize, ldn_dst[2] * elemsize,
				    async_data);
	}
	else if(ndim == 2)
	{
		return starpu_interface_copy2d(src, src_offset, src_node,
				    dst, dst_offset, dst_node,
				    nn[0] * elemsize,
				    nn[1], ldn_src[1] * elemsize, ldn_dst[1] * elemsize,
				    async_data);
	}
	else if (ndim == 1)
	{
		return starpu_interface_copy(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     nn[0] * elemsize,
					     async_data);
	}
	else if (ndim == 0)
	{
		return starpu_interface_copy(src, 0, src_node,
					     dst, 0, dst_node,
					     elemsize,
					     async_data);
	}

	return ret;
}

void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	_starpu_simgrid_wait_transfer_event(&async_channel->event);
#else /* !SIMGRID */
	const struct _starpu_node_ops *node_ops = async_channel->node_ops;
	if (node_ops && node_ops->wait_request_completion != NULL)
	{
		node_ops->wait_request_completion(async_channel);
	}
	else
	{
		STARPU_ABORT_MSG("No wait_request_completion function defined for node %s\n", node_ops?node_ops->name:"unknown");
	}
#endif /* !SIMGRID */
}

unsigned _starpu_driver_test_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	return _starpu_simgrid_test_transfer_event(&async_channel->event);
#else /* !SIMGRID */
	const struct _starpu_node_ops *node_ops = async_channel->node_ops;
	if (node_ops && node_ops->test_request_completion != NULL)
	{
		return node_ops->test_request_completion(async_channel);
	}
	else
	{
		STARPU_ABORT_MSG("No test_request_completion function defined for node %s\n", node_ops?node_ops->name:"unknown");
	}
#endif /* !SIMGRID */
}
