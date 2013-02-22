/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include <common/fxt.h>
#include "copy_driver.h"
#include "memalloc.h"
#include <starpu_opencl.h>
#include <starpu_cuda.h>
#include <profiling/profiling.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#include <msg/msg.h>
#endif

void _starpu_wake_all_blocked_workers_on_node(unsigned nodeid)
{
	/* wake up all workers on that memory node */
	unsigned cond_id;

	struct _starpu_memory_node_descr * const descr = _starpu_memory_node_get_description();

	_STARPU_PTHREAD_RWLOCK_RDLOCK(&descr->conditions_rwlock);

	unsigned nconds = descr->condition_count[nodeid];
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		struct _starpu_cond_and_mutex *condition;
		condition  = &descr->conditions_attached_to_node[nodeid][cond_id];

		/* wake anybody waiting on that condition */
		_STARPU_PTHREAD_MUTEX_LOCK(condition->mutex);
		_STARPU_PTHREAD_COND_BROADCAST(condition->cond);
		_STARPU_PTHREAD_MUTEX_UNLOCK(condition->mutex);
	}

	_STARPU_PTHREAD_RWLOCK_UNLOCK(&descr->conditions_rwlock);
}

void starpu_wake_all_blocked_workers(void)
{
	/* workers may be blocked on the various queues' conditions */
	unsigned cond_id;

	struct _starpu_memory_node_descr * const descr = _starpu_memory_node_get_description();

	_STARPU_PTHREAD_RWLOCK_RDLOCK(&descr->conditions_rwlock);

	unsigned nconds = descr->total_condition_count;
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		struct _starpu_cond_and_mutex *condition;
		condition  = &descr->conditions_all[cond_id];

		/* wake anybody waiting on that condition */
		_STARPU_PTHREAD_MUTEX_LOCK(condition->mutex);
		_STARPU_PTHREAD_COND_BROADCAST(condition->cond);
		_STARPU_PTHREAD_MUTEX_UNLOCK(condition->mutex);
	}

	_STARPU_PTHREAD_RWLOCK_UNLOCK(&descr->conditions_rwlock);
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
				    struct _starpu_data_request *req STARPU_ATTRIBUTE_UNUSED)
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
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		break;
#ifdef STARPU_USE_CUDA
	case _STARPU_MEMORY_NODE_TUPLE(STARPU_CUDA_RAM,STARPU_CPU_RAM):
		/* only the proper CUBLAS thread can initiate this directly ! */
#if !defined(HAVE_CUDA_MEMCPY_PEER)
		STARPU_ASSERT(_starpu_memory_node_get_local_key() == src_node);
#endif
		if (!req || !(copy_methods->cuda_to_ram_async || copy_methods->any_to_any))
		{
			/* this is not associated to a request so it's synchronous */
			STARPU_ASSERT(copy_methods->cuda_to_ram);
			copy_methods->cuda_to_ram(src_interface, src_node, dst_interface, dst_node);
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
		if (!req || !(copy_methods->ram_to_cuda_async || copy_methods->any_to_any))
		{
			/* this is not associated to a request so it's synchronous */
			STARPU_ASSERT(copy_methods->ram_to_cuda);
			copy_methods->ram_to_cuda(src_interface, src_node, dst_interface, dst_node);
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
		if (!req || !(copy_methods->cuda_to_cuda_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->cuda_to_cuda);
			/* this is not associated to a request so it's synchronous */
			copy_methods->cuda_to_cuda(src_interface, src_node, dst_interface, dst_node);
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
		if (!req || !(copy_methods->opencl_to_ram_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->opencl_to_ram);
			/* this is not associated to a request so it's synchronous */
			copy_methods->opencl_to_ram(src_interface, src_node, dst_interface, dst_node);
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
		if (!req || !(copy_methods->ram_to_opencl_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->ram_to_opencl);
			/* this is not associated to a request so it's synchronous */
			copy_methods->ram_to_opencl(src_interface, src_node, dst_interface, dst_node);
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
		if (!req || !(copy_methods->opencl_to_opencl_async || copy_methods->any_to_any))
		{
			STARPU_ASSERT(copy_methods->opencl_to_opencl);
			/* this is not associated to a request so it's synchronous */
			copy_methods->opencl_to_opencl(src_interface, src_node, dst_interface, dst_node);
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
	default:
		STARPU_ABORT();
		break;
	}

	return ret;
#endif /* !SIMGRID */
}

int __attribute__((warn_unused_result)) _starpu_driver_copy_data_1_to_1(starpu_data_handle_t handle,
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

		ret_alloc = _starpu_allocate_memory_on_node(handle, dst_replicate,req->prefetch);
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

int starpu_interface_copy(uintptr_t src, unsigned src_node, size_t src_offset, uintptr_t dst, unsigned dst_node, size_t dst_offset, size_t size, void *async_data)
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
				src, src_node, src_offset,
				dst, dst_node, dst_offset,
				size,
				&async_channel->event.opencl_event);
#endif
	default:
		STARPU_ABORT();
		return -1;
	}
	return 0;
}

void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel)
{
#ifdef STARPU_SIMGRID
	_STARPU_PTHREAD_MUTEX_LOCK(&async_channel->event.mutex);
	while (!async_channel->event.finished)
		_STARPU_PTHREAD_COND_WAIT(&async_channel->event.cond, &async_channel->event.mutex);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&async_channel->event.mutex);
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
	_STARPU_PTHREAD_MUTEX_LOCK(&async_channel->event.mutex);
	ret = async_channel->event.finished;
	_STARPU_PTHREAD_MUTEX_UNLOCK(&async_channel->event.mutex);
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
	case STARPU_CPU_RAM:
	default:
		STARPU_ABORT();
	}

	return success;
#endif /* !SIMGRID */
}
