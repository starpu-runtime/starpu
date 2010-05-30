/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <pthread.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/policies/sched_policy.h>
#include <datawizard/datastats.h>
#include <common/fxt.h>
#include "copy_driver.h"
#include "memalloc.h"

void _starpu_wake_all_blocked_workers_on_node(unsigned nodeid)
{
	/* wake up all queues on that node */
	unsigned q_id;

	starpu_mem_node_descr * const descr = _starpu_get_memory_node_description();

	PTHREAD_RWLOCK_RDLOCK(&descr->attached_queues_rwlock);

	unsigned nqueues = descr->queues_count[nodeid];
	for (q_id = 0; q_id < nqueues; q_id++)
	{
		struct starpu_jobq_s *q;
		q  = descr->attached_queues_per_node[nodeid][q_id];

		/* wake anybody waiting on that queue */
		PTHREAD_MUTEX_LOCK(&q->activity_mutex);
		PTHREAD_COND_BROADCAST(&q->activity_cond);
		PTHREAD_MUTEX_UNLOCK(&q->activity_mutex);
	}

	PTHREAD_RWLOCK_UNLOCK(&descr->attached_queues_rwlock);
}

void starpu_wake_all_blocked_workers(void)
{
	/* workers may be blocked on the policy's global condition */
	struct starpu_sched_policy_s *sched = _starpu_get_sched_policy();
	pthread_cond_t *sched_cond = &sched->sched_activity_cond;
	pthread_mutex_t *sched_mutex = &sched->sched_activity_mutex;

	PTHREAD_MUTEX_LOCK(sched_mutex);
	PTHREAD_COND_BROADCAST(sched_cond);
	PTHREAD_MUTEX_UNLOCK(sched_mutex);

	/* workers may be blocked on the various queues' conditions */
	unsigned node;
	unsigned nnodes = _starpu_get_memory_nodes_count();
	for (node = 0; node < nnodes; node++)
	{
		_starpu_wake_all_blocked_workers_on_node(node);
	}
}

#ifdef STARPU_USE_FXT
/* we need to identify each communication so that we can match the beginning
 * and the end of a communication in the trace, so we use a unique identifier
 * per communication */
static unsigned communication_cnt = 0;
#endif

static int copy_data_1_to_1_generic(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, struct starpu_data_request_s *req __attribute__((unused)))
{
	int ret = 0;

	//ret = handle->ops->copy_data_1_to_1(handle, src_node, dst_node);

	const struct starpu_copy_data_methods_s *copy_methods = handle->ops->copy_methods;

	starpu_node_kind src_kind = _starpu_get_node_kind(src_node);
	starpu_node_kind dst_kind = _starpu_get_node_kind(dst_node);

	STARPU_ASSERT(handle->per_node[src_node].refcnt);
	STARPU_ASSERT(handle->per_node[dst_node].refcnt);

	STARPU_ASSERT(handle->per_node[src_node].allocated);
	STARPU_ASSERT(handle->per_node[dst_node].allocated);

#ifdef STARPU_USE_CUDA
cudaError_t cures;
cudaStream_t *stream;
#endif

	switch (dst_kind) {
	case STARPU_RAM:
		switch (src_kind) {
			case STARPU_RAM:
				/* STARPU_RAM -> STARPU_RAM */
				STARPU_ASSERT(copy_methods->ram_to_ram);
				copy_methods->ram_to_ram(handle, src_node, dst_node);
				break;
#ifdef STARPU_USE_CUDA
			case STARPU_CUDA_RAM:
				/* CUBLAS_RAM -> STARPU_RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				if (_starpu_get_local_memory_node() == src_node)
				{
					/* only the proper CUBLAS thread can initiate this directly ! */
					STARPU_ASSERT(copy_methods->cuda_to_ram);
					if (!req || !copy_methods->cuda_to_ram_async)
					{
						/* this is not associated to a request so it's synchronous */
						copy_methods->cuda_to_ram(handle, src_node, dst_node);
					}
					else {
						cures = cudaEventCreate(&req->async_channel.cuda_event);
						if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

						stream = starpu_cuda_get_local_stream();
						ret = copy_methods->cuda_to_ram_async(handle, src_node, dst_node, stream);

						cures = cudaEventRecord(req->async_channel.cuda_event, *stream);
						if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);
					}
				}
				else
				{
					/* we should not have a blocking call ! */
					STARPU_ABORT();
				}
				break;
#endif
#ifdef STARPU_USE_OPENCL
    		        case STARPU_OPENCL_RAM:
				/* OpenCL -> RAM */
				if (_starpu_get_local_memory_node() == src_node)
				{
					STARPU_ASSERT(copy_methods->opencl_to_ram);
					if (!req || !copy_methods->opencl_to_ram_async)
					{
						/* this is not associated to a request so it's synchronous */
                                                copy_methods->opencl_to_ram(handle, src_node, dst_node);
                                        }
                                        else {
                                                ret = copy_methods->opencl_to_ram_async(handle, src_node, dst_node, &(req->async_channel.opencl_event));
                                        }
				}
				else
				{
					/* we should not have a blocking call ! */
					STARPU_ABORT();
				}
				break;
#endif
			case STARPU_SPU_LS:
				STARPU_ABORT(); // TODO
				break;
			case STARPU_UNUSED:
				printf("error node %u STARPU_UNUSED\n", src_node);
			default:
				assert(0);
				break;
		}
		break;
#ifdef STARPU_USE_CUDA
	case STARPU_CUDA_RAM:
		switch (src_kind) {
			case STARPU_RAM:
				/* STARPU_RAM -> CUBLAS_RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				STARPU_ASSERT(_starpu_get_local_memory_node() == dst_node);
				STARPU_ASSERT(copy_methods->ram_to_cuda);
				if (!req || !copy_methods->ram_to_cuda_async)
				{
					/* this is not associated to a request so it's synchronous */
					copy_methods->ram_to_cuda(handle, src_node, dst_node);
				}
				else {
					cures = cudaEventCreate(&req->async_channel.cuda_event);
					if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

					stream = starpu_cuda_get_local_stream();
					ret = copy_methods->ram_to_cuda_async(handle, src_node, dst_node, stream);

					cures = cudaEventRecord(req->async_channel.cuda_event, *stream);
					if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);
				}
				break;
			case STARPU_CUDA_RAM:
			case STARPU_SPU_LS:
				STARPU_ABORT(); // TODO 
				break;
			case STARPU_UNUSED:
			default:
				STARPU_ABORT();
				break;
		}
		break;
#endif
#ifdef STARPU_USE_OPENCL
	case STARPU_OPENCL_RAM:
		switch (src_kind) {
		        case STARPU_RAM:
				/* STARPU_RAM -> STARPU_OPENCL_RAM */
				STARPU_ASSERT(_starpu_get_local_memory_node() == dst_node);
				STARPU_ASSERT(copy_methods->ram_to_opencl);
				if (!req || !copy_methods->ram_to_opencl_async)
				{
					/* this is not associated to a request so it's synchronous */
					copy_methods->ram_to_opencl(handle, src_node, dst_node);
				}
				else {
                                        ret = copy_methods->ram_to_opencl_async(handle, src_node, dst_node, &(req->async_channel.opencl_event));
				}
				break;
			case STARPU_CUDA_RAM:
			case STARPU_OPENCL_RAM:
			case STARPU_SPU_LS:
				STARPU_ABORT(); // TODO 
				break;
			case STARPU_UNUSED:
			default:
				STARPU_ABORT();
				break;
		}
		break;
#endif
	case STARPU_SPU_LS:
		STARPU_ABORT(); // TODO
		break;
	case STARPU_UNUSED:
	default:
		assert(0);
		break;
	}

	return ret;
}

int __attribute__((warn_unused_result)) _starpu_driver_copy_data_1_to_1(starpu_data_handle handle, uint32_t src_node, 
		uint32_t dst_node, unsigned donotread, struct starpu_data_request_s *req, unsigned may_alloc)
{
	if (!donotread)
	{
		STARPU_ASSERT(handle->per_node[src_node].allocated);
		STARPU_ASSERT(handle->per_node[src_node].refcnt);
	}

	int ret_alloc, ret_copy;
	unsigned __attribute__((unused)) com_id = 0;

	/* first make sure the destination has an allocated buffer */
	ret_alloc = _starpu_allocate_memory_on_node(handle, dst_node, may_alloc);
	if (ret_alloc)
		goto nomem;

	STARPU_ASSERT(handle->per_node[dst_node].allocated);
	STARPU_ASSERT(handle->per_node[dst_node].refcnt);

	/* if there is no need to actually read the data, 
	 * we do not perform any transfer */
	if (!donotread) {
		STARPU_ASSERT(handle->ops);
		//STARPU_ASSERT(handle->ops->copy_data_1_to_1);

#ifdef STARPU_DATA_STATS
		size_t size = handle->ops->get_size(handle);
		_starpu_update_comm_amount(src_node, dst_node, size);
#endif
		
#ifdef STARPU_USE_FXT
		com_id = STARPU_ATOMIC_ADD(&communication_cnt, 1);

		if (req)
			req->com_id = com_id;
#endif

		/* for now we set the size to 0 in the FxT trace XXX */
		STARPU_TRACE_START_DRIVER_COPY(src_node, dst_node, 0, com_id);
		ret_copy = copy_data_1_to_1_generic(handle, src_node, dst_node, req);

#ifdef STARPU_USE_FXT
		if (ret_copy != EAGAIN)
		{
			size_t size = handle->ops->get_size(handle);
			STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, com_id);
		}
#endif

		return ret_copy;
	}

	return 0;

nomem:
	return ENOMEM;
}

void _starpu_driver_wait_request_completion(starpu_async_channel *async_channel __attribute__ ((unused)),
					unsigned handling_node)
{
	starpu_node_kind kind = _starpu_get_node_kind(handling_node);
#ifdef STARPU_USE_CUDA
	cudaEvent_t event;
	cudaError_t cures;
#endif

	switch (kind) {
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			event = (*async_channel).cuda_event;

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
                        fprintf(stderr, "not implemented yet\n");
			STARPU_ABORT();
                        break;
#endif
		case STARPU_RAM:
		default:
			STARPU_ABORT();
	}
}

unsigned _starpu_driver_test_request_completion(starpu_async_channel *async_channel __attribute__ ((unused)),
					unsigned handling_node)
{
	starpu_node_kind kind = _starpu_get_node_kind(handling_node);
	unsigned success;
#ifdef STARPU_USE_CUDA
	cudaEvent_t event;
#endif

	switch (kind) {
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			event = (*async_channel).cuda_event;

			success = (cudaEventQuery(event) == cudaSuccess);
			if (success)
				cudaEventDestroy(event);

			break;
#endif
#ifdef STARPU_USE_OPENCL
                case STARPU_OPENCL_RAM:
                        {
                                cl_int event_status;
                                cl_event opencl_event = (*async_channel).opencl_event;
                                if (opencl_event == NULL) STARPU_ABORT();
                                clGetEventInfo(opencl_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
                                success = (event_status == CL_COMPLETE);
                                break;
                        }
#endif
		case STARPU_RAM:
		default:
			STARPU_ABORT();
			success = 0;
	}

	return success;
}
