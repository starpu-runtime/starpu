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
#include <core/policies/sched_policy.h>
#include <datawizard/datastats.h>
#include <common/fxt.h>
#include "copy-driver.h"
#include "memalloc.h"

void wake_all_blocked_workers_on_node(unsigned nodeid)
{
	/* wake up all queues on that node */
	unsigned q_id;

	mem_node_descr * const descr = get_memory_node_description();

	pthread_rwlock_rdlock(&descr->attached_queues_rwlock);

	unsigned nqueues = descr->queues_count[nodeid];
	for (q_id = 0; q_id < nqueues; q_id++)
	{
		struct jobq_s *q;
		q  = descr->attached_queues_per_node[nodeid][q_id];

		/* wake anybody waiting on that queue */
		pthread_mutex_lock(&q->activity_mutex);
		pthread_cond_broadcast(&q->activity_cond);
		pthread_mutex_unlock(&q->activity_mutex);
	}

	pthread_rwlock_unlock(&descr->attached_queues_rwlock);
}

void wake_all_blocked_workers(void)
{
	/* workers may be blocked on the policy's global condition */
	struct sched_policy_s *sched = get_sched_policy();
	pthread_cond_t *sched_cond = &sched->sched_activity_cond;
	pthread_mutex_t *sched_mutex = &sched->sched_activity_mutex;
	mem_node_descr * const descr = get_memory_node_description();

	pthread_mutex_lock(sched_mutex);
	pthread_cond_broadcast(sched_cond);
	pthread_mutex_unlock(sched_mutex);

	/* workers may be blocked on the various queues' conditions */
	unsigned node;
	unsigned nnodes =  descr->nnodes;
	for (node = 0; node < nnodes; node++)
	{
		wake_all_blocked_workers_on_node(node);
	}
}

int allocate_per_node_buffer(data_state *state, uint32_t node)
{
	int ret;

	if (!state->per_node[node].allocated) {
		/* there is no room available for the data yet */
		ret = allocate_memory_on_node(state, node);
		if (STARPU_UNLIKELY(ret == -ENOMEM))
			goto nomem;
	}

	return 0;
nomem:
	/* there was not enough memory to allocate the buffer */
	return -ENOMEM;
}

#ifdef USE_FXT
/* we need to identify each communication so that we can match the beginning
 * and the end of a communication in the trace, so we use a unique identifier
 * per communication */
static unsigned communication_cnt = 0;
#endif

static int copy_data_1_to_1_generic(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	int ret;

	//ret = state->ops->copy_data_1_to_1(state, src_node, dst_node);

	const struct copy_data_methods_s *copy_methods = state->ops->copy_methods;

	node_kind src_kind = get_node_kind(src_node);
	node_kind dst_kind = get_node_kind(dst_node);

	switch (dst_kind) {
	case RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> RAM */
				STARPU_ASSERT(copy_methods->ram_to_ram);
				copy_methods->ram_to_ram(state, src_node, dst_node);
				break;
#ifdef USE_CUDA
			case CUDA_RAM:
				/* CUBLAS_RAM -> RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				if (get_local_memory_node() == src_node)
				{
					/* only the proper CUBLAS thread can initiate this directly ! */
					STARPU_ASSERT(copy_methods->cuda_to_ram);
					copy_methods->cuda_to_ram(state, src_node, dst_node);
				}
				else
				{
					/* put a request to the corresponding GPU */
					post_data_request(state, src_node, dst_node);
				}
				break;
#endif
			case SPU_LS:
				STARPU_ASSERT(0); // TODO
				break;
			case UNUSED:
				printf("error node %d UNUSED\n", src_node);
			default:
				assert(0);
				break;
		}
		break;
#ifdef USE_CUDA
	case CUDA_RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> CUBLAS_RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				STARPU_ASSERT(get_local_memory_node() == dst_node);
				STARPU_ASSERT(copy_methods->ram_to_cuda);
				copy_methods->ram_to_cuda(state, src_node, dst_node);
				break;
			case CUDA_RAM:
			case SPU_LS:
				STARPU_ASSERT(0); // TODO 
				break;
			case UNUSED:
			default:
				STARPU_ASSERT(0);
				break;
		}
		break;
#endif
	case SPU_LS:
		STARPU_ASSERT(0); // TODO
		break;
	case UNUSED:
	default:
		assert(0);
		break;
	}

	/* XXX */
	ret = 0;

	return ret;
}

int __attribute__((warn_unused_result)) driver_copy_data_1_to_1(data_state *state, uint32_t src_node, 
				uint32_t dst_node, unsigned donotread)
{
	int ret_alloc, ret_copy;
	unsigned __attribute__((unused)) com_id = 0;

	/* first make sure the destination has an allocated buffer */
	ret_alloc = allocate_per_node_buffer(state, dst_node);
	if (ret_alloc)
		goto nomem;

	/* if there is no need to actually read the data, 
	 * we do not perform any transfer */
	if (!donotread) {
		STARPU_ASSERT(state->ops);
		//STARPU_ASSERT(state->ops->copy_data_1_to_1);

#ifdef DATA_STATS
		size_t size = state->ops->get_size(state);
		update_comm_ammount(src_node, dst_node, size);
#endif
		
#ifdef USE_FXT
		com_id = STARPU_ATOMIC_ADD(&communication_cnt, 1);
#endif

		/* for now we set the size to 0 in the FxT trace XXX */
		TRACE_START_DRIVER_COPY(src_node, dst_node, 0, com_id);
		ret_copy = copy_data_1_to_1_generic(state, src_node, dst_node);
		TRACE_END_DRIVER_COPY(src_node, dst_node, 0, com_id);

		return ret_copy;
	}

	return 0;

nomem:
	return -ENOMEM;
}
