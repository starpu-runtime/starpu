/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <datawizard/datawizard.h>
#include <datawizard/memalloc.h>
#include <datawizard/memory_nodes.h>
#include <core/workers.h>
#include <core/progress_hook.h>
#include <core/topology.h>
#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

static int ____starpu_datawizard_progress(unsigned memory_node, unsigned peer_start, unsigned peer_end, enum  _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned push_requests)
{
	int ret = 0;
	unsigned peer_node;

	/* in case some other driver requested data */
	for (peer_node = peer_start; peer_node < peer_end; peer_node++)
	{
		if (_starpu_handle_pending_node_data_requests(memory_node, peer_node, inout))
			ret = 1;
	}

	starpu_memchunk_tidy(memory_node);

	if (ret || push_requests)
	{
		/* Some transfers have finished, or the driver requests to really push more */
		unsigned pushed;
		unsigned ok = 1;

		for (peer_node = peer_start; ok && peer_node < peer_end; peer_node++)
		{
			if (_starpu_handle_node_data_requests(memory_node, peer_node, inout, may_alloc, &pushed) == -ENOMEM)
				ok = 0;
			if (pushed)
				ret = 1;
		}

		if (ok)
		{
			unsigned doidle = 1;

			/* We pushed all pending requests, we can afford pushing
			 * prefetch requests */
			for (peer_node = peer_start; ok && peer_node < peer_end; peer_node++)
			{
				if (_starpu_handle_node_prefetch_requests(memory_node, peer_node, inout, may_alloc, &pushed) == -ENOMEM)
					ok = 0;
				if (pushed)
					ret = 1;
				if (!_starpu_check_that_no_data_request_is_pending(memory_node, peer_node, inout))
					doidle = 0;
			}

			if (doidle)
				/* No pending transfer, push some idle transfer */
				for (peer_node = peer_start; ok && peer_node < peer_end; peer_node++)
				{
					if (_starpu_handle_node_idle_requests(memory_node, peer_node, inout, may_alloc, &pushed) == -ENOMEM)
						ok = 0;
					if (pushed)
						ret = 1;
				}
		}

	}

	return ret;
}

static int ___starpu_datawizard_progress(unsigned memory_node, unsigned nnodes, enum _starpu_may_alloc may_alloc, unsigned push_requests)
{
	int ret = 0;
	unsigned peer_node;

#ifdef STARPU_SIMGRID
	/* XXX */
	starpu_sleep(0.000001);
#endif
	STARPU_UYIELD();

	/* First handle all incoming transfers */
	ret |= ____starpu_datawizard_progress(memory_node, 0, nnodes, _STARPU_DATA_REQUEST_IN, may_alloc, push_requests);

	/* Then handle outgoing transfers */
	for (peer_node = 0; peer_node < nnodes; peer_node++)
		ret |= ____starpu_datawizard_progress(memory_node, peer_node, peer_node+1, _STARPU_DATA_REQUEST_OUT, may_alloc, push_requests);

	return ret;
}

int __starpu_datawizard_progress(enum _starpu_may_alloc may_alloc, unsigned push_requests)
{
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	unsigned memnode;

	if (!worker)
	{
		/* Call from main application, only make RAM requests progress */
		int ret = 0;
		int nnumas = starpu_memory_nodes_get_numa_count();
		int numa;
		for (numa = 0; numa < nnumas; numa++)
			ret |=	___starpu_datawizard_progress(numa, nnumas, may_alloc, push_requests);
		_starpu_execute_registered_progression_hooks();

		return ret;
	}

	/* processing requests may release some tasks, we cannot be already
	 * scheduling a task. */
	STARPU_ASSERT(!worker->state_sched_op_pending);

	if (worker->set)
		/* Runing one of the workers of a worker set. The reference for
		 * driving memory is its worker 0 (see registrations in topology.c) */
		worker = &worker->set->workers[0];

	unsigned current_worker_id = worker->workerid;
	int ret = 0;
	unsigned nnodes = starpu_memory_nodes_get_count();

	for (memnode = 0; memnode < nnodes; memnode++)
	{
		if (_starpu_worker_drives_memory[current_worker_id][memnode] == 1)
		{
			if(_starpu_config.conf.cuda_only_fast_alloc_other_memnodes && worker->arch == STARPU_CUDA_WORKER && worker->memory_node != memnode)
				ret |=  ___starpu_datawizard_progress(memnode, nnodes, _STARPU_DATAWIZARD_ONLY_FAST_ALLOC, push_requests);
			else
				ret |=  ___starpu_datawizard_progress(memnode, nnodes, may_alloc, push_requests);
			}
	}

	_starpu_execute_registered_progression_hooks();

	return ret;
}

void _starpu_datawizard_progress(enum _starpu_may_alloc may_alloc)
{
	__starpu_datawizard_progress(may_alloc, 1);
}

void _starpu_datawizard_handle_all_pending_node_data_requests(unsigned memnode)
{
	unsigned nnodes = starpu_memory_nodes_get_count();
	unsigned memnode2;

	for (memnode2 = 0; memnode2 < nnodes; memnode2++)
	{
		_starpu_handle_all_pending_node_data_requests(memnode, memnode2, _STARPU_DATA_REQUEST_IN);
		_starpu_handle_all_pending_node_data_requests(memnode, memnode2, _STARPU_DATA_REQUEST_OUT);
	}
}
