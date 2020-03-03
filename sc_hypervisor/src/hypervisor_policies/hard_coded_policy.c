/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "sc_hypervisor_policy.h"
#include "sc_hypervisor_lp.h"
#include "sc_hypervisor_policy.h"

unsigned hard_coded_worker_belong_to_other_sched_ctx(unsigned sched_ctx, int worker)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	int i;
	for(i = 0; i < nsched_ctxs; i++)
		if(sched_ctxs[i] != sched_ctx && starpu_sched_ctx_contains_worker(worker, sched_ctxs[i]))
			return 1;
	return 0;
}

void hard_coded_handle_idle_cycle(unsigned sched_ctx, int worker)
{
	unsigned criteria = sc_hypervisor_get_resize_criteria();
        if(criteria != SC_NOTHING)// && criteria == SC_SPEED)
        {

		int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{
//			if(sc_hypervisor_criteria_fulfilled(sched_ctx, worker))
//			if(sc_hypervisor_check_speed_gap_btw_ctxs(NULL, -1, NULL, -1))
			if(sc_hypervisor_check_idle(sched_ctx, worker))
			{
				if(hard_coded_worker_belong_to_other_sched_ctx(sched_ctx, worker))
					sc_hypervisor_remove_workers_from_sched_ctx(&worker, 1, sched_ctx, 1);
				else
				{
					//	sc_hypervisor_policy_resize_to_unknown_receiver(sched_ctx, 0);
					unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
					int ns = sc_hypervisor_get_nsched_ctxs();


					int nworkers = (int)starpu_worker_get_count();
					struct types_of_workers *tw = sc_hypervisor_get_types_of_workers(NULL, nworkers);
					int nw = tw->nw;
					double w_in_s[ns][nw];
					w_in_s[0][0] = 1;
					w_in_s[0][1] = 3;

					w_in_s[1][0] = 8;
					w_in_s[1][1] = 0;

//				sc_hypervisor_lp_place_resources_in_ctx(ns, nw, w_in_s, sched_ctxs, NULL, 1, tw);
					sc_hypervisor_lp_distribute_floating_no_resources_in_ctxs(sched_ctxs, ns, tw->nw, w_in_s, NULL, nworkers, tw);
					free(tw);
				}
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
		}
	}
}
static void hard_coded_handle_poped_task(unsigned sched_ctx, __attribute__((unused))int worker, struct starpu_task *task, uint32_t footprint)
{
	unsigned criteria = sc_hypervisor_get_resize_criteria();
        if(criteria != SC_NOTHING && criteria == SC_SPEED)
        {

		int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{

			if(sc_hypervisor_criteria_fulfilled(sched_ctx, worker))
			{
				//	sc_hypervisor_policy_resize_to_unknown_receiver(sched_ctx, 0);
				unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
				int ns = sc_hypervisor_get_nsched_ctxs();

				int nworkers = (int)starpu_worker_get_count();
				struct types_of_workers *tw = sc_hypervisor_get_types_of_workers(NULL, nworkers);
				int nw = tw->nw;
				double w_in_s[ns][nw];
				w_in_s[0][0] = 1;
				w_in_s[0][1] = 3;

				w_in_s[1][0] = 8;
				w_in_s[1][1] = 0;
//				sc_hypervisor_lp_place_resources_in_ctx(ns, nw, w_in_s, sched_ctxs, NULL, 1, tw);
				sc_hypervisor_lp_distribute_floating_no_resources_in_ctxs(sched_ctxs, ns, tw->nw, w_in_s, NULL, nworkers, tw);
				free(tw);
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
		}
	}
}
struct sc_hypervisor_policy hard_coded_policy =
{
	.size_ctxs = NULL,
	.handle_poped_task = hard_coded_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = hard_coded_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.init_worker = NULL,
	.custom = 0,
	.name = "hard_coded"
};
