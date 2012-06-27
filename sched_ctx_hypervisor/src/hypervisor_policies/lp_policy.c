/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012  INRIA
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

#include "lp_tools.h"


static void lp_handle_poped_task(unsigned sched_ctx, int worker)
{
	if(_velocity_gap_btw_ctxs())
	{
		int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
		
		double nworkers[nsched_ctxs][2];

		int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{ 
			int total_nw[2];
			_get_total_nw(NULL, -1, 2, total_nw);
			double vmax = _lp_get_nworkers_per_ctx(nsched_ctxs, 2, nworkers, total_nw);
			if(vmax != 0.0)
			{
//				printf("********resize\n");
/* 			for( i = 0; i < nsched_ctxs; i++) */
/* 			{ */
/* 				printf("ctx %d/worker type %d: n = %lf \n", i, 0, res[i][0]); */
/* 				printf("ctx %d/worker type %d: n = %lf \n", i, 1, res[i][1]); */
/* 			} */
				int nworkers_rounded[nsched_ctxs][2];
				_lp_round_double_to_int(nsched_ctxs, 2, nworkers, nworkers_rounded);
  /*     		for( i = 0; i < nsched_ctxs; i++) */
/* 			{ */
/* 				printf("ctx %d/worker type %d: n = %d \n", i, 0, res_rounded[i][0]); */
/* 				printf("ctx %d/worker type %d: n = %d \n", i, 1, res_rounded[i][1]); */
/* 			} */
				
				_lp_redistribute_resources_in_ctxs(nsched_ctxs, 2, nworkers_rounded, nworkers);
			}
			pthread_mutex_unlock(&act_hypervisor_mutex);
		}
	}		
}
static void lp_size_ctxs(int *sched_ctxs, int ns, int *workers, int nworkers)
{	
	int nsched_ctxs = sched_ctxs == NULL ? sched_ctx_hypervisor_get_nsched_ctxs() : ns;
	double nworkers_per_type[nsched_ctxs][2];
	int total_nw[2];
	_get_total_nw(workers, nworkers, 2, total_nw);

	pthread_mutex_lock(&act_hypervisor_mutex);
	double vmax = _lp_get_nworkers_per_ctx(nsched_ctxs, 2, nworkers_per_type, total_nw);
	if(vmax != 0.0)
	{
		int i;
		printf("********size\n");
/* 		for( i = 0; i < nsched_ctxs; i++) */
/* 		{ */
/* 			printf("ctx %d/worker type %d: n = %lf \n", i, 0, res[i][0]); */
/* 			printf("ctx %d/worker type %d: n = %lf \n", i, 1, res[i][1]); */
/* 		} */
		int nworkers_per_type_rounded[nsched_ctxs][2];
		_lp_round_double_to_int(nsched_ctxs, 2, nworkers_per_type, nworkers_per_type_rounded);
/*       		for( i = 0; i < nsched_ctxs; i++) */
/* 		{ */
/* 			printf("ctx %d/worker type %d: n = %d \n", i, 0, res_rounded[i][0]); */
/* 			printf("ctx %d/worker type %d: n = %d \n", i, 1, res_rounded[i][1]); */
/* 		} */
		
		_lp_distribute_resources_in_ctxs(sched_ctxs, nsched_ctxs, 2, nworkers_per_type_rounded, nworkers_per_type, workers, nworkers);
	}
	pthread_mutex_unlock(&act_hypervisor_mutex);
}

#ifdef HAVE_GLPK_H
struct hypervisor_policy lp_policy = {
	.size_ctxs = lp_size_ctxs,
	.handle_poped_task = lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.custom = 0,
	.name = "lp"
};
	
#endif /* HAVE_GLPK_H */

