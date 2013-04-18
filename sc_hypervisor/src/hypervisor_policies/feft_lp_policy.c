/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011 - 2013  INRIA
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

#include "sched_ctx_hypervisor_lp.h"
#include <starpu_config.h>
#include <sys/time.h>

#ifdef STARPU_HAVE_GLPK_H
static void feft_lp_handle_poped_task(unsigned sched_ctx, int worker, struct starpu_task *task, uint32_t footprint)
{
	if(_velocity_gap_btw_ctxs())
	{
		int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

		double nworkers[nsched_ctxs][2];

		int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{
			int nw = 1;
#ifdef STARPU_USE_CUDA
			int ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
			nw = ncuda != 0 ? 2 : 1;
#endif
			int total_nw[nw];
			_get_total_nw(NULL, -1, nw, total_nw);


			struct timeval start_time;
			struct timeval end_time;
			gettimeofday(&start_time, NULL);

			double vmax = _lp_get_nworkers_per_ctx(nsched_ctxs, nw, nworkers, total_nw);
			gettimeofday(&end_time, NULL);

			long diff_s = end_time.tv_sec  - start_time.tv_sec;
			long diff_us = end_time.tv_usec  - start_time.tv_usec;

			float timing = (float)(diff_s*1000000 + diff_us)/1000;

			if(vmax != 0.0)
			{
				int nworkers_rounded[nsched_ctxs][nw];
				_lp_round_double_to_int(nsched_ctxs, nw, nworkers, nworkers_rounded);
				_lp_redistribute_resources_in_ctxs(nsched_ctxs, nw, nworkers_rounded, nworkers);
			}
			starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
		}
	}
}
static void feft_lp_size_ctxs(int *sched_ctxs, int ns, int *workers, int nworkers)
{
	int nsched_ctxs = sched_ctxs == NULL ? sched_ctx_hypervisor_get_nsched_ctxs() : ns;
	int nw = 1;
#ifdef STARPU_USE_CUDA
	int ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
	nw = ncuda != 0 ? 2 : 1;
#endif
	double nworkers_per_type[nsched_ctxs][nw];
	int total_nw[nw];
	_get_total_nw(workers, nworkers, nw, total_nw);

	starpu_pthread_mutex_lock(&act_hypervisor_mutex);
	double vmax = _lp_get_nworkers_per_ctx(nsched_ctxs, nw, nworkers_per_type, total_nw);
	if(vmax != 0.0)
	{
// 		printf("********size\n");
/* 		int i; */
/* 		for( i = 0; i < nsched_ctxs; i++) */
/* 		{ */
/* 			printf("ctx %d/worker type %d: n = %lf \n", i, 0, nworkers_per_type[i][0]); */
/* #ifdef STARPU_USE_CUDA */
/* 			int ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER); */
/* 			if(ncuda != 0) */
/* 				printf("ctx %d/worker type %d: n = %lf \n", i, 1, nworkers_per_type[i][1]); */
/* #endif */
/* 		} */
		int nworkers_per_type_rounded[nsched_ctxs][nw];
		_lp_round_double_to_int(nsched_ctxs, nw, nworkers_per_type, nworkers_per_type_rounded);
/*       	for( i = 0; i < nsched_ctxs; i++) */
/* 		{ */
/* 			printf("ctx %d/worker type %d: n = %d \n", i, 0, nworkers_per_type_rounded[i][0]); */
/* #ifdef STARPU_USE_CUDA */
/* 			int ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER); */
/* 			if(ncuda != 0) */
/* 				printf("ctx %d/worker type %d: n = %d \n", i, 1, nworkers_per_type_rounded[i][1]); */
/* #endif */
/* 		} */
		int *current_sched_ctxs = sched_ctxs == NULL ? sched_ctx_hypervisor_get_sched_ctxs() : 
			sched_ctxs;

		unsigned has_workers = 0;
		int s;
		for(s = 0; s < ns; s++)
		{
			int nworkers_ctx = sched_ctx_hypervisor_get_nworkers_ctx(current_sched_ctxs[s], 
									     STARPU_ANY_WORKER);
			if(nworkers_ctx != 0)
			{
				has_workers = 1;
				break;
			}
		}
		if(has_workers)
			_lp_redistribute_resources_in_ctxs(nsched_ctxs, nw, nworkers_per_type_rounded, nworkers_per_type);
		else
			_lp_distribute_resources_in_ctxs(sched_ctxs, nsched_ctxs, nw, nworkers_per_type_rounded, nworkers_per_type, workers, nworkers);
	}
	starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
}

struct sched_ctx_hypervisor_policy feft_lp_policy = {
	.size_ctxs = feft_lp_size_ctxs,
	.handle_poped_task = feft_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.custom = 0,
	.name = "feft_lp"
};

#endif /* STARPU_HAVE_GLPK_H */
