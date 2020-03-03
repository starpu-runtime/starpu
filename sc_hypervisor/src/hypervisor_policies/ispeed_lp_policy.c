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

#include <starpu_config.h>
#include "sc_hypervisor_lp.h"
#include "sc_hypervisor_policy.h"
#include <math.h>
#include <sys/time.h>

struct ispeed_lp_data
{
	double **speed;
	double *flops;
	double **flops_on_w;
	int *workers;
};

#ifdef STARPU_HAVE_GLPK_H
static double _compute_workers_distrib(int ns, int nw, double final_w_in_s[ns][nw],
					unsigned is_integer, double tmax, void *specific_data)
{
	struct ispeed_lp_data *sd = (struct ispeed_lp_data *)specific_data;

	double **speed = sd->speed;
	double *flops = sd->flops;
	
	double **final_flops_on_w = sd->flops_on_w;
	
	return sc_hypervisor_lp_simulate_distrib_flops_on_sample(ns, nw, final_w_in_s, is_integer, tmax, speed, flops, final_flops_on_w);
}

static unsigned _compute_flops_distribution_over_ctxs(int ns, int nw, double w_in_s[ns][nw], double **flops_on_w, unsigned *sched_ctxs, int *workers)
{
	double *flops = (double*)malloc(ns*sizeof(double));
	double **speed = (double **)malloc(ns*sizeof(double*));
	int i;
	for(i = 0; i < ns; i++)
		speed[i] = (double*)malloc(nw*sizeof(double));
	
	int w,s;

	struct sc_hypervisor_wrapper* sc_w = NULL;
	for(s = 0; s < ns; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);
		for(w = 0; w < nw; w++)
		{
			w_in_s[s][w] = 0.0;
			int worker = workers == NULL ? w : workers[w];

			speed[s][w] = sc_hypervisor_get_speed_per_worker(sc_w, worker);
			if(speed[s][w] == -1.0)
			{
				enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
				speed[s][w] = sc_hypervisor_get_speed(sc_w, arch);
				if(arch == STARPU_CUDA_WORKER)
				{
					unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, sc_w->sched_ctx);
					if(!worker_in_ctx)
					{
						double transfer_speed = starpu_transfer_bandwidth(STARPU_MAIN_RAM, starpu_worker_get_memory_node(worker)) / 1000;
						speed[s][w] = (speed[s][w] * transfer_speed) / (speed[s][w] + transfer_speed);
					}
				}

			}
			
//			printf("v[w%d][s%d] = %lf\n",w, s, speed[s][w]);
		}
		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs[s]);
		flops[s] = config->ispeed_ctx_sample/1000000000; /* in gflops */
	}
	
	/* take the exec time of the slowest ctx 
	   as starting point and then try to minimize it
	   as increasing it a little for the faster ctxs */
	double tmax = sc_hypervisor_get_slowest_ctx_exec_time();
 	double smallest_tmax = sc_hypervisor_get_fastest_ctx_exec_time(); //tmax - 0.5*tmax; 
//	printf("tmax %lf smallest %lf\n", tmax, smallest_tmax);
	double tmin = 0.0;

        struct ispeed_lp_data specific_data;
        specific_data.speed = speed;
        specific_data.flops = flops;
        specific_data.flops_on_w = flops_on_w;
        specific_data.workers = workers;

        unsigned found_sol = sc_hypervisor_lp_execute_dichotomy(ns, nw, w_in_s, 1, (void*)&specific_data, 
								tmin, tmax, smallest_tmax, _compute_workers_distrib);

	for(i = 0; i < ns; i++)
		free(speed[i]);
	free(speed);
	
	return found_sol;
}

static void _try_resizing(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
        int ns = sched_ctxs == NULL ? sc_hypervisor_get_nsched_ctxs() : nsched_ctxs;
	int nw = nworkers == -1 ? (int)starpu_worker_get_count() : nworkers; /* Number of different workers */
        unsigned *curr_sched_ctxs = sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : sched_ctxs;

        struct types_of_workers *tw = sc_hypervisor_get_types_of_workers(workers, nw);
        int ntypes_of_workers = tw->nw;


	double w_in_s[ns][nw];

	double **flops_on_w = (double**)malloc(ns*sizeof(double*));
	int i;
	for(i = 0; i < ns; i++)
		flops_on_w[i] = (double*)malloc(nw*sizeof(double));

	struct timeval start_time;
	struct timeval end_time;
	gettimeofday(&start_time, NULL);
	unsigned found_sol = _compute_flops_distribution_over_ctxs(ns, nw,  w_in_s, flops_on_w, curr_sched_ctxs, workers);
	gettimeofday(&end_time, NULL);
	
	long diff_s = end_time.tv_sec  - start_time.tv_sec;
	long diff_us = end_time.tv_usec  - start_time.tv_usec;
	
	__attribute__((unused))	float timing = (float)(diff_s*1000000 + diff_us)/1000.0;

	/* if we did find at least one solution redistribute the resources */
	if(found_sol)
	{
		int w, s;
		double nworkers_per_ctx[ns][ntypes_of_workers];
		int nworkers_per_ctx_rounded[ns][ntypes_of_workers];
		for(s = 0; s < ns; s++)
		{
			for(w = 0; w < ntypes_of_workers; w++)
			{
				nworkers_per_ctx[s][w] = 0.0;
				nworkers_per_ctx_rounded[s][w] = 0;
			}
		}
		
		for(s = 0; s < ns; s++)
		{
			for(w = 0; w < nw; w++)
			{
				enum starpu_worker_archtype arch = starpu_worker_get_type(w);
		
				int idx = sc_hypervisor_get_index_for_arch(arch, tw);
				nworkers_per_ctx[s][idx] += w_in_s[s][w];
				if(arch == STARPU_CUDA_WORKER)
				{
					if(w_in_s[s][w] >= 0.3)
						nworkers_per_ctx_rounded[s][idx]++;
				}
				else
				{
					if(w_in_s[s][w] > 0.5)
						nworkers_per_ctx_rounded[s][idx]++;
				}
			}
		}
/* 				for(s = 0; s < ns; s++) */
/* 					printf("%d: cpus = %lf gpus = %lf cpus_round = %d gpus_round = %d\n", s, nworkers[s][1], nworkers[s][0], */
/* 					       nworkers_rounded[s][1], nworkers_rounded[s][0]); */
		
		sc_hypervisor_lp_redistribute_resources_in_ctxs(ns, ntypes_of_workers, nworkers_per_ctx_rounded, nworkers_per_ctx, curr_sched_ctxs, tw);
	}
	free(tw);
	for(i = 0; i < ns; i++)
		free(flops_on_w[i]);
	free(flops_on_w);
}

static void ispeed_lp_handle_poped_task(__attribute__((unused))unsigned sched_ctx, __attribute__((unused))int worker, 
					__attribute__((unused))struct starpu_task *task, __attribute__((unused))uint32_t footprint)
{
        int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
        if(ret != EBUSY)
        {
                unsigned criteria = sc_hypervisor_get_resize_criteria();
                if(criteria != SC_NOTHING && criteria == SC_SPEED)
                {
                        if(sc_hypervisor_check_speed_gap_btw_ctxs(NULL, -1, NULL, -1))
                        {
                                _try_resizing(NULL, -1, NULL, -1);
                        }
                }
                STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
        }
}

static void ispeed_lp_handle_idle_cycle(unsigned sched_ctx, int worker)
{
        int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
        if(ret != EBUSY)
        {
                unsigned criteria = sc_hypervisor_get_resize_criteria();
                if(criteria != SC_NOTHING && criteria == SC_IDLE)
                {

			if(sc_hypervisor_check_idle(sched_ctx, worker))
                        {
                                _try_resizing(NULL, -1, NULL, -1);
                        }
                }
                STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
        }
}

static void ispeed_lp_resize_ctxs(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		_try_resizing(sched_ctxs, nsched_ctxs, workers, nworkers);
		STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
	}
}

static void ispeed_lp_end_ctx(__attribute__((unused))unsigned sched_ctx)
{
/* 	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx); */
/* 	int worker; */
/* 	for(worker = 0; worker < 12; worker++) */
/* 		printf("%d/%d: speed %lf\n", worker, sched_ctx, sc_w->ref_speed[worker]); */

	return;
}

struct sc_hypervisor_policy ispeed_lp_policy = {
	.size_ctxs = NULL,
	.resize_ctxs = ispeed_lp_resize_ctxs,
	.handle_poped_task = ispeed_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = ispeed_lp_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = ispeed_lp_end_ctx,
	.init_worker = NULL,
	.custom = 0,
	.name = "ispeed_lp"
};

#endif /* STARPU_HAVE_GLPK_H */
