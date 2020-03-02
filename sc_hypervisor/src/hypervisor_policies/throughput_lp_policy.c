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

static double _glp_resolve(int ns, int nw, double speed[ns][nw], double w_in_s[ns][nw], unsigned integer);


static unsigned _compute_max_speed(int ns, int nw, double w_in_s[ns][nw], unsigned *in_sched_ctxs, int *workers)
{
	double speed[ns][nw];

	unsigned *sched_ctxs = in_sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : in_sched_ctxs;

	int w,s;

	struct sc_hypervisor_wrapper* sc_w = NULL;
	for(s = 0; s < ns; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);
		for(w = 0; w < nw; w++)
		{
			w_in_s[s][w] = 0.0;
			int worker = workers == NULL ? w : workers[w];

			enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
			speed[s][w] = sc_hypervisor_get_speed(sc_w, arch);
		}
	}


	struct timeval start_time;
	struct timeval end_time;
	gettimeofday(&start_time, NULL);

	double res = _glp_resolve(ns, nw, speed, w_in_s, 1);
	gettimeofday(&end_time, NULL);

	long diff_s = end_time.tv_sec  - start_time.tv_sec;
	long diff_us = end_time.tv_usec  - start_time.tv_usec;

	__attribute__((unused)) float timing = (float)(diff_s*1000000 + diff_us)/1000;

	if(res > 0.0)
		return 1;
	return 0;
}

/*
 * GNU Linear Programming Kit backend
 */
#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
static double _glp_resolve(int ns, int nw, double speed[ns][nw], double w_in_s[ns][nw], unsigned integer)
{
	int w = 0, s = 0;
	glp_prob *lp;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total speed");

	{
		int ne = 2 * ns * nw /* worker execution time */
			+ 1
			+ 1 ; /* glp dumbness */
		int n = 1;
		int ia[ne], ja[ne];
		double ar[ne];


		/* Variables:  x[s][w]
		 the acknwoledgment that the worker w belongs to the context s */
		glp_add_cols(lp, nw*ns + 1);

		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%ds%dn", w, s);
				glp_set_col_name(lp, s*nw+w+1, name);
				if (integer)
				{
                                        glp_set_col_kind(lp, s*nw+w+1, GLP_IV);
					glp_set_col_bnds(lp, s*nw+w+1, GLP_DB, 0, 1);
				}
				else
					glp_set_col_bnds(lp, s*nw+w+1, GLP_DB, 0.0, 1.0);

			}

		/* vmax should be positif */
		/* Z = vmax structural variable, x[s][w] are auxiliar variables */
		glp_set_col_name(lp, nw*ns+1, "vmax");
		glp_set_col_bnds(lp, nw*ns+1, GLP_LO, 0.0, 0.0);
		glp_set_obj_coef(lp, nw*ns+1, 1.);


		int curr_row_idx = 0;
		/* Total worker speed */
		glp_add_rows(lp, 1);

		/*sum(x[s][w]*speed[s][w]) >= vmax */
		char name[32], title[64];
		starpu_worker_get_name(w, name, sizeof(name));
		snprintf(title, sizeof(title), "worker %s", name);
		glp_set_row_name(lp, curr_row_idx + 1, title);

		for(s = 0; s < ns; s++)
		{
			for (w = 0; w < nw; w++)
			{
				/* x[s][w] */
				ia[n] = curr_row_idx + 1;
				ja[n] = s*nw+w+1;
				ar[n] = speed[s][w];
				n++;
			}
		}
		/* vmax */
		ia[n] = curr_row_idx + 1;
		ja[n] = nw*ns+1;
		ar[n] = (-1);
		n++;
		glp_set_row_bnds(lp, curr_row_idx + 1, GLP_LO, 0.0, 0.0);

		curr_row_idx += 1 ;

		/* sum(x[s][w]) = 1 */
		glp_add_rows(lp, nw);
		for (w = 0; w < nw; w++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "w%x", w);
			glp_set_row_name(lp, curr_row_idx+w+1, title);
			for(s = 0; s < ns; s++)
			{
				ia[n] = curr_row_idx+w+1;
				ja[n] = s*nw+w+1;
				ar[n] = 1;
				n++;
			}
			if(integer)
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1, 1);
			else
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1.0, 1.0);
		}

		if(n != ne)
			printf("ns= %d nw = %d n = %d ne = %d\n", ns, nw, n, ne);
		STARPU_ASSERT(n == ne);

		glp_load_matrix(lp, ne-1, ia, ja, ar);
	}

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	int ret = glp_simplex(lp, &parm);
	if (ret)
	{
		glp_delete_prob(lp);
		lp = NULL;
		return 0.0;
	}

        if (integer)
        {
                glp_iocp iocp;
                glp_init_iocp(&iocp);
                iocp.msg_lev = GLP_MSG_OFF;
                glp_intopt(lp, &iocp);
		int stat = glp_mip_status(lp);
		/* if we don't have a solution return */
		if(stat == GLP_NOFEAS)
		{
			glp_delete_prob(lp);
			lp = NULL;
			return 0.0;
		}
        }

	int stat = glp_get_prim_stat(lp);
	/* if we don't have a solution return */
	if(stat == GLP_NOFEAS)
	{
		glp_delete_prob(lp);
		lp = NULL;
		printf("No sol!!!\n");
		return 0.0;
	}

	double res = glp_get_obj_val(lp);

	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			if (integer)
				w_in_s[s][w] = (double)glp_mip_col_val(lp, s*nw+w+1);
			else
				w_in_s[s][w] = glp_get_col_prim(lp, s*nw+w+1);
		}

	glp_delete_prob(lp);
	return res;
}


static void _try_resizing(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	int ns = sched_ctxs == NULL ? sc_hypervisor_get_nsched_ctxs() : nsched_ctxs;
	int nw = workers == NULL ? (int)starpu_worker_get_count() : nworkers; /* Number of different workers */

	sched_ctxs = sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : sched_ctxs;

	double w_in_s[ns][nw];
	unsigned found_sol = _compute_max_speed(ns, nw,  w_in_s, sched_ctxs, workers);
	/* if we did find at least one solution redistribute the resources */
	if(found_sol)
	{
		struct types_of_workers *tw = sc_hypervisor_get_types_of_workers(workers, nw);
		int w, s;
		double nworkers_per_ctx[ns][tw->nw];
		int nworkers_per_ctx_rounded[ns][tw->nw];
		for(s = 0; s < ns; s++)
		{
			for(w = 0; w < nw; w++)
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
				int idx = sc_hypervisor_get_index_for_arch(STARPU_CUDA_WORKER, tw);
				nworkers_per_ctx[s][idx] += w_in_s[s][w];

				if(arch == STARPU_CUDA_WORKER)
				{
					if(w_in_s[s][w] >= 0.3)
						nworkers_per_ctx_rounded[s][idx]++;
				}
				else
				{
					int idx = sc_hypervisor_get_index_for_arch(STARPU_CPU_WORKER, tw);
					nworkers_per_ctx[s][idx] += w_in_s[s][w];
					if(w_in_s[s][w] > 0.5)
						nworkers_per_ctx_rounded[s][idx]++;
				}
			}
		}
/* 				for(s = 0; s < ns; s++) */
/* 					printf("%d: cpus = %lf gpus = %lf cpus_round = %d gpus_round = %d\n", s, nworkers[s][1], nworkers[s][0], */
/* 					       nworkers_rounded[s][1], nworkers_rounded[s][0]); */


		sc_hypervisor_lp_redistribute_resources_in_ctxs(ns, tw->nw, nworkers_per_ctx_rounded, nworkers_per_ctx, sched_ctxs, tw);
		free(tw);
	}
}

static void throughput_lp_handle_poped_task(__attribute__((unused))unsigned sched_ctx, __attribute__((unused))int worker,
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

static void throughput_lp_handle_idle_cycle(unsigned sched_ctx, int worker)
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
//                              sc_hypervisor_move_workers(sched_ctx, 3 - sched_ctx, &worker, 1, 1);
			}
                }
                STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
        }
}

static void throughput_lp_resize_ctxs(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		_try_resizing(sched_ctxs, nsched_ctxs, workers, nworkers);
		STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
	}
}

static void throughput_lp_end_ctx(__attribute__((unused))unsigned sched_ctx)
{
/* 	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx); */
/* 	int worker; */
/* 	for(worker = 0; worker < 12; worker++) */
/* 		printf("%d/%d: speed %lf\n", worker, sched_ctx, sc_w->ref_speed[worker]); */

	return;
}

struct sc_hypervisor_policy throughput_lp_policy = {
	.size_ctxs = NULL,
	.resize_ctxs = throughput_lp_resize_ctxs,
	.handle_poped_task = throughput_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = throughput_lp_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = throughput_lp_end_ctx,
	.init_worker = NULL,
	.custom = 0,
	.name = "throughput_lp"
};

#endif /* STARPU_HAVE_GLPK_H */
