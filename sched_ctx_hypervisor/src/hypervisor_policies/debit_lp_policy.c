/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2013  INRIA
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
#include "lp_tools.h"
#include <math.h>

static double _glp_resolve(int ns, int nw, double velocity[ns][nw], double w_in_s[ns][nw], int *workers, unsigned integer);


static unsigned _compute_max_velocity(int ns, int nw, double w_in_s[ns][nw], int *in_sched_ctxs, int *workers)
{
	double velocity[ns][nw];

	int *sched_ctxs = in_sched_ctxs == NULL ? sched_ctx_hypervisor_get_sched_ctxs() : in_sched_ctxs;
	
	int w,s;

	struct sched_ctx_hypervisor_wrapper* sc_w = NULL;
	for(s = 0; s < ns; s++)
	{
		sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[s]);
		for(w = 0; w < nw; w++)
		{
			w_in_s[s][w] = 0.0;
			int worker = workers == NULL ? w : workers[w];

			enum starpu_archtype arch = starpu_worker_get_type(worker);
			velocity[s][w] = sched_ctx_hypervisor_get_velocity(sc_w, arch);
		}
	}
	

	struct timeval start_time;
	struct timeval end_time;
	gettimeofday(&start_time, NULL);

	double res = _glp_resolve(ns, nw, velocity, w_in_s, workers, 1);
	gettimeofday(&end_time, NULL);

	long diff_s = end_time.tv_sec  - start_time.tv_sec;
	long diff_us = end_time.tv_usec  - start_time.tv_usec;

	float timing = (float)(diff_s*1000000 + diff_us)/1000;

	if(res > 0.0)
		return 1;
	return 0;
}

/*
 * GNU Linear Programming Kit backend
 */
#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
static double _glp_resolve(int ns, int nw, double velocity[ns][nw], double w_in_s[ns][nw], int *workers, unsigned integer)
{
	int w, s;
	glp_prob *lp;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total velocity");

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
		/* Total worker velocity */
		glp_add_rows(lp, 1);

		/*sum(x[s][w]*velocity[s][w]) >= vmax */
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
				ar[n] = velocity[s][w];
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


static void debit_lp_handle_poped_task(unsigned sched_ctx, int worker, struct starpu_task *task, uint32_t footprint)
{
	struct sched_ctx_hypervisor_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	_get_velocity_per_worker(sc_w, worker);
	int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(_velocity_gap_btw_ctxs())
		{
			int ns = sched_ctx_hypervisor_get_nsched_ctxs();
			int nw = starpu_worker_get_count(); /* Number of different workers */

			double w_in_s[ns][nw];
			unsigned found_sol = _compute_max_velocity(ns, nw,  w_in_s, NULL, NULL);
			/* if we did find at least one solution redistribute the resources */
			if(found_sol)
			{
				int w, s;
				double nworkers[ns][2];
				int nworkers_rounded[ns][2];
				for(s = 0; s < ns; s++)
				{
					nworkers[s][0] = 0.0;
					nworkers[s][1] = 0.0;
					nworkers_rounded[s][0] = 0;
					nworkers_rounded[s][1] = 0;

				}

				for(s = 0; s < ns; s++)
				{
					for(w = 0; w < nw; w++)
					{
						enum starpu_archtype arch = starpu_worker_get_type(w);

						if(arch == STARPU_CUDA_WORKER)
						{
							nworkers[s][0] += w_in_s[s][w];
							if(w_in_s[s][w] >= 0.3)
								nworkers_rounded[s][0]++;
						}
						else
						{
							nworkers[s][1] += w_in_s[s][w];
							if(w_in_s[s][w] > 0.5)
								nworkers_rounded[s][1]++;
						}
					}
				}
/* 				for(s = 0; s < ns; s++) */
/* 					printf("%d: cpus = %lf gpus = %lf cpus_round = %d gpus_round = %d\n", s, nworkers[s][1], nworkers[s][0], */
/* 					       nworkers_rounded[s][1], nworkers_rounded[s][0]); */

				_lp_redistribute_resources_in_ctxs(ns, 2, nworkers_rounded, nworkers);

			}
		}
		pthread_mutex_unlock(&act_hypervisor_mutex);
	}
}

static void debit_lp_end_ctx(unsigned sched_ctx)
{
	struct sched_ctx_hypervisor_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	int worker;
/* 	for(worker = 0; worker < 12; worker++) */
/* 		printf("%d/%d: speed %lf\n", worker, sched_ctx, sc_w->ref_velocity[worker]); */

	return;
}

struct sched_ctx_hypervisor_policy debit_lp_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = debit_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = debit_lp_end_ctx,
	.custom = 0,
	.name = "debit_lp"
};

#endif /* STARPU_HAVE_GLPK_H */
