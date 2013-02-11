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

#include <starpu_config.h>
#include "lp_tools.h"
#include <math.h>

static double _glp_resolve(int ns, int nw, double velocity[ns][nw], double flops[ns], double tmax, double flops_on_w[ns][nw], double w_in_s[ns][nw], int *workers);
static double _find_tmax(double t1, double t2);


static unsigned _compute_flops_distribution_over_ctxs(int ns, int nw, double w_in_s[ns][nw], double flops_on_w[ns][nw], int *in_sched_ctxs, int *workers)
{
	double draft_w_in_s[ns][nw];
	double draft_flops_on_w[ns][nw];
	double flops[ns];
	double velocity[ns][nw];

	int *sched_ctxs = in_sched_ctxs == NULL ? sched_ctx_hypervisor_get_sched_ctxs() : in_sched_ctxs;
	
	int w,s;

	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			w_in_s[s][w] = 0.0;
			draft_w_in_s[s][w] = 0.0;
			flops_on_w[s][w] = 0.0;
			draft_flops_on_w[s][w] = 0.0;
			int worker = workers == NULL ? w : workers[w];

			velocity[s][w] = _get_velocity_per_worker(sched_ctx_hypervisor_get_wrapper(sched_ctxs[s]), worker);
			if(velocity[s][w] == -1.0)
			{
				enum starpu_archtype arch = starpu_worker_get_type(worker);
				velocity[s][w] = _get_velocity_per_worker_type(sched_ctx_hypervisor_get_wrapper(sched_ctxs[s]), arch);
				if(velocity[s][w] == -1.0)
					velocity[s][w] = arch == STARPU_CPU_WORKER ? 5.0 : 50.0;
			}
			
//			printf("v[w%d][s%d] = %lf\n",w, s, velocity[s][w]);
		}
		struct sched_ctx_hypervisor_policy_config *config = sched_ctx_hypervisor_get_config(sched_ctxs[s]);
		flops[s] = config->ispeed_ctx_sample/1000000000; /* in gflops */
	}


	/* take the exec time of the slowest ctx 
	   as starting point and then try to minimize it
	   as increasing it a little for the faster ctxs */
	double tmax = _get_slowest_ctx_exec_time();
	double smallest_tmax = tmax - 0.5*tmax;

	double res = 1.0;
	unsigned has_sol = 0;
	double tmin = 0.0;
	double old_tmax = 0.0;
	unsigned found_sol = 0;

	struct timeval start_time;
	struct timeval end_time;
	int nd = 0;
	gettimeofday(&start_time, NULL);

	/* we fix tmax and we do not treat it as an unknown
	   we just vary by dichotomy its values*/
	while(tmax > 1.0)
	{
		/* find solution and save the values in draft tables
		   only if there is a solution for the system we save them
		   in the proper table */
		res = _glp_resolve(ns, nw, velocity, flops, tmax, draft_flops_on_w, draft_w_in_s, workers);
		if(res != 0.0)
		{
			for(s = 0; s < ns; s++)
				for(w = 0; w < nw; w++)
				{
					w_in_s[s][w] = draft_w_in_s[s][w];
					flops_on_w[s][w] = draft_flops_on_w[s][w];
				}
			has_sol = 1;
			found_sol = 1;
		}
		else
			has_sol = 0;

		/* if we have a solution with this tmax try a smaller value
		   bigger than the old min */
		if(has_sol)
		{
			if(old_tmax != 0.0 && (old_tmax - tmax) < 0.5)
				break;
			old_tmax = tmax;
		}
		else /*else try a bigger one but smaller than the old tmax */
		{
			tmin = tmax;
			if(old_tmax != 0.0)
				tmax = old_tmax;
		}
		if(tmin == tmax) break;
		tmax = _find_tmax(tmin, tmax);

		if(tmax < smallest_tmax)
		{
			tmax = old_tmax;
			tmin = smallest_tmax;
			tmax = _find_tmax(tmin, tmax);
		}
		nd++;
	}
	gettimeofday(&end_time, NULL);

	long diff_s = end_time.tv_sec  - start_time.tv_sec;
	long diff_us = end_time.tv_usec  - start_time.tv_usec;

	float timing = (float)(diff_s*1000000 + diff_us)/1000;

//        fprintf(stdout, "nd = %d total time: %f ms \n", nd, timing);

	return found_sol;
}

/*
 * GNU Linear Programming Kit backend
 */
#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
static double _glp_resolve(int ns, int nw, double velocity[ns][nw], double flops[ns], double tmax, double flops_on_w[ns][nw], double w_in_s[ns][nw], int *workers)
{
	int w, s;
	glp_prob *lp;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total execution time");

	{
		int ne = 4 * ns * nw /* worker execution time */
			+ 1; /* glp dumbness */
		int n = 1;
		int ia[ne], ja[ne];
		double ar[ne];


		/* Variables: number of flops assigned to worker w in context s, and 
		 the acknwoledgment that the worker w belongs to the context s */
		glp_add_cols(lp, 2*nw*ns);
#define colnum(w, s) ((s)*nw+(w)+1)
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
				glp_set_obj_coef(lp, nw*ns+colnum(w,s), 1.);
		
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
			{
				char name[32];
				snprintf(name, sizeof(name), "flopsw%ds%dn", w, s);
				glp_set_col_name(lp, colnum(w,s), name);
				glp_set_col_bnds(lp, colnum(w,s), GLP_LO, 0., 0.);

				snprintf(name, sizeof(name), "w%ds%dn", w, s);
				glp_set_col_name(lp, nw*ns+colnum(w,s), name);
				glp_set_col_bnds(lp, nw*ns+colnum(w,s), GLP_DB, 0.0, 1.0);

			}


		int curr_row_idx = 0;
		/* Total worker execution time */
		glp_add_rows(lp, nw*ns);

		/*nflops[s][w]/v[s][w] < x[s][w]*tmax */
		for(s = 0; s < ns; s++)
		{
			for (w = 0; w < nw; w++)
			{
				char name[32], title[64];
				starpu_worker_get_name(w, name, sizeof(name));
				snprintf(title, sizeof(title), "worker %s", name);
				glp_set_row_name(lp, curr_row_idx+s*nw+w+1, title);

				/* nflosp[s][w] */
				ia[n] = curr_row_idx+s*nw+w+1;
				ja[n] = colnum(w, s);
				ar[n] = 1 / velocity[s][w];

				n++;
				
				/* x[s][w] = 1 | 0 */
				ia[n] = curr_row_idx+s*nw+w+1;
				ja[n] = nw*ns+colnum(w,s);
				ar[n] = (-1) * tmax;
				n++;
				glp_set_row_bnds(lp, curr_row_idx+s*nw+w+1, GLP_UP, 0.0, 0.0);
			}
		}

		curr_row_idx += nw*ns;

		/* sum(flops[s][w]) = flops[s] */
		glp_add_rows(lp, ns);
		for (s = 0; s < ns; s++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "flops %lf ctx%d", flops[s], s);
			glp_set_row_name(lp, curr_row_idx+s+1, title);
			for (w = 0; w < nw; w++)
			{
				ia[n] = curr_row_idx+s+1;
				ja[n] = colnum(w, s);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, curr_row_idx+s+1, GLP_FX, flops[s], flops[s]);
		}

		curr_row_idx += ns;

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
				ja[n] = nw*ns+colnum(w,s);
				ar[n] = 1;
				n++;
			}

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

	int stat = glp_get_prim_stat(lp);
	/* if we don't have a solution return */
	if(stat == GLP_NOFEAS)
	{
		glp_delete_prob(lp);
		lp = NULL;
		return 0.0;
	}

	double res = glp_get_obj_val(lp);

	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			flops_on_w[s][w] = glp_get_col_prim(lp, colnum(w, s));
			w_in_s[s][w] = glp_get_col_prim(lp, nw*ns+colnum(w,s));
//			printf("w_in_s[s%d][w%d] = %lf flops[s%d][w%d] = %lf \n", s, w, w_in_s[s][w], s, w, flops_on_w[s][w]);
		}

	glp_delete_prob(lp);
	return res;
}


static double _find_tmax(double t1, double t2)
{
	return t1 + ((t2 - t1)/2);
}


static void ispeed_lp_handle_poped_task(unsigned sched_ctx, int worker)
{

	int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(_velocity_gap_btw_ctxs())
		{
			int ns = sched_ctx_hypervisor_get_nsched_ctxs();
			int nw = starpu_worker_get_count(); /* Number of different workers */

			double w_in_s[ns][nw];
			double flops_on_w[ns][nw];

			unsigned found_sol = _compute_flops_distribution_over_ctxs(ns, nw,  w_in_s, flops_on_w, NULL, NULL);
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
							if(w_in_s[s][w] > 0.3)
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


struct sched_ctx_hypervisor_policy ispeed_lp_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = ispeed_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.custom = 0,
	.name = "ispeed_lp"
};

#endif /* STARPU_HAVE_GLPK_H */
