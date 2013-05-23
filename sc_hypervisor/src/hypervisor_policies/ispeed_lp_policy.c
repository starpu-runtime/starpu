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
#include "sc_hypervisor_lp.h"
#include "sc_hypervisor_policy.h"
#include <math.h>
#include <sys/time.h>

struct ispeed_lp_data
{
	double **velocity;
	double *flops;
	double **flops_on_w;
	int *workers;
};

/*
 * GNU Linear Programming Kit backend
 */
#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
static double _glp_resolve (int ns, int nw, double final_w_in_s[ns][nw],
			    unsigned is_integer, double tmax, void *specific_data)
{
	struct ispeed_lp_data *sd = (struct ispeed_lp_data *)specific_data;

	double **velocity = sd->velocity;
	double *flops = sd->flops;
	
	double **final_flops_on_w = sd->flops_on_w;
        int *workers = sd->workers;
	
	double w_in_s[ns][nw];
	double flops_on_w[ns][nw];

	int w, s;
	glp_prob *lp;

//	printf("try with tmax %lf\n", tmax);
	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total execution time");

	{
		int ne = 5 * ns * nw /* worker execution time */
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
				if (is_integer)
				{
                                        glp_set_col_kind(lp, nw*ns+colnum(w, s), GLP_IV);
					glp_set_col_bnds(lp, nw*ns+colnum(w,s), GLP_DB, 0, 1);
				}
				else
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
			if(is_integer)				
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1, 1);
			else
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1.0, 1.0);
		}

		curr_row_idx += nw;

		/* sum(nflops[s][w]) > 0*/
		glp_add_rows(lp, nw);
		for (w = 0; w < nw; w++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "flopsw%x", w);
			glp_set_row_name(lp, curr_row_idx+w+1, title);
			for(s = 0; s < ns; s++)
			{
				ia[n] = curr_row_idx+w+1;
				ja[n] = colnum(w,s);
				ar[n] = 1;
				n++;
			}

			glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_LO, 0.1, 0.);
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

        if (is_integer)
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
		return 0.0;
	}

	double res = glp_get_obj_val(lp);

	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			flops_on_w[s][w] = glp_get_col_prim(lp, colnum(w, s));
			if (is_integer)
				w_in_s[s][w] = (double)glp_mip_col_val(lp, nw*ns+colnum(w, s));
			else
				w_in_s[s][w] = glp_get_col_prim(lp, nw*ns+colnum(w,s));
//			printf("w_in_s[s%d][w%d] = %lf flops[s%d][w%d] = %lf \n", s, w, w_in_s[s][w], s, w, flops_on_w[s][w]);
		}

	glp_delete_prob(lp);
	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			final_w_in_s[s][w] = w_in_s[s][w];
			final_flops_on_w[s][w] = flops_on_w[s][w];
		}

	return res;
}

static unsigned _compute_flops_distribution_over_ctxs(int ns, int nw, double w_in_s[ns][nw], double **flops_on_w, int *in_sched_ctxs, int *workers)
{
//	double flops[ns];
//	double velocity[ns][nw];
	double *flops = (double*)malloc(ns*sizeof(double));
	double **velocity = (double **)malloc(ns*sizeof(double*));
	int i;
	for(i = 0; i < ns; i++)
		velocity[i] = (double*)malloc(nw*sizeof(double));

	int *sched_ctxs = in_sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : in_sched_ctxs;
	
	int w,s;

	struct sc_hypervisor_wrapper* sc_w = NULL;
	double total_flops = 0.0;
	for(s = 0; s < ns; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);
		for(w = 0; w < nw; w++)
		{
			w_in_s[s][w] = 0.0;
			int worker = workers == NULL ? w : workers[w];

			velocity[s][w] = sc_hypervisor_get_velocity_per_worker(sc_w, worker);
			if(velocity[s][w] == -1.0)
			{
				enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
				velocity[s][w] = sc_hypervisor_get_velocity(sc_w, arch);
				if(arch == STARPU_CUDA_WORKER)
				{
					unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, sc_w->sched_ctx);
					if(!worker_in_ctx)
					{
						double transfer_velocity = starpu_get_bandwidth_RAM_CUDA(worker) / 1000;
						velocity[s][w] = (velocity[s][w] * transfer_velocity) / (velocity[s][w] + transfer_velocity);
					}
				}

			}
			
//			printf("v[w%d][s%d] = %lf\n",w, s, velocity[s][w]);
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
        specific_data.velocity = velocity;
        specific_data.flops = flops;
        specific_data.flops_on_w = flops_on_w;
        specific_data.workers = workers;

        unsigned found_sol = sc_hypervisor_lp_execute_dichotomy(ns, nw, w_in_s, 1, (void*)&specific_data, 
								tmin, tmax, smallest_tmax, _glp_resolve);

	for(i = 0; i < ns; i++)
		free(velocity[i]);
	free(velocity);
	
	return found_sol;
}



static void ispeed_lp_handle_poped_task(unsigned sched_ctx, int worker, struct starpu_task *task, uint32_t footprint)
{
	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	sc_hypervisor_get_velocity_per_worker(sc_w, worker);
	int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(sc_hypervisor_has_velocity_gap_btw_ctxs())
		{
			int ns = sc_hypervisor_get_nsched_ctxs();
			int nw = starpu_worker_get_count(); /* Number of different workers */

			double w_in_s[ns][nw];
//			double flops_on_w[ns][nw];
			double **flops_on_w = (double**)malloc(ns*sizeof(double*));
			int i;
			for(i = 0; i < ns; i++)
				flops_on_w[i] = (double*)malloc(nw*sizeof(double));

			printf("ns = %d nw = %d\n", ns, nw);
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
						enum starpu_worker_archtype arch = starpu_worker_get_type(w);

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

				sc_hypervisor_lp_redistribute_resources_in_ctxs(ns, 2, nworkers_rounded, nworkers);
			}
			for(i = 0; i < ns; i++)
				free(flops_on_w[i]);
			free(flops_on_w);
		}
		starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
	}
}

static void ispeed_lp_end_ctx(unsigned sched_ctx)
{
	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	int worker;
/* 	for(worker = 0; worker < 12; worker++) */
/* 		printf("%d/%d: speed %lf\n", worker, sched_ctx, sc_w->ref_velocity[worker]); */

	return;
}

struct sc_hypervisor_policy ispeed_lp_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = ispeed_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = ispeed_lp_end_ctx,
	.custom = 0,
	.name = "ispeed_lp"
};

#endif /* STARPU_HAVE_GLPK_H */
