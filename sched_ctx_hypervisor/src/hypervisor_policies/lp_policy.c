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

#include "policy_utils.h"


/*                                                                                                                                                                                                                  
 * GNU Linear Programming Kit backend                                                                                                                                                                               
 */
#ifdef HAVE_GLPK_H
#include <glpk.h>
static void _glp_resolve(int ns, int nw, double v[ns][nw], double flops[ns], double res[ns][nw])
{
	int s, w;
	glp_prob *lp;
	
	int ne =
		(ns*nw+1)*(ns+nw)
		+ 1; /* glp dumbness */
	int n = 1;
	int ia[ne], ja[ne];
	double ar[ne];

	lp = glp_create_prob();

	glp_set_prob_name(lp, "sample");
	glp_set_obj_dir(lp, GLP_MAX);
        glp_set_obj_name(lp, "max speed");

	/* we add nw*ns columns one for each type of worker in each context 
	   and another column corresponding to the 1/tmax bound (bc 1/tmax is a variable too)*/
	glp_add_cols(lp, nw*ns+1);

	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			char name[32];
			snprintf(name, sizeof(name), "worker%dctx%d", w, s);
			glp_set_col_name(lp, n, name);
			glp_set_col_bnds(lp, n, GLP_LO, 0.3, 0.0);
			n++;
		}
	}

	/*1/tmax should belong to the interval [0.0;1.0]*/
	glp_set_col_name(lp, n, "vmax");
	glp_set_col_bnds(lp, n, GLP_DB, 0.0, 1.0);
	/* Z = 1/tmax -> 1/tmax structural variable, nCPUs & nGPUs in ctx are auxiliar variables */
	glp_set_obj_coef(lp, n, 1.0);

	n = 1;
	/* one row corresponds to one ctx*/
	glp_add_rows(lp, ns);

	for(s = 0; s < ns; s++)
	{
		char name[32];
		snprintf(name, sizeof(name), "ctx%d", s);
		glp_set_row_name(lp, s+1, name);
		glp_set_row_bnds(lp, s+1, GLP_LO, 0., 0.);

		for(w = 0; w < nw; w++)
		{
			int s2;
			for(s2 = 0; s2 < ns; s2++)
			{
				if(s2 == s)
				{
					ia[n] = s+1;
					ja[n] = w + nw*s2 + 1;
					ar[n] = v[s][w];
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				else
				{
					ia[n] = s+1;
					ja[n] = w + nw*s2 + 1;
					ar[n] = 0.0;
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				n++;
			}
		}
		/* 1/tmax */
		ia[n] = s+1;
		ja[n] = ns*nw+1;
		ar[n] = (-1) * flops[s];
//		printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
		n++;
	}
	
	/*we add another linear constraint : sum(all cpus) = 9 and sum(all gpus) = 3 */
	glp_add_rows(lp, nw);

	for(w = 0; w < nw; w++)
	{
		char name[32];
		snprintf(name, sizeof(name), "w%d", w);
		glp_set_row_name(lp, ns+w+1, name);
		for(s = 0; s < ns; s++)
		{
			int w2;
			for(w2 = 0; w2 < nw; w2++)
			{
				if(w2 == w)
				{
					ia[n] = ns+w+1;
					ja[n] = w2+s*nw + 1;
					ar[n] = 1.0;
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				else
				{
					ia[n] = ns+w+1;
					ja[n] = w2+s*nw + 1;
					ar[n] = 0.0;
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				n++;
			}
		}
		/* 1/tmax */
		ia[n] = ns+w+1;
		ja[n] = ns*nw+1;
		ar[n] = 0.0;
//		printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
		n++;

		/*sum(all gpus) = 3*/
		if(w == 0)
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, 3., 3.);

		/*sum(all cpus) = 9*/
		if(w == 1) 
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, 9., 9.);
	}

	STARPU_ASSERT(n == ne);

	glp_load_matrix(lp, ne-1, ia, ja, ar);

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	glp_simplex(lp, &parm);
//	glp_simplex(lp, NULL);
	
	double vmax1 = glp_get_obj_val(lp);
	printf("vmax1 = %lf \n", vmax1);

	n = 1;
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			res[s][w] = glp_get_col_prim(lp, n);
			n++;
		}
	}

	glp_delete_prob(lp);
	return;
}

static void _round_double_to_int(int ns, int nw, double res[ns][nw], int res_rounded[ns][nw])
{
	int s, w;
	double left_res[nw];
	for(w = 0; w < nw; w++)
		left_res[nw] = 0.0;
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			int x = floor(res[s][w]);
			double x_double = (double)x;
			double diff = res[s][w] - x_double;
			
			if(diff != 0.0)
			{
				if(diff > 0.5)
				{
					if(left_res[w] != 0.0)
					{
						if((diff + left_res[w]) > 0.5)
						{
							res_rounded[s][w] = x + 1;
							left_res[w] = (-1.0) * (x_double + 1.0 - (res[s][w] + left_res[w]));
						}
						else
						{
							res_rounded[s][w] = x;
							left_res[w] = (-1.0) * (diff + left_res[w]);
						}
					}
					else
					{
						res_rounded[s][w] = x + 1;
						left_res[w] = (-1.0) * (x_double + 1.0 - res[s][w]);
					}

				}
				else
				{
					if((diff + left_res[w]) > 0.5)
					{
						res_rounded[s][w] = x + 1;
						left_res[w] = (-1.0) * (x_double + 1.0 - (res[s][w] + left_res[w]));
					}
					else
					{
						res_rounded[s][w] = x;
						left_res[w] = diff;
					}
				}
			}
		}
	}		
}

static void _redistribute_resources_in_ctxs(int ns, int nw, int res_rounded[ns][nw], double res[ns][nw])
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int s, s2, w;
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			enum starpu_archtype arch;
			if(w == 0) arch = STARPU_CUDA_WORKER;
			if(w == 1) arch = STARPU_CPU_WORKER;

			if(w == 1)
			{
				unsigned nworkers_ctx = get_nworkers_ctx(sched_ctxs[s], arch);
				if(nworkers_ctx > res_rounded[s][w])
				{
					int nworkers_to_move = nworkers_ctx - res_rounded[s][w];
					int receiving_s = -1;
					
					for(s2 = 0; s2 < ns; s2++)
					{
						if(sched_ctxs[s2] != sched_ctxs[s])
						{
							int nworkers_ctx2 = get_nworkers_ctx(sched_ctxs[s2], arch);
							if((res_rounded[s2][w] - nworkers_ctx2) >= nworkers_to_move)
							{
								receiving_s = sched_ctxs[s2];
								break;
							}
						}
					}
					if(receiving_s != -1)
					{
						int *workers_to_move = _get_first_workers(sched_ctxs[s], &nworkers_to_move, arch);
						sched_ctx_hypervisor_move_workers(sched_ctxs[s], receiving_s, workers_to_move, nworkers_to_move);
						struct policy_config *new_config = sched_ctx_hypervisor_get_config(receiving_s);
						int i;
						for(i = 0; i < nworkers_to_move; i++)
							new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;
						
						free(workers_to_move);
					}
				}
			}
			else
			{
				double nworkers_ctx = get_nworkers_ctx(sched_ctxs[s], arch) * 1.0;
				if(nworkers_ctx > res[s][w])
				{
					double nworkers_to_move = nworkers_ctx - res[s][w];
					int receiving_s = -1;
					
					for(s2 = 0; s2 < ns; s2++)
					{
						if(sched_ctxs[s2] != sched_ctxs[s])
						{
							double nworkers_ctx2 = get_nworkers_ctx(sched_ctxs[s2], arch) * 1.0;
							if((res[s2][w] - nworkers_ctx2) >= nworkers_to_move)
							{
								receiving_s = sched_ctxs[s2];
								break;
							}
						}
					}
					if(receiving_s != -1)
					{
						int x = floor(nworkers_to_move);
						double x_double = (double)x;
						double diff = nworkers_to_move - x_double;
						if(diff == 0)
						{
							int *workers_to_move = _get_first_workers(sched_ctxs[s], &x, arch);
							sched_ctx_hypervisor_move_workers(sched_ctxs[s], receiving_s, workers_to_move, x);
							struct policy_config *new_config = sched_ctx_hypervisor_get_config(receiving_s);
							int i;
							for(i = 0; i < x; i++)
								new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;
							
							free(workers_to_move);
						}
						else
						{
							x+=1;
							int *workers_to_move = _get_first_workers(sched_ctxs[s], &x, arch);
							sched_ctx_hypervisor_remove_workers_from_sched_ctx(workers_to_move, x-1, sched_ctxs[s]);
							if(diff > 0.3)
								sched_ctx_hypervisor_add_workers_to_sched_ctx(workers_to_move, x, receiving_s);
							else
								sched_ctx_hypervisor_add_workers_to_sched_ctx(workers_to_move, x-1, receiving_s);

							struct policy_config *new_config = sched_ctx_hypervisor_get_config(receiving_s);
							int i;
							for(i = 0; i < x-1; i++)
								new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;

							free(workers_to_move);
							

						}
					}
				}
			}

		}
	}
}

static void lp_handle_poped_task(unsigned sched_ctx, int worker)
{
	if(_velocity_gap_btw_ctxs())
	{
		int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
		int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
//		int nsched_ctxs = 3;
		
		double v[nsched_ctxs][2];
		double flops[nsched_ctxs];
		double res[nsched_ctxs][2];

		int i = 0;
		struct sched_ctx_wrapper* sc_w;
		for(i = 0; i < nsched_ctxs; i++)
		{
			sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[i]);
			v[i][0] = 200.0;//_get_velocity_per_worker_type(sc_w, STARPU_CUDA_WORKER);
			v[i][1] = 20.0;//_get_velocity_per_worker_type(sc_w, STARPU_CPU_WORKER);
			flops[i] = sc_w->remaining_flops/1000000000; //sc_w->total_flops/1000000000; /* in gflops*/
			printf("%d: flops %lf\n", sched_ctxs[i], flops[i]);
		}
                
		int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{
			_glp_resolve(nsched_ctxs, 2, v, flops, res);
			for( i = 0; i < nsched_ctxs; i++)
			{
				printf("ctx %d/worker type %d: n = %lf \n", i, 0, res[i][0]);
				printf("ctx %d/worker type %d: n = %lf \n", i, 1, res[i][1]);
			}
			int res_rounded[nsched_ctxs][2];
			_round_double_to_int(nsched_ctxs, 2, res, res_rounded);
			for( i = 0; i < nsched_ctxs; i++)
			{
				printf("ctx %d/worker type %d: n = %d \n", i, 0, res_rounded[i][0]);
				printf("ctx %d/worker type %d: n = %d \n", i, 1, res_rounded[i][1]);
			}
			
			_redistribute_resources_in_ctxs(nsched_ctxs, 2, res_rounded, res);
			
			pthread_mutex_unlock(&act_hypervisor_mutex);
		}
	}		
}

struct hypervisor_policy lp_policy = {
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

