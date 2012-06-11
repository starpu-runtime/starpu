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

//#include "policy_utils.h"


/*                                                                                                                                                                                                                  
 * GNU Linear Programming Kit backend                                                                                                                                                                               
 */
#include "lp_tools.h"

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
		int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
//		int nsched_ctxs = 3;
		
		double res[nsched_ctxs][2];

		int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{ 
			double vmax = _lp_get_nworkers_per_ctx(nsched_ctxs, 2, res);
			if(vmax != 0.0)
			{
/* 			for( i = 0; i < nsched_ctxs; i++) */
/* 			{ */
/* 				printf("ctx %d/worker type %d: n = %lf \n", i, 0, res[i][0]); */
/* 				printf("ctx %d/worker type %d: n = %lf \n", i, 1, res[i][1]); */
/* 			} */
				int res_rounded[nsched_ctxs][2];
				_round_double_to_int(nsched_ctxs, 2, res, res_rounded);
  /*     		for( i = 0; i < nsched_ctxs; i++) */
/* 			{ */
/* 				printf("ctx %d/worker type %d: n = %d \n", i, 0, res_rounded[i][0]); */
/* 				printf("ctx %d/worker type %d: n = %d \n", i, 1, res_rounded[i][1]); */
/* 			} */
				
				_redistribute_resources_in_ctxs(nsched_ctxs, 2, res_rounded, res);
				
				pthread_mutex_unlock(&act_hypervisor_mutex);
			}
		}
	}		
}

#ifdef HAVE_GLPK_H
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

