/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  INRIA
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

#include <math.h>
#include "sc_hypervisor_lp.h"
#include "sc_hypervisor_policy.h"
#include <starpu_config.h>

#ifdef STARPU_HAVE_GLPK_H

double _lp_compute_nworkers_per_ctx(int ns, int nw, double v[ns][nw], double flops[ns], double res[ns][nw], int  total_nw[nw])
{
	int integer = 1;
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
			if (integer)
			{
				glp_set_col_kind(lp, n, GLP_IV);
				glp_set_col_bnds(lp, n, GLP_LO, 0, 0);
			}
			else
				glp_set_col_bnds(lp, n, GLP_LO, 0.0, 0.0);
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
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, total_nw[0], total_nw[0]);

		/*sum(all cpus) = 9*/
		if(w == 1)
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, total_nw[1], total_nw[1]);
	}

	STARPU_ASSERT(n == ne);

	glp_load_matrix(lp, ne-1, ia, ja, ar);

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	int ret = glp_simplex(lp, &parm);
	if (ret)
        {
                printf("error in simplex\n");
		glp_delete_prob(lp);
                lp = NULL;
                return 0.0;
        }

	int stat = glp_get_prim_stat(lp);
        /* if we don't have a solution return */
        if(stat == GLP_NOFEAS)
        {
                glp_delete_prob(lp);
//              printf("no_sol in tmax = %lf\n", tmax);                                                                                                                                                             
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
//                      printf("no int sol in tmax = %lf\n", tmax);                                                                                                                                                 
                        glp_delete_prob(lp);
                        lp = NULL;
                        return 0.0;
                }
        }

	double vmax = glp_get_obj_val(lp);

	n = 1;
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			if (integer)
                                res[s][w] = (double)glp_mip_col_val(lp, n);
			else
				res[s][w] = glp_get_col_prim(lp, n);
//			printf("%d/%d: res %lf flops = %lf v = %lf\n", w,s, res[s][w], flops[s], v[s][w]);
			n++;
		}
	}

	glp_delete_prob(lp);
	return vmax;
}

#endif //STARPU_HAVE_GLPK_H

double _lp_get_nworkers_per_ctx(int nsched_ctxs, int ntypes_of_workers, double res[nsched_ctxs][ntypes_of_workers], int total_nw[ntypes_of_workers])
{
	int *sched_ctxs = sc_hypervisor_get_sched_ctxs();
#ifdef STARPU_HAVE_GLPK_H
	double v[nsched_ctxs][ntypes_of_workers];
	double flops[nsched_ctxs];

	int i = 0;
	struct sc_hypervisor_wrapper* sc_w;
	for(i = 0; i < nsched_ctxs; i++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
#ifdef STARPU_USE_CUDA
		int ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
		if(ncuda != 0)
		{
			v[i][0] = sc_hypervisor_get_velocity(sc_w, STARPU_CUDA_WORKER);
			v[i][1] = sc_hypervisor_get_velocity(sc_w, STARPU_CPU_WORKER);
		}
		else
			v[i][0] = sc_hypervisor_get_velocity(sc_w, STARPU_CPU_WORKER);
#else
		v[i][0] = sc_hypervisor_get_velocity(sc_w, STARPU_CPU_WORKER);
#endif // STARPU_USE_CUDA
		flops[i] = sc_w->remaining_flops/1000000000; //sc_w->total_flops/1000000000; /* in gflops*/
//		printf("%d: flops %lf\n", sched_ctxs[i], flops[i]);
	}

	return 1/_lp_compute_nworkers_per_ctx(nsched_ctxs, ntypes_of_workers, v, flops, res, total_nw);
#else//STARPU_HAVE_GLPK_H
	return 0.0;
#endif//STARPU_HAVE_GLPK_H
}

double _lp_get_tmax(int nw, int *workers)
{
	int ntypes_of_workers = 2;
	int total_nw[ntypes_of_workers];
	_get_total_nw(workers, nw, 2, total_nw);

	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	double res[nsched_ctxs][ntypes_of_workers];
	return _lp_get_nworkers_per_ctx(nsched_ctxs, ntypes_of_workers, res, total_nw) * 1000;
}

void _lp_round_double_to_int(int ns, int nw, double res[ns][nw], int res_rounded[ns][nw])
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
			else 
				res_rounded[s][w] = x;
		}
	}
}

void _lp_find_workers_to_give_away(int nw, int ns, unsigned sched_ctx, int sched_ctx_idx, 
				  int tmp_nw_move[nw], int tmp_workers_move[nw][STARPU_NMAXWORKERS], 
				  int tmp_nw_add[nw], int tmp_workers_add[nw][STARPU_NMAXWORKERS],
				  int res_rounded[ns][nw], double res[ns][nw])
{
	int w;
	for(w = 0; w < nw; w++)
	{
		enum starpu_archtype arch = STARPU_ANY_WORKER;
		if(w == 0) arch = STARPU_CUDA_WORKER;
		if(w == 1) arch = STARPU_CPU_WORKER;
		
		
		if(w == 1)
		{
			int nworkers_ctx = sc_hypervisor_get_nworkers_ctx(sched_ctx, arch);
			if(nworkers_ctx > res_rounded[sched_ctx_idx][w])
			{
				int nworkers_to_move = nworkers_ctx - res_rounded[sched_ctx_idx][w];
				int *workers_to_move = _get_first_workers(sched_ctx, &nworkers_to_move, arch);
				int i;
				for(i = 0; i < nworkers_to_move; i++)
					tmp_workers_move[w][tmp_nw_move[w]++] = workers_to_move[i];
				free(workers_to_move);
			}
		}
		else
		{
			double nworkers_ctx = sc_hypervisor_get_nworkers_ctx(sched_ctx, arch) * 1.0;
			if(nworkers_ctx > res[sched_ctx_idx][w])
			{
				double nworkers_to_move = nworkers_ctx - res[sched_ctx_idx][w];
				int x = floor(nworkers_to_move);
				double x_double = (double)x;
				double diff = nworkers_to_move - x_double;
				if(diff == 0.0)
				{
					int *workers_to_move = _get_first_workers(sched_ctx, &x, arch);
					if(x > 0)
					{
						int i;
						for(i = 0; i < x; i++)
							tmp_workers_move[w][tmp_nw_move[w]++] = workers_to_move[i];
						
					}
					free(workers_to_move);
				}
				else
				{
					x+=1;
					int *workers_to_move = _get_first_workers(sched_ctx, &x, arch);
					if(x > 0)
					{
						int i;
						for(i = 0; i < x-1; i++)
							tmp_workers_move[w][tmp_nw_move[w]++] = workers_to_move[i];
						
						if(diff > 0.8)
							tmp_workers_move[w][tmp_nw_move[w]++] = workers_to_move[x-1];
						else
							if(diff > 0.3)
								tmp_workers_add[w][tmp_nw_add[w]++] = workers_to_move[x-1];
						
					}
					free(workers_to_move);
				}
			}
		}
	}
}

void _lp_find_workers_to_accept(int nw, int ns, unsigned sched_ctx, int sched_ctx_idx, 
				int tmp_nw_move[nw], int tmp_workers_move[nw][STARPU_NMAXWORKERS], 
				int tmp_nw_add[nw], int tmp_workers_add[nw][STARPU_NMAXWORKERS],
				int *nw_move, int workers_move[STARPU_NMAXWORKERS], 
				int *nw_add, int workers_add[STARPU_NMAXWORKERS],
				int res_rounded[ns][nw], double res[ns][nw])
{
	int w;
	int j = 0, k = 0;
	for(w = 0; w < nw; w++)
	{
		enum starpu_archtype arch = STARPU_ANY_WORKER;
		if(w == 0) arch = STARPU_CUDA_WORKER;
		if(w == 1) arch = STARPU_CPU_WORKER;
		
		int nw_ctx2 = sc_hypervisor_get_nworkers_ctx(sched_ctx, arch);
		int nw_needed = res_rounded[sched_ctx_idx][w] - nw_ctx2;
		
		if( nw_needed > 0 && tmp_nw_move[w] > 0)
		{
			*nw_move += nw_needed >= tmp_nw_move[w] ? tmp_nw_move[w] : nw_needed;
			int i = 0;
			for(i = 0; i < STARPU_NMAXWORKERS; i++)
			{
				if(tmp_workers_move[w][i] != -1)
				{
					workers_move[j++] = tmp_workers_move[w][i];
					tmp_workers_move[w][i] = -1;
					if(j == *nw_move)
						break;
				}
			}
			tmp_nw_move[w] -=  *nw_move;
		}
		
		
		double needed = res[sched_ctx_idx][w] - (nw_ctx2 * 1.0);
		int x = floor(needed);
		double x_double = (double)x;
		double diff = needed - x_double;
		if(diff > 0.3 && tmp_nw_add[w] > 0)
		{
			*nw_add = tmp_nw_add[w];
			int i = 0;
			for(i = 0; i < STARPU_NMAXWORKERS; i++)
			{
				if(tmp_workers_add[w][i] != -1)
				{
					workers_add[k++] = tmp_workers_add[w][i];
					tmp_workers_add[w][i] = -1;
					if(k == *nw_add)
						break;
				}
			}
			tmp_nw_add[w] -=  *nw_add;
		}
	}
}

void _lp_find_workers_to_remove(int nw, int tmp_nw_move[nw], int tmp_workers_move[nw][STARPU_NMAXWORKERS], 
				int *nw_move, int workers_move[STARPU_NMAXWORKERS])
{
	int w;
	for(w = 0; w < nw; w++)
	{
		if(tmp_nw_move[w] > 0)
		{
			*nw_move += tmp_nw_move[w];
			int i = 0, j = 0;
			for(i = 0; i < STARPU_NMAXWORKERS; i++)
			{
				if(tmp_workers_move[w][i] != -1)
				{
					workers_move[j++] = tmp_workers_move[w][i];
					tmp_workers_move[w][i] = -1;
					if(j == *nw_move)
						break;
				}
			}
			
		}
	}
}

void _lp_redistribute_resources_in_ctxs(int ns, int nw, int res_rounded[ns][nw], double res[ns][nw])
{
	int *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int s, s2, w;
	for(s = 0; s < ns; s++)
	{
		int tmp_workers_move[nw][STARPU_NMAXWORKERS];
		int tmp_nw_move[nw];

		int tmp_workers_add[nw][STARPU_NMAXWORKERS];
		int tmp_nw_add[nw];
		

		for(w = 0; w < nw; w++)		
		{
			tmp_nw_move[w] = 0;
			tmp_nw_add[w] = 0;
			int i;
			for(i = 0; i < STARPU_NMAXWORKERS; i++)
			{
				tmp_workers_move[w][i] = -1;
				tmp_workers_add[w][i] = -1;
			}
		}

		/* find workers that ctx s has to give away */
		_lp_find_workers_to_give_away(nw, ns, sched_ctxs[s], s, 
					      tmp_nw_move, tmp_workers_move, 
					      tmp_nw_add, tmp_workers_add, res_rounded, res);

		for(s2 = 0; s2 < ns; s2++)
		{
			if(sched_ctxs[s2] != sched_ctxs[s])
			{
				/* find workers that ctx s2 wants to accept from ctx s 
				   the rest of it will probably accepted by another ctx */
				int workers_move[STARPU_NMAXWORKERS];
				int nw_move = 0;
				
				int workers_add[STARPU_NMAXWORKERS];
				int nw_add = 0;
				

				_lp_find_workers_to_accept(nw, ns, sched_ctxs[s2], s2, 
							   tmp_nw_move, tmp_workers_move, 
							   tmp_nw_add, tmp_workers_add,
							   &nw_move, workers_move, 
							   &nw_add, workers_add,
							   res_rounded, res);
				
				if(nw_move > 0)
				{
					sc_hypervisor_move_workers(sched_ctxs[s], sched_ctxs[s2], workers_move, nw_move, 0);
					nw_move = 0;
				}

				if(nw_add > 0)
				{
					sc_hypervisor_add_workers_to_sched_ctx(workers_add, nw_add, sched_ctxs[s2]);
					nw_add = 0;
				}
			}
		}

		/* if there are workers that weren't accepted by anyone but ctx s wants
		   to get rid of them just remove them from ctx s */
		int workers_move[STARPU_NMAXWORKERS];
		int nw_move = 0;
				
		_lp_find_workers_to_remove(nw, tmp_nw_move, tmp_workers_move, 
					   &nw_move, workers_move);
		if(nw_move > 0)
			sc_hypervisor_remove_workers_from_sched_ctx(workers_move, nw_move, sched_ctxs[s], 0);
	}
}

void _lp_distribute_resources_in_ctxs(int* sched_ctxs, int ns, int nw, int res_rounded[ns][nw], double res[ns][nw], int *workers, int nworkers)
{
	unsigned current_nworkers = workers == NULL ? starpu_worker_get_count() : (unsigned)nworkers;
	int s, w;
	int start[nw];
	for(w = 0; w < nw; w++)
		start[w] = 0;
	for(s = 0; s < ns; s++)
	{
		int workers_add[STARPU_NMAXWORKERS];
                int nw_add = 0;
		
		for(w = 0; w < nw; w++)
		{
			enum starpu_archtype arch;

#ifdef STARPU_USE_CUDA
			int ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
			if(ncuda != 0)
			{
				if(w == 0) arch = STARPU_CUDA_WORKER;
				if(w == 1) arch = STARPU_CPU_WORKER;
			}
			else
				if(w == 0) arch = STARPU_CPU_WORKER;
#else
			if(w == 0) arch = STARPU_CPU_WORKER;
#endif //STARPU_USE_CUDA
			if(w == 1)
			{
				int nworkers_to_add = res_rounded[s][w];
				int *workers_to_add = _get_first_workers_in_list(&start[w], workers, current_nworkers, &nworkers_to_add, arch);
				int i;
				for(i = 0; i < nworkers_to_add; i++)
					workers_add[nw_add++] = workers_to_add[i];
				free(workers_to_add);
			}
			
			else
			{
				double nworkers_to_add = res[s][w];
				int x = floor(nworkers_to_add);
				double x_double = (double)x;
				double diff = nworkers_to_add - x_double;
				if(diff == 0.0)
				{
					int *workers_to_add = _get_first_workers_in_list(&start[w], workers, current_nworkers, &x, arch);
					int i;
					for(i = 0; i < x; i++)
						workers_add[nw_add++] = workers_to_add[i];
					free(workers_to_add);
				}
				else
				{
					x+=1;
					int *workers_to_add = _get_first_workers_in_list(&start[w], workers, current_nworkers, &x, arch);
					int i;
					if(diff >= 0.3)
						for(i = 0; i < x; i++)
							workers_add[nw_add++] = workers_to_add[i];
					else
						for(i = 0; i < x-1; i++)
							workers_add[nw_add++] = workers_to_add[i];

					free(workers_to_add);
				}
			}
		}
		if(nw_add > 0)
		{
			sc_hypervisor_add_workers_to_sched_ctx(workers_add, nw_add, sched_ctxs[s]);
			sc_hypervisor_start_resize(sched_ctxs[s]);
		}

//		sc_hypervisor_stop_resize(current_sched_ctxs[s]);
	}
}

/* nw = all the workers (either in a list or on all machine) */
void _lp_place_resources_in_ctx(int ns, int nw, double w_in_s[ns][nw], int *sched_ctxs_input, int *workers_input, unsigned do_size)
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
	
/* 	for(s = 0; s < ns; s++) */
/* 		printf("%d: cpus = %d gpus = %d \n", s, nworkers_rounded[s][1], nworkers_rounded[s][0]); */

	if(!do_size)
		_lp_redistribute_resources_in_ctxs(ns, 2, nworkers_rounded, nworkers);
	else
	{
		int *current_sched_ctxs = sched_ctxs_input == NULL ? sc_hypervisor_get_sched_ctxs() : sched_ctxs_input;

		unsigned has_workers = 0;
		for(s = 0; s < ns; s++)
		{
			int nworkers_ctx = sc_hypervisor_get_nworkers_ctx(current_sched_ctxs[s], 
										 STARPU_ANY_WORKER);
			if(nworkers_ctx != 0)
			{
				has_workers = 1;
				break;
			}
		}
		if(has_workers)
			_lp_redistribute_resources_in_ctxs(ns, 2, nworkers_rounded, nworkers);
		else
			_lp_distribute_resources_in_ctxs(current_sched_ctxs, ns, 2, nworkers_rounded, nworkers, workers_input, nw);
	}
	return;
}

double _lp_find_tmax(double t1, double t2)
{
	return t1 + ((t2 - t1)/2);
}
