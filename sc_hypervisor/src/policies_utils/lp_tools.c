/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "sc_hypervisor_intern.h"
#include <starpu_config.h>

double sc_hypervisor_lp_get_nworkers_per_ctx(int nsched_ctxs, int ntypes_of_workers, double res[nsched_ctxs][ntypes_of_workers],
					     int total_nw[ntypes_of_workers], struct types_of_workers *tw, unsigned *in_sched_ctxs)
{
	unsigned *sched_ctxs = in_sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : in_sched_ctxs;
#ifdef STARPU_HAVE_GLPK_H
	double v[nsched_ctxs][ntypes_of_workers];
	double flops[nsched_ctxs];

/* 	unsigned nhierarchy_levels = sc_hypervisor_get_nhierarchy_levels(); */
/* 	if(nhierarchy_levels <= 1) */
	sc_hypervisor_update_resize_interval(sched_ctxs, nsched_ctxs, total_nw[0]);

	int nw = tw->nw;
	int i = 0;
	struct sc_hypervisor_wrapper* sc_w;

	for(i = 0; i < nsched_ctxs; i++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
		int w;
		for(w = 0; w < nw; w++)
			v[i][w] = sc_hypervisor_get_speed(sc_w, sc_hypervisor_get_arch_for_index(w, tw));

		double ready_flops = starpu_sched_ctx_get_nready_flops(sc_w->sched_ctx);
		unsigned nhierarchy_levels = sc_hypervisor_get_nhierarchy_levels();
		if(nhierarchy_levels > 1)
			ready_flops = sc_hypervisor_get_nready_flops_of_all_sons_of_sched_ctx(sc_w->sched_ctx);

		int nready_tasks = starpu_sched_ctx_get_nready_tasks(sc_w->sched_ctx);

		if(sc_w->to_be_sized)
		{
			flops[i] = sc_w->remaining_flops/1000000000.0; /* in gflops*/
			sc_w->to_be_sized = 0;
		}
		else
		{
			if(nhierarchy_levels > 1)
				flops[i] = sc_w->remaining_flops/1000000000.0; /* in gflops*/
			else
			if(sc_w->remaining_flops < 0.0)
				flops[i] = ready_flops/1000000000.0; /* in gflops*/
			else
			{
				if((ready_flops/1000000000.0) <= 0.000002)
					flops[i] = 0.0;
				else
					flops[i] = sc_w->remaining_flops/1000000000.0; /* in gflops*/
			}
		}
		if(flops[i] < 0.0)
			flops[i] = 0.0;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
		printf("%u: flops %lf remaining flops %lf ready flops %lf nready_tasks %d\n",
		       sched_ctxs[i], flops[i], sc_w->remaining_flops/1000000000, ready_flops/1000000000, nready_tasks);
#endif

	}
	sc_hypervisor_check_if_consider_max(tw);
	int w;
	for(w = 0; w < nw; w++)
	{
		double avg_speed = sc_hypervisor_get_avg_speed(sc_hypervisor_get_arch_for_index(w, tw));
		if(avg_speed != -1.0)
		{
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("avg_speed for cpus is %lf \n", avg_speed);
#endif
			unsigned consider_max_for_all = 0;
			for(i = 0; i < nsched_ctxs; i++)
			{
				sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);

				if(!sc_w->consider_max)
				{
#ifdef STARPU_SC_HYPERVISOR_DEBUG
					printf("ctx %u: current speed is %lf and compare speed is min %lf max %lf\n", sched_ctxs[i], v[i][w], (0.1*avg_speed), (2*avg_speed));
#endif
					if(v[i][w] < 0.1*avg_speed || v[i][w] > 2*avg_speed)
					{
						sc_w->consider_max = 1;
						consider_max_for_all = 1;
					}
#ifdef STARPU_SC_HYPERVISOR_DEBUG
					printf("ctx %u consider max %d \n", sched_ctxs[i], sc_w->consider_max);
#endif
				}

			}
			if(consider_max_for_all)
			{
				for(i = 0; i < nsched_ctxs; i++)
				{
					sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
					sc_w->consider_max = 1;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
					printf("ctx %u consider max %d anyway \n", sched_ctxs[i], sc_w->consider_max);
#endif
				}
			}

		}
	}

	if(nsched_ctxs == 1)
	{
		int w;
		for(w = 0; w < nw; w++)
			res[0][w] = total_nw[w];
		double optimal_v = 0.0;
#ifdef STARPU_USE_CUDA
		optimal_v = res[0][0] * v[0][0] + res[0][1]* v[0][1];
#else
		optimal_v = res[0][0] * v[0][0];
#endif //STARPU_USE_CUDA
		_set_optimal_v(sched_ctxs[0], optimal_v);
		return 1.0;
	}


	unsigned tmp_sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	double tmp_flops[STARPU_NMAX_SCHED_CTXS];
	double tmp_v[STARPU_NMAX_SCHED_CTXS][ntypes_of_workers];
	double tmp_res[STARPU_NMAX_SCHED_CTXS][ntypes_of_workers];
	int tmp_nsched_ctxs = 0;
	for(i = 0; i < nsched_ctxs; i++)
	{
		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs[i]);
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
		if(config->max_nworkers != 0 || !sc_w->consider_max)
		{
			tmp_sched_ctxs[tmp_nsched_ctxs] = sched_ctxs[i];
			tmp_flops[tmp_nsched_ctxs] = flops[i];
			int w;
			for(w = 0; w < ntypes_of_workers; w++)
				tmp_v[tmp_nsched_ctxs][w] = v[i][w];
			tmp_nsched_ctxs++;
		}
	}
	if(tmp_nsched_ctxs == 0)
		return -1.0;
	double ret = sc_hypervisor_lp_simulate_distrib_flops(tmp_nsched_ctxs, ntypes_of_workers, tmp_v, tmp_flops, tmp_res, total_nw, tmp_sched_ctxs, -1.0);

	int j;
	for(i = 0; i < nsched_ctxs; i++)
	{
		unsigned found = 0;
		for(j = 0; j < tmp_nsched_ctxs; j++)
		{
			if(sched_ctxs[i] == tmp_sched_ctxs[j])
			{
				int w;
				for(w = 0; w < ntypes_of_workers; w++)
					res[i][w] = tmp_res[j][w];
				found = 1;
				break;
			}
		}
		if(!found)
		{
			int w;
			for(w = 0; w < ntypes_of_workers; w++)
				res[i][w] = 0.0;
		}
	}

	double vmax = 0.0;

	if(ret != 0.0)
	{
		/* redo the lp after cleaning out the contexts that got all the max workers required */
		unsigned selected_sched_ctxs[STARPU_NMAX_SCHED_CTXS];
		double selected_flops[STARPU_NMAX_SCHED_CTXS];
		double selected_v[STARPU_NMAX_SCHED_CTXS][ntypes_of_workers];
		int nselected = 0;
		int available_cpus = total_nw[0];
		int used_cpus = 0;

		for(i = 0; i < nsched_ctxs; i++)
		{
			struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs[i]);
			if(res[i][0] < config->max_nworkers && config->max_nworkers != 0 && flops[i] > 0.0001)
			{
				selected_flops[nselected] = flops[i];
				selected_v[nselected][0] = v[i][0];
				selected_sched_ctxs[nselected++] = sched_ctxs[i];
			}
			else
				available_cpus -= res[i][0];
			used_cpus += res[i][0];
		}

		if(used_cpus < 0.8 * total_nw[0] && nselected > 1)
		{
			double old_ret = ret;

			if(nselected <= 0 || nselected == nsched_ctxs)
			{
				nselected = nsched_ctxs;
				for(i = 0; i < nsched_ctxs; i++)
				{
					selected_flops[i] = flops[i];
					selected_v[i][0] = v[i][0];
					selected_sched_ctxs[i] = sched_ctxs[i];
				}
			}
			else
				total_nw[0] = available_cpus;

			double selected_res[nselected][ntypes_of_workers];
			ret = sc_hypervisor_lp_simulate_distrib_flops(nselected, ntypes_of_workers, selected_v, selected_flops, selected_res, total_nw, selected_sched_ctxs, ret);

			if(ret != 0)
			{
				int j;
				for(i = 0; i < nsched_ctxs; i++)
				{
					for(j = 0; j < nselected; j++)
					{
						if(sched_ctxs[i] == selected_sched_ctxs[j])
						{
							res[i][0] = selected_res[j][0];
						}
					}
				}
			}
			else
				ret = old_ret;
		}

	}

	/* if the lp could not give any workers to any context
	   just split the workers btw the contexts */
	if(ret == 0.0)
	{
		double rand_res[nw];
		int w;
		for(w = 0; w < nw; w++)
			rand_res[w] = total_nw[w]/nsched_ctxs;
		int s;
		for(s = 0; s < nsched_ctxs; s++)
			for(w = 0; w < nw; w++)
				res[s][w] = rand_res[w];
	}

	else
	/* keep the first speed */
//	if(ret != 0.0)
	{
		vmax = 1 / ret;
	}
	double optimal_v = 0.0;
	for(i = 0; i < nsched_ctxs; i++)
	{
#ifdef STARPU_USE_CUDA
		optimal_v = res[i][0] * v[i][0] + res[i][1]* v[i][1];
#else
		optimal_v = res[i][0] * v[i][0];
#endif //STARPU_USE_CUDA
		int w;
		unsigned no_workers = 1;
		for(w = 0; w < nw; w++)
		{
			if(res[i][w] != 0.0)
			{
				no_workers = 0;
				break;
			}
		}

		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);

/* if the hypervisor gave 0 workers to a context but the context still
   has some last flops or a ready task that does not even have any flops
   we give a worker (in shared mode) to the context in order to leave him
   finish its work = we give -1.0 value instead of 0.0 and further on in
   the distribution function we take this into account and revert the variable
   to its 0.0 value */
//		if(no_workers && (flops[i] != 0.0 || sc_w->nready_tasks > 0))
		if(no_workers)
		{
			for(w = 0; w < nw; w++)
				res[i][w] = -1.0;
		}

//			if(optimal_v != 0.0)
		_set_optimal_v(sched_ctxs[i], optimal_v);
	}

	return vmax;
#else//STARPU_HAVE_GLPK_H
	return 0.0;
#endif//STARPU_HAVE_GLPK_H
}

double sc_hypervisor_lp_get_tmax(int nworkers, int *workers)
{
	struct types_of_workers *tw = sc_hypervisor_get_types_of_workers(workers, nworkers);
        int nw = tw->nw;

        int total_nw[nw];
        sc_hypervisor_group_workers_by_type(tw, total_nw);

	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	double res[nsched_ctxs][nw];
	double ret = sc_hypervisor_lp_get_nworkers_per_ctx(nsched_ctxs, nw, res, total_nw, tw, NULL) * 1000.0;
	free(tw);
	return ret;
}

void sc_hypervisor_lp_round_double_to_int(int ns, int nw, double res[ns][nw], int res_rounded[ns][nw])
{
	int s, w;
	double left_res[nw];
	for(w = 0; w < nw; w++)
		left_res[w] = 0.0;
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
				   int res_rounded[ns][nw], double res[ns][nw], struct types_of_workers *tw)
{
	int w;
	double target_res = 0.0;
	for(w = 0; w < nw; w++)
	{
		target_res += res[sched_ctx_idx][w];
		if(res[sched_ctx_idx][w] == -1.0) res[sched_ctx_idx][w] = 0.0;
	}

	for(w = 0; w < nw; w++)
	{
		enum starpu_worker_archtype arch = sc_hypervisor_get_arch_for_index(w, tw);

		if(arch == STARPU_CPU_WORKER)
		{
			int nworkers_ctx = sc_hypervisor_get_nworkers_ctx(sched_ctx, arch);
			if(nworkers_ctx > res_rounded[sched_ctx_idx][w])
			{
				int nworkers_to_move = nworkers_ctx - res_rounded[sched_ctx_idx][w];
				int *workers_to_move = sc_hypervisor_get_idlest_workers(sched_ctx, &nworkers_to_move, arch);
				int i;
				if(target_res < 0.0 && nworkers_to_move > 0)
				{
					tmp_workers_add[w][tmp_nw_add[w]++] = workers_to_move[0];
					for(i = 1; i < nworkers_to_move; i++)
						tmp_workers_move[w][tmp_nw_move[w]++] = workers_to_move[i];
				}
				else
				{
					for(i = 0; i < nworkers_to_move; i++)
						tmp_workers_move[w][tmp_nw_move[w]++] = workers_to_move[i];
				}
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
					int *workers_to_move = sc_hypervisor_get_idlest_workers(sched_ctx, &x, arch);
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
					int *workers_to_move = sc_hypervisor_get_idlest_workers(sched_ctx, &x, arch);
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
				int res_rounded[ns][nw], double res[ns][nw], struct types_of_workers *tw)
{
	int w;
	int j = 0, k = 0;
	for(w = 0; w < nw; w++)
	{
		enum starpu_worker_archtype arch = sc_hypervisor_get_arch_for_index(w, tw);

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
		if((diff > 0.3 || needed > 0.3) && tmp_nw_add[w] > 0)
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

void sc_hypervisor_lp_redistribute_resources_in_ctxs(int ns, int nw, int res_rounded[ns][nw], double res[ns][nw], unsigned *sched_ctxs, struct types_of_workers *tw)
{
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
					      tmp_nw_add, tmp_workers_add, res_rounded,
					      res, tw);
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
							   res_rounded, res, tw);

				if(nw_move > 0)
				{
					sc_hypervisor_move_workers(sched_ctxs[s], sched_ctxs[s2], workers_move, nw_move, !(_sc_hypervisor_use_lazy_resize()));
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
			sc_hypervisor_remove_workers_from_sched_ctx(workers_move, nw_move, sched_ctxs[s], !(_sc_hypervisor_use_lazy_resize()));
	}
}
int _lp_get_unwanted_workers(int *workers_add, int nw_add, unsigned sched_ctx, int *workers_remove)
{
	int nw_remove = 0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		int i;
		unsigned found = 0;
		for(i = 0; i < nw_add; i++)
		{
			if(worker == workers_add[i])
			{
				found = 1;
				break;
			}
		}
		if(!found)
			workers_remove[nw_remove++] = worker;
	}
	return nw_remove;
}

void sc_hypervisor_lp_distribute_resources_in_ctxs(unsigned* sched_ctxs, int ns, int nw, int res_rounded[ns][nw],
						   double res[ns][nw], int *workers, int nworkers, struct types_of_workers *tw)
{
	int s, w;
	int start[nw];
	for(w = 0; w < nw; w++)
		start[w] = 0;
	for(s = 0; s < ns; s++)
	{
		int workers_add[STARPU_NMAXWORKERS];
                int nw_add = 0;
		double target_res = 0.0;
		for(w = 0; w < nw; w++)
		{
			target_res += res[s][w];
			if(res[s][w] == -1.0) res[s][w] = 0.0;
		}

		for(w = 0; w < nw; w++)
		{
			enum starpu_worker_archtype arch = sc_hypervisor_get_arch_for_index(w, tw);

			if(arch == STARPU_CPU_WORKER)
			{
				int nworkers_to_add = res_rounded[s][w];
				if(target_res < 0.0)
				{
					nworkers_to_add=1;
					int old_start = start[w];
					if(start[w] != 0)
						start[w]--;
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &nworkers_to_add, arch);
					start[w] = old_start;
					int i;
					for(i = 0; i < nworkers_to_add; i++)
					{
						workers_add[nw_add++] = workers_to_add[i];
					}
					free(workers_to_add);
				}
				else
				{
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &nworkers_to_add, arch);
					int i;
					for(i = 0; i < nworkers_to_add; i++)
						workers_add[nw_add++] = workers_to_add[i];
					free(workers_to_add);
				}
			}
			else
			{
				double nworkers_to_add = res[s][w];
				int x = floor(nworkers_to_add);
				double x_double = (double)x;
				double diff = nworkers_to_add - x_double;
				if(diff == 0.0)
				{
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &x, arch);
					int i;
					for(i = 0; i < x; i++)
						workers_add[nw_add++] = workers_to_add[i];
					free(workers_to_add);
				}
				else
				{
					x+=1;
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &x, arch);
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
//		sc_hypervisor_start_resize(sched_ctxs[s]);
		sc_hypervisor_add_workers_to_sched_ctx(workers_add, nw_add, sched_ctxs[s]);
		int workers_remove[STARPU_NMAXWORKERS];
		int nw_remove = _lp_get_unwanted_workers(workers_add, nw_add, sched_ctxs[s], workers_remove);
		sc_hypervisor_remove_workers_from_sched_ctx(workers_remove, nw_remove, sched_ctxs[s], !(_sc_hypervisor_use_lazy_resize()));
	}
}

void sc_hypervisor_lp_distribute_floating_no_resources_in_ctxs(unsigned* sched_ctxs, int ns, int nw, double res[ns][nw],
							       int *workers, int nworkers, struct types_of_workers *tw)
{
	int s, w;
	int start[nw];
	for(w = 0; w < nw; w++)
		start[w] = 0;
	for(s = 0; s < ns; s++)
	{
		int workers_add[STARPU_NMAXWORKERS];
                int nw_add = 0;
		double target_res = 0.0;
		for(w = 0; w < nw; w++)
		{
			target_res += res[s][w];
			if(res[s][w] == -1.0) res[s][w] = 0.0;
		}

		for(w = 0; w < nw; w++)
		{
			enum starpu_worker_archtype arch = sc_hypervisor_get_arch_for_index(w, tw);

			if(arch == STARPU_CPU_WORKER)
			{
				int nworkers_to_add = ceil(res[s][w]);
				double ceil_double = (double)nworkers_to_add;
				double diff = ceil_double - res[s][w];

				if(target_res < 0.0)
				{
					nworkers_to_add=1;
					int old_start = start[w];
					if(start[w] != 0)
						start[w]--;
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &nworkers_to_add, arch);
					start[w] = old_start;
					int i;
					for(i = 0; i < nworkers_to_add; i++)
					{
						workers_add[nw_add++] = workers_to_add[i];
					}
					free(workers_to_add);
				}
				else
				{
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &nworkers_to_add, arch);
					int i;
					for(i = 0; i < nworkers_to_add; i++)
						workers_add[nw_add++] = workers_to_add[i];
					free(workers_to_add);
				}
				if(diff != 0.0)
					start[w]--;
			}
			else
			{
				double nworkers_to_add = res[s][w];
				int x = floor(nworkers_to_add);
				double x_double = (double)x;
				double diff = nworkers_to_add - x_double;
				if(diff == 0.0)
				{
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &x, arch);
					int i;
					for(i = 0; i < x; i++)
						workers_add[nw_add++] = workers_to_add[i];
					free(workers_to_add);
				}
				else
				{
					x+=1;
					int *workers_to_add = sc_hypervisor_get_idlest_workers_in_list(&start[w], workers, nworkers, &x, arch);
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
//		sc_hypervisor_start_resize(sched_ctxs[s]);
		sc_hypervisor_add_workers_to_sched_ctx(workers_add, nw_add, sched_ctxs[s]);
		int workers_remove[STARPU_NMAXWORKERS];
		int nw_remove = _lp_get_unwanted_workers(workers_add, nw_add, sched_ctxs[s], workers_remove);
		sc_hypervisor_remove_workers_from_sched_ctx(workers_remove, nw_remove, sched_ctxs[s], !(_sc_hypervisor_use_lazy_resize()));
	}
}

/* nw = all the workers (either in a list or on all machine) */
void sc_hypervisor_lp_place_resources_in_ctx(int ns, int nw, double w_in_s[ns][nw], unsigned *sched_ctxs_input, int *workers_input, unsigned do_size, struct types_of_workers *tw)
{
	int w, s;
	int ntypes_of_workers = tw->nw;
	double nworkers[ns][ntypes_of_workers];
	int nworkers_rounded[ns][ntypes_of_workers];
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < ntypes_of_workers; w++)
		{
			nworkers[s][w] = 0.0;
			nworkers_rounded[s][w] = 0;
		}

	}

	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			enum starpu_worker_archtype arch = starpu_worker_get_type(w);
			int idx = sc_hypervisor_get_index_for_arch(arch, tw);
			nworkers[s][idx] += w_in_s[s][w];

			if(arch == STARPU_CUDA_WORKER)
			{
				if(w_in_s[s][w] >= 0.3)
					nworkers_rounded[s][idx]++;
			}
			else
			{
				if(w_in_s[s][w] > 0.5)
					nworkers_rounded[s][idx]++;
			}
		}
	}

	if(!do_size)
		sc_hypervisor_lp_redistribute_resources_in_ctxs(ns, ntypes_of_workers, nworkers_rounded, nworkers, sched_ctxs_input, tw);
	else
	{
		unsigned *current_sched_ctxs = sched_ctxs_input == NULL ? sc_hypervisor_get_sched_ctxs() : sched_ctxs_input;

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
			sc_hypervisor_lp_redistribute_resources_in_ctxs(ns, ntypes_of_workers, nworkers_rounded, nworkers, current_sched_ctxs, tw);
		else
			sc_hypervisor_lp_distribute_resources_in_ctxs(current_sched_ctxs, ns, ntypes_of_workers, nworkers_rounded, nworkers, workers_input, nw, tw);
	}
	return;
}

void sc_hypervisor_lp_share_remaining_resources(int ns, unsigned *sched_ctxs,  int nworkers, int *workers)
{
	int s, w, worker, nw = 0;
	int remaining_workers[nworkers];
	for(w = 0; w < nworkers; w++)
	{
		worker = workers == NULL ? w : workers[w];
		unsigned found = 0;
		for(s = 0; s < ns; s++)
		{
			if(starpu_sched_ctx_contains_worker(worker, sched_ctxs[s]))
			{
				found = 1;
				break;
			}
		}
		if(!found)
		{
			remaining_workers[nw++] = worker;
		}
	}

	if(nw > 0)
	{
		for(s = 0; s < ns; s++)
		{
			for(w = 0; w < nw; w++)
				_sc_hypervisor_allow_compute_idle(sched_ctxs[s], remaining_workers[w], 0);
			sc_hypervisor_add_workers_to_sched_ctx(remaining_workers, nw, sched_ctxs[s]);
		}
	}
}

double sc_hypervisor_lp_find_tmax(double t1, double t2)
{
	return t1 + ((t2 - t1)/2);
}
