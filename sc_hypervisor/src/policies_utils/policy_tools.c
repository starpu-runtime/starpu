/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#include "sc_hypervisor_policy.h"
#include "sc_hypervisor_intern.h"
#include "sc_hypervisor_lp.h"

static int _compute_priority(unsigned sched_ctx)
{
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctx);

	int total_priority = 0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		total_priority += config->priority[worker];
	}

	return total_priority;
}

/* find the context with the lowest priority */
unsigned sc_hypervisor_find_lowest_prio_sched_ctx(unsigned req_sched_ctx, int nworkers_to_move)
{
	int i;
	int highest_priority = -1;
	int current_priority = 0;
	unsigned sched_ctx = STARPU_NMAX_SCHED_CTXS;
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();


	struct sc_hypervisor_policy_config *config = NULL;

	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && sched_ctxs[i] != req_sched_ctx)
		{
			int nworkers = (int)starpu_sched_ctx_get_nworkers(sched_ctxs[i]);
			config  = sc_hypervisor_get_config(sched_ctxs[i]);
			if((nworkers + nworkers_to_move) <= config->max_nworkers)
			{
				current_priority = _compute_priority(sched_ctxs[i]);
				if (highest_priority < current_priority)
				{
					highest_priority = current_priority;
					sched_ctx = sched_ctxs[i];
				}
			}
		}
	}

	return sched_ctx;
}

int* sc_hypervisor_get_idlest_workers_in_list(int *start, int *workers, int nall_workers,  int *nworkers, enum starpu_worker_archtype arch)
{
	int *curr_workers = (int*)malloc((*nworkers)*sizeof(int));

	int w, worker;
	int nfound_workers = 0;
	for(w = 0; w < nall_workers; w++)
	{
		if(nfound_workers >= *nworkers)
			break;

		worker = workers == NULL ? w : workers[w];
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
		if(arch == STARPU_ANY_WORKER || curr_arch == arch)
		{
			if(w >= *start)
			{
				curr_workers[nfound_workers++] = worker;
				*start = w+1;
			}
		}
	}
	if(nfound_workers < *nworkers)
		*nworkers = nfound_workers;
	return curr_workers;
}

/* get first nworkers with the highest idle time in the context */
int* sc_hypervisor_get_idlest_workers(unsigned sched_ctx, int *nworkers, enum starpu_worker_archtype arch)
{
	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctx);

	int *curr_workers = (int*)malloc((*nworkers) * sizeof(int));
	int i;
	for(i = 0; i < *nworkers; i++)
		curr_workers[i] = -1;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	int index;
	int worker;
	int considered = 0;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);

	for(index = 0; index < *nworkers; index++)
	{
		while(workers->has_next(workers, &it))
		{
			considered = 0;
			worker = workers->get_next(workers, &it);
			enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
			if(arch == STARPU_ANY_WORKER || curr_arch == arch)
			{

				if(!config->fixed_workers[worker])
				{
					for(i = 0; i < index; i++)
					{
						if(curr_workers[i] == worker)
						{
							considered = 1;
							break;
						}
					}

					if(!considered)
					{
						/* the first iteration*/
						if(curr_workers[index] < 0)
						curr_workers[index] = worker;
						/* small priority worker is the first to leave the ctx*/
						else if(config->priority[worker] <
							config->priority[curr_workers[index]])
						curr_workers[index] = worker;
						/* if we don't consider priorities check for the workers
						   with the biggest idle time */
						else if(config->priority[worker] ==
							config->priority[curr_workers[index]])
						{
							double worker_idle_time = sc_w->current_idle_time[worker];
							double curr_worker_idle_time = sc_w->current_idle_time[curr_workers[index]];
							if(worker_idle_time > curr_worker_idle_time)
								curr_workers[index] = worker;
						}
					}
				}
			}
		}

		if(curr_workers[index] < 0)
		{
			*nworkers = index;
			break;
		}
	}

	return curr_workers;
}

/* get the number of workers in the context that are allowed to be moved (that are not fixed) */
int sc_hypervisor_get_movable_nworkers(struct sc_hypervisor_policy_config *config, unsigned sched_ctx, enum starpu_worker_archtype arch)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

	int potential_workers = 0;
	int worker;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
                if(arch == STARPU_ANY_WORKER || curr_arch == arch)
                {
			if(!config->fixed_workers[worker])
				potential_workers++;
		}
	}

	return potential_workers;
}

/* compute the number of workers that should be moved depending:
   - on the min/max number of workers in a context imposed by the user,
   - on the resource granularity imposed by the user for the resizing process*/
int sc_hypervisor_compute_nworkers_to_move(unsigned req_sched_ctx)
{
       	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(req_sched_ctx);
	int nworkers = (int)starpu_sched_ctx_get_nworkers(req_sched_ctx);
	int nworkers_to_move = 0;

	int potential_moving_workers = (int)sc_hypervisor_get_movable_nworkers(config, req_sched_ctx, STARPU_ANY_WORKER);
	if(potential_moving_workers > 0)
	{
		if(potential_moving_workers <= config->min_nworkers)
			/* if we have to give more than min better give it all */
			/* => empty ctx will block until having the required workers */
			nworkers_to_move = potential_moving_workers;
		else if(potential_moving_workers > config->max_nworkers)
		{
			if((potential_moving_workers - config->granularity) > config->max_nworkers)
//				nworkers_to_move = config->granularity;
				nworkers_to_move = potential_moving_workers;
			else
				nworkers_to_move = potential_moving_workers - config->max_nworkers;

		}
		else if(potential_moving_workers > config->granularity)
		{
			if((nworkers - config->granularity) > config->min_nworkers)
				nworkers_to_move = config->granularity;
			else
				nworkers_to_move = potential_moving_workers - config->min_nworkers;
		}
		else
		{
			int nfixed_workers = nworkers - potential_moving_workers;
			if(nfixed_workers >= config->min_nworkers)
				nworkers_to_move = potential_moving_workers;
			else
				nworkers_to_move = potential_moving_workers - (config->min_nworkers - nfixed_workers);
		}

		if((nworkers - nworkers_to_move) > config->max_nworkers)
			nworkers_to_move = nworkers - config->max_nworkers;
	}
	return nworkers_to_move;
}

unsigned sc_hypervisor_policy_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize, unsigned now)
{
	int ret = 1;
	if(force_resize)
		STARPU_PTHREAD_MUTEX_LOCK(&act_hypervisor_mutex);
	else
		ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		int nworkers_to_move = sc_hypervisor_compute_nworkers_to_move(sender_sched_ctx);
		if(nworkers_to_move > 0)
		{
			unsigned poor_sched_ctx = STARPU_NMAX_SCHED_CTXS;
			if(receiver_sched_ctx == STARPU_NMAX_SCHED_CTXS)
			{
				poor_sched_ctx = sc_hypervisor_find_lowest_prio_sched_ctx(sender_sched_ctx, (unsigned)nworkers_to_move);
			}
			else
			{
				poor_sched_ctx = receiver_sched_ctx;
				struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(poor_sched_ctx);
				int nworkers = (int)starpu_sched_ctx_get_nworkers(poor_sched_ctx);
				int nshared_workers = (int)starpu_sched_ctx_get_nshared_workers(sender_sched_ctx, poor_sched_ctx);
				if((nworkers+nworkers_to_move-nshared_workers) > config->max_nworkers)
					nworkers_to_move = nworkers > config->max_nworkers ? 0 : (config->max_nworkers - nworkers+nshared_workers);
				if(nworkers_to_move == 0) poor_sched_ctx = STARPU_NMAX_SCHED_CTXS;
			}
			if(poor_sched_ctx != STARPU_NMAX_SCHED_CTXS)
			{
				int *workers_to_move = sc_hypervisor_get_idlest_workers(sender_sched_ctx, &nworkers_to_move, STARPU_ANY_WORKER);
				sc_hypervisor_move_workers(sender_sched_ctx, poor_sched_ctx, workers_to_move, nworkers_to_move, now);

				struct sc_hypervisor_policy_config *new_config = sc_hypervisor_get_config(poor_sched_ctx);
				int i;
				for(i = 0; i < nworkers_to_move; i++)
					new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;

				free(workers_to_move);
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
		return 1;
	}
	return 0;

}


unsigned sc_hypervisor_policy_resize_to_unknown_receiver(unsigned sender_sched_ctx, unsigned now)
{
	return sc_hypervisor_policy_resize(sender_sched_ctx, STARPU_NMAX_SCHED_CTXS, 0, now);
}

double sc_hypervisor_get_slowest_ctx_exec_time(void)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

/* 	double curr_time = starpu_timing_now(); */
	double slowest_time = 0.0;

	int s;
	struct sc_hypervisor_wrapper* sc_w;
	for(s = 0; s < nsched_ctxs; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);

//		double elapsed_time  = (curr_time - sc_w->start_time)/1000000;
		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
		double elapsed_time = (config->ispeed_ctx_sample/1000000000.0)/sc_hypervisor_get_ctx_speed(sc_w);
		if(elapsed_time > slowest_time)
			slowest_time = elapsed_time;

        }
	return slowest_time;
}

double sc_hypervisor_get_fastest_ctx_exec_time(void)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	double curr_time = starpu_timing_now();
 	double fastest_time = curr_time;

	int s;
	struct sc_hypervisor_wrapper* sc_w;
	for(s = 0; s < nsched_ctxs; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);

		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
		double elapsed_time = (config->ispeed_ctx_sample/1000000000.0)/sc_hypervisor_get_ctx_speed(sc_w);

		if(elapsed_time < fastest_time)
			fastest_time = elapsed_time;

        }

	return fastest_time;
}

void sc_hypervisor_group_workers_by_type(struct types_of_workers *tw, int *total_nw)
{
	unsigned w;
	for(w = 0; w < tw->nw; w++)
		total_nw[w] = 0;

	if(tw->ncpus != 0)
	{
		total_nw[0] = tw->ncpus;
		if(tw->ncuda != 0)
			total_nw[1] = tw->ncuda;
	}
	else
	{
		if(tw->ncuda != 0)
			total_nw[0] =tw->ncuda;
	}

}

enum starpu_worker_archtype sc_hypervisor_get_arch_for_index(unsigned w, struct types_of_workers *tw)
{
	if(w == 0)
	{
		if(tw->ncpus != 0)
			return STARPU_CPU_WORKER;
		else
			return STARPU_CUDA_WORKER;
	}
	else
		if(tw->ncuda != 0)
			return STARPU_CUDA_WORKER;

	return STARPU_CPU_WORKER;
}

unsigned sc_hypervisor_get_index_for_arch(enum starpu_worker_archtype arch, struct types_of_workers *tw)
{

	if(arch == STARPU_CPU_WORKER)
	{
		if(tw->ncpus != 0)
			return 0;
	}
	else
	{
		if(arch == STARPU_CUDA_WORKER)
		{
			if(tw->ncpus != 0)
				return 1;
			else
				return 0;
		}
	}
	return 0;
}

void sc_hypervisor_get_tasks_times(int nw, int nt, double times[nw][nt], int *workers, unsigned size_ctxs, struct sc_hypervisor_policy_task_pool *task_pools)
{
        struct sc_hypervisor_policy_task_pool *tp;
        int w, t;
	for(w = 0; w < nw; w++)
		for(t = 0; t < nt; t++)
			times[w][t] = NAN;
        for (w = 0; w < nw; w++)
        {
                for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
                {
			int worker = workers == NULL ? w : workers[w];
                        struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(worker, STARPU_NMAX_SCHED_CTXS);
                        double length = starpu_perfmodel_history_based_expected_perf(tp->cl->model, arch, tp->footprint);

                        if (isnan(length))
                                times[w][t] = NAN;
			else
			{
                                times[w][t] = (length / 1000.);
				double transfer_time = 0.0;
				unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, tp->sched_ctx_id);
				enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
				if(!worker_in_ctx && !size_ctxs)
				{
					if(arch == STARPU_CUDA_WORKER)
					{
						double transfer_speed = starpu_transfer_bandwidth(STARPU_MAIN_RAM, starpu_worker_get_memory_node(worker));
						if(transfer_speed > 0.0)
							transfer_time +=  (tp->data_size / transfer_speed) / 1000. ;

						double latency = starpu_transfer_latency(STARPU_MAIN_RAM, starpu_worker_get_memory_node(worker));
						transfer_time += latency/1000.;
//						transfer_time *=4;
					}
					else if(arch == STARPU_CPU_WORKER)
					{
						if(!starpu_sched_ctx_contains_type_of_worker(arch, tp->sched_ctx_id))
						{
							double transfer_speed = starpu_transfer_bandwidth(starpu_worker_get_memory_node(worker), STARPU_MAIN_RAM);
							if(transfer_speed > 0.0)
								transfer_time += (tp->data_size / transfer_speed) / 1000. ;

							double latency = starpu_transfer_latency(starpu_worker_get_memory_node(worker), STARPU_MAIN_RAM);
							transfer_time += latency / 1000.;
						}
					}
				}

//				printf("%d/%d %s x %d time = %lf transfer_time = %lf\n", w, tp->sched_ctx_id, tp->cl->model->symbol, tp->n, times[w][t], transfer_time);
				times[w][t] += transfer_time;
			}
//			printf("sc%d w%d task %s nt %d times %lf s\n", tp->sched_ctx_id, w, tp->cl->model->symbol, tp->n, times[w][t]);
                }
        }
}

unsigned sc_hypervisor_check_idle(unsigned sched_ctx, int worker)
{
	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	struct sc_hypervisor_policy_config *config = sc_w->config;
	if(config != NULL)
	{
		if(sc_w->idle_time[worker] > config->max_idle[worker])
		{
//			printf("w%d/ctx%d: current idle %lf  max_idle %lf\n", worker, sched_ctx, sc_w->idle_time[worker], config->max_idle[worker]);
			return 1;
		}
	}

	return 0;
}

/* check if there is a big speed gap between the contexts */
unsigned sc_hypervisor_check_speed_gap_btw_ctxs(unsigned *sched_ctxs_in, int ns_in, int *workers_in, int nworkers_in)
{
	unsigned *sched_ctxs = sched_ctxs_in == NULL ? sc_hypervisor_get_sched_ctxs() : sched_ctxs_in;
	int ns = ns_in == -1 ? sc_hypervisor_get_nsched_ctxs() : ns_in;
	int *workers = workers_in;
	int nworkers = nworkers_in == -1 ? starpu_worker_get_count() : nworkers_in;
	int i = 0, j = 0;
	struct sc_hypervisor_wrapper* sc_w;
	struct sc_hypervisor_wrapper* other_sc_w;


	double optimal_v[ns];
	unsigned has_opt_v = 1;
	for(i = 0; i < ns; i++)
	{
		optimal_v[i] = _get_optimal_v(sched_ctxs[i]);
		if(optimal_v[i] == 0.0)
		{
			has_opt_v = 0;
			break;
		}
	}

/*if an optimal speed has not been computed yet do it now */
	if(!has_opt_v)
	{
		struct types_of_workers *tw = sc_hypervisor_get_types_of_workers(workers, nworkers);
		int nw = tw->nw;
		double nworkers_per_ctx[ns][nw];
		int total_nw[nw];
		sc_hypervisor_group_workers_by_type(tw, total_nw);

//		double vmax = sc_hypervisor_lp_get_nworkers_per_ctx(ns, nw, nworkers_per_ctx, total_nw, tw, sched_ctxs);


//		if(vmax != 0.0)
		{
			for(i = 0; i < ns; i++)
			{
				sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
				double v[nw];
				optimal_v[i] = 0.0;
				int w;
				for(w = 0; w < nw; w++)
				{
					v[w] = sc_hypervisor_get_speed(sc_w, sc_hypervisor_get_arch_for_index(w, tw));
					optimal_v[i] += nworkers_per_ctx[i][w] == -1.0 ? 0.0 : nworkers_per_ctx[i][w]*v[w];
				}
				_set_optimal_v(sched_ctxs[i], optimal_v[i]);
			}
			has_opt_v = 1;
		}
		free(tw);
	}

/* if we have an optimal speed for each type of worker compare the monitored one with the
   theoretical one */
	if(has_opt_v)
	{
		for(i = 0; i < ns; i++)
		{
			sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);

			double ctx_v = sc_hypervisor_get_ctx_speed(sc_w);
			if(ctx_v == -1.0)
				return 0;
		}

		for(i = 0; i < ns; i++)
		{
			sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);

			double ctx_v = sc_hypervisor_get_ctx_speed(sc_w);
			ctx_v = ctx_v < 0.01 ? 0.0 : ctx_v;
			double max_vel = _get_max_speed_gap();
			if(ctx_v != -1.0 && ((ctx_v < (1-max_vel)*optimal_v[i]) || ctx_v > (1+max_vel)*optimal_v[i]))
			{
				return 1;
			}
		}
	}
	else /* if we have not been able to compute a theoretical speed consider the env variable
		SC_MAX_SPEED_GAP and compare the speed of the contexts, whenever the difference
		btw them is greater than the max value the function returns true */
	{
		for(i = 0; i < ns; i++)
		{
			sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
			double ctx_v = sc_hypervisor_get_ctx_speed(sc_w);
			if(ctx_v != -1.0)
			{
				for(j = 0; j < ns; j++)
				{
					if(sched_ctxs[i] != sched_ctxs[j])
					{
						unsigned nworkers = starpu_sched_ctx_get_nworkers(sched_ctxs[j]);
						if(nworkers == 0)
							return 1;

						other_sc_w = sc_hypervisor_get_wrapper(sched_ctxs[j]);
						double other_ctx_v = sc_hypervisor_get_ctx_speed(other_sc_w);
						if(other_ctx_v != -1.0)
						{
							double gap = ctx_v < other_ctx_v ? other_ctx_v / ctx_v : ctx_v / other_ctx_v;
							double max_vel = _get_max_speed_gap();
							if(gap > max_vel)
								return 1;
						}
					}
				}
			}

		}
	}
	return 0;
}

unsigned sc_hypervisor_check_speed_gap_btw_ctxs_on_level(int level, int *workers_in, int nworkers_in, unsigned father_sched_ctx_id, unsigned **sched_ctxs, int *nsched_ctxs)
{
	sc_hypervisor_get_ctxs_on_level(sched_ctxs, nsched_ctxs, level, father_sched_ctx_id);

	if(*nsched_ctxs  > 0)
		return sc_hypervisor_check_speed_gap_btw_ctxs(*sched_ctxs, *nsched_ctxs, workers_in, nworkers_in);
	return 0;
}

unsigned sc_hypervisor_criteria_fulfilled(unsigned sched_ctx, int worker)
{
	unsigned criteria = sc_hypervisor_get_resize_criteria();
	if(criteria != SC_NOTHING)
	{
		if(criteria == SC_IDLE)
			return sc_hypervisor_check_idle(sched_ctx, worker);
		else
			return sc_hypervisor_check_speed_gap_btw_ctxs(NULL, -1, NULL, -1);
	}
	else
		return 0;
}
