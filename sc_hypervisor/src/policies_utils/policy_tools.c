/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  INRIA
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

#include <math.h>

static int _compute_priority(unsigned sched_ctx)
{
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctx);

	int total_priority = 0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
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
	int *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();


	struct sc_hypervisor_policy_config *config = NULL;

	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && sched_ctxs[i] != req_sched_ctx)
		{
			unsigned nworkers = starpu_sched_ctx_get_nworkers(sched_ctxs[i]);
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
	if(workers->init_iterator)
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
unsigned sc_hypervisor_get_movable_nworkers(struct sc_hypervisor_policy_config *config, unsigned sched_ctx, enum starpu_worker_archtype arch)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

	unsigned potential_workers = 0;
	int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
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
	unsigned nworkers = starpu_sched_ctx_get_nworkers(req_sched_ctx);
	unsigned nworkers_to_move = 0;

	unsigned potential_moving_workers = sc_hypervisor_get_movable_nworkers(config, req_sched_ctx, STARPU_ANY_WORKER);
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
		starpu_pthread_mutex_lock(&act_hypervisor_mutex);
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
				poor_sched_ctx = sc_hypervisor_find_lowest_prio_sched_ctx(sender_sched_ctx, nworkers_to_move);
			}
			else
			{
				poor_sched_ctx = receiver_sched_ctx;
				struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(poor_sched_ctx);
				unsigned nworkers = starpu_sched_ctx_get_nworkers(poor_sched_ctx);
				unsigned nshared_workers = starpu_sched_ctx_get_nshared_workers(sender_sched_ctx, poor_sched_ctx);
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
		starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
		return 1;
	}
	return 0;

}


unsigned sc_hypervisor_policy_resize_to_unknown_receiver(unsigned sender_sched_ctx, unsigned now)
{
	return sc_hypervisor_policy_resize(sender_sched_ctx, STARPU_NMAX_SCHED_CTXS, 0, now);
}

static double _get_ispeed_sample_for_type_of_worker(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype req_arch)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
        int worker;

	double avg = 0.0;
	int n = 0;
	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
                workers->init_iterator(workers, &it);

        while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);
                enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
                if(arch == req_arch)
                {
			struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
			avg += config->ispeed_w_sample[worker];
			n++;
		}
        }

	return n != 0 ? avg/n : 0;
}

static double _get_ispeed_sample_for_sched_ctx(unsigned sched_ctx)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctx);
        
	int worker;
	double ispeed_sample = 0.0;
	struct starpu_sched_ctx_iterator it;

	if(workers->init_iterator)
                workers->init_iterator(workers, &it);

        while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);
	        ispeed_sample += config->ispeed_w_sample[worker];
        }

	return ispeed_sample;
}

double sc_hypervisor_get_ctx_velocity(struct sc_hypervisor_wrapper* sc_w)
{
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
        double elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);
	double sample = _get_ispeed_sample_for_sched_ctx(sc_w->sched_ctx);

/* 	double total_elapsed_flops = sc_hypervisor_get_total_elapsed_flops_per_sched_ctx(sc_w); */
/* 	double prc = config->ispeed_ctx_sample != 0.0 ? elapsed_flops : elapsed_flops/sc_w->total_flops; */
/* 	double redim_sample = config->ispeed_ctx_sample != 0.0 ? config->ispeed_ctx_sample :  */
/* 		(elapsed_flops == total_elapsed_flops ? HYPERVISOR_START_REDIM_SAMPLE : HYPERVISOR_REDIM_SAMPLE); */
//	printf("%d: prc %lf sample %lf\n", sc_w->sched_ctx, prc, redim_sample);

/* 	double curr_time2 = starpu_timing_now(); */
/* 	double elapsed_time2 = (curr_time2 - sc_w->start_time) / 1000000.0; /\* in seconds *\/ */
/* 	if(elapsed_time2 > 5.0 && elapsed_flops < sample) */
/* 		return (elapsed_flops/1000000000.0)/elapsed_time2;/\* in Gflops/s *\/ */

	if(elapsed_flops >= sample)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */
                return (elapsed_flops/1000000000.0)/elapsed_time;/* in Gflops/s */
        }
	return -1.0;
}

double sc_hypervisor_get_slowest_ctx_exec_time(void)
{
	int *sched_ctxs = sc_hypervisor_get_sched_ctxs();
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
		double elapsed_time = (config->ispeed_ctx_sample/1000000000.0)/sc_hypervisor_get_ctx_velocity(sc_w);
		if(elapsed_time > slowest_time)
			slowest_time = elapsed_time;

        }
	return slowest_time;
}

double sc_hypervisor_get_fastest_ctx_exec_time(void)
{
	int *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	double curr_time = starpu_timing_now();
 	double fastest_time = curr_time;

	int s;
	struct sc_hypervisor_wrapper* sc_w;		
	for(s = 0; s < nsched_ctxs; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);

		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
		double elapsed_time = (config->ispeed_ctx_sample/1000000000.0)/sc_hypervisor_get_ctx_velocity(sc_w);
		
		if(elapsed_time < fastest_time)
			fastest_time = elapsed_time;

        }

	return fastest_time;
}


double sc_hypervisor_get_velocity_per_worker(struct sc_hypervisor_wrapper *sc_w, unsigned worker)
{
	if(!starpu_sched_ctx_contains_worker(worker, sc_w->sched_ctx))
		return -1.0;

        double elapsed_flops = sc_w->elapsed_flops[worker] / 1000000000.0; /*in gflops */
	size_t elapsed_data_used = sc_w->elapsed_data[worker];
	int elapsed_tasks = sc_w->elapsed_tasks[worker];
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
	double sample = config->ispeed_w_sample[worker] / 1000000000.0; /*in gflops */

	double ctx_elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);
	double ctx_sample = config->ispeed_ctx_sample;
	if(ctx_elapsed_flops > ctx_sample && elapsed_flops == 0.0)
		return 0.00000000000001;

/*         if( elapsed_flops >= sample) */
/*         { */
/*                 double curr_time = starpu_timing_now(); */
/*                 double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /\* in seconds *\/ */
/* 		sc_w->ref_velocity[worker] = (elapsed_flops/elapsed_time); /\* in Gflops/s *\/ */
/*                 return sc_w->ref_velocity[worker]; */
/*         } */

/*         return -1.0; */

        if( elapsed_flops != 0.0)
        {
                double curr_time = starpu_timing_now();
		size_t elapsed_data_used = sc_w->elapsed_data[worker];
                double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */
 		enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
		if(arch == STARPU_CUDA_WORKER)
		{
/* 			unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, sc_w->sched_ctx); */
/* 			if(!worker_in_ctx) */
/* 			{ */

/* 				double transfer_velocity = starpu_transfer_bandwidth(0, starpu_worker_get_memory_node(worker)); */
/* 				elapsed_time +=  (elapsed_data_used / transfer_velocity) / 1000000 ; */
/* 			} */
			double latency = starpu_transfer_latency(0, starpu_worker_get_memory_node(worker));
//			printf("%d/%d: latency %lf elapsed_time before %lf ntasks %d\n", worker, sc_w->sched_ctx, latency, elapsed_time, elapsed_tasks);
			elapsed_time += (elapsed_tasks * latency)/1000000;
//			printf("elapsed time after %lf \n", elapsed_time);
		}
			
                double vel  = (elapsed_flops/elapsed_time);/* in Gflops/s */
		sc_w->ref_velocity[worker] = sc_w->ref_velocity[worker] > 1.0 ? (sc_w->ref_velocity[worker] + vel) / 2 : vel; 
                return vel;
        }

        return 0.00000000000001;


}

static double _get_best_elapsed_flops(struct sc_hypervisor_wrapper* sc_w, int *npus, enum starpu_worker_archtype req_arch)
{
	double ret_val = 0.0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
        int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
                workers->init_iterator(workers, &it);

        while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);
                enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
                if(arch == req_arch)
                {
			if(sc_w->elapsed_flops[worker] > ret_val)
				ret_val = sc_w->elapsed_flops[worker];
			(*npus)++;
                }
        }

	return ret_val;
}

/* compute an average value of the cpu/cuda velocity */
double sc_hypervisor_get_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
        int npus = 0;
        double elapsed_flops = _get_best_elapsed_flops(sc_w, &npus, arch) / 1000000000.0 ; /* in gflops */
	if(npus == 0)
		return -1.0; 

        if( elapsed_flops != 0.0)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */
		double velocity = (elapsed_flops/elapsed_time); /* in Gflops/s */
                return velocity;
        }

        return -1.0;
}


/* check if there is a big velocity gap between the contexts */
int sc_hypervisor_has_velocity_gap_btw_ctxs()
{
	int *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();
	int i = 0, j = 0;
	struct sc_hypervisor_wrapper* sc_w;
	struct sc_hypervisor_wrapper* other_sc_w;

	for(i = 0; i < nsched_ctxs; i++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[i]);
		double ctx_v = sc_hypervisor_get_ctx_velocity(sc_w);
		if(ctx_v != -1.0)
		{
			for(j = 0; j < nsched_ctxs; j++)
			{
				if(sched_ctxs[i] != sched_ctxs[j])
				{
					unsigned nworkers = starpu_sched_ctx_get_nworkers(sched_ctxs[j]);
					if(nworkers == 0) 
						return 1;

					other_sc_w = sc_hypervisor_get_wrapper(sched_ctxs[j]);
					double other_ctx_v = sc_hypervisor_get_ctx_velocity(other_sc_w);
					if(other_ctx_v != -1.0)
					{
						double gap = ctx_v < other_ctx_v ? other_ctx_v / ctx_v : ctx_v / other_ctx_v ;
//						if(gap > 1.5)
						if(gap > 3.0)
							return 1;
					}
				}
			}
		}

	}
	return 0;
}


void sc_hypervisor_group_workers_by_type(int *workers, int nworkers, int ntypes_of_workers, int total_nw[ntypes_of_workers])
{
	int current_nworkers = workers == NULL ? starpu_worker_get_count() : nworkers;
	int w;
	for(w = 0; w < ntypes_of_workers; w++)
		total_nw[w] = 0;

	for(w = 0; w < current_nworkers; w++)
	{
 		enum starpu_worker_archtype arch = workers == NULL ? starpu_worker_get_type(w) :
			starpu_worker_get_type(workers[w]);
		if(ntypes_of_workers == 2)
		{
			if(arch == STARPU_CPU_WORKER)
				total_nw[1]++;
			else
				total_nw[0]++;
		}
		else
			total_nw[0]++;
	}
}

void sc_hypervisor_get_tasks_times(int nw, int nt, double times[nw][nt], int *workers, unsigned size_ctxs, struct sc_hypervisor_policy_task_pool *task_pools)
{
        struct sc_hypervisor_policy_task_pool *tp;
        int w, t;
        for (w = 0; w < nw; w++)
        {
                for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
                {
			int worker = workers == NULL ? w : workers[w];
                        enum starpu_perfmodel_archtype arch = starpu_worker_get_perf_archtype(worker);
                        double length = starpu_permodel_history_based_expected_perf(tp->cl->model, arch, tp->footprint);

                        if (isnan(length))
                                times[w][t] = NAN;
			else
			{
                                times[w][t] = length / 1000.;

				double transfer_time = 0.0;
				enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
				if(arch == STARPU_CUDA_WORKER)
				{
					unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, tp->sched_ctx_id);
					if(!worker_in_ctx && !size_ctxs)
					{
						double transfer_velocity = starpu_transfer_bandwidth(0, starpu_worker_get_memory_node(worker));
						transfer_time +=  (tp->footprint / transfer_velocity) / 1000. ;
						
						
					}
					double latency = starpu_transfer_latency(0, starpu_worker_get_memory_node(worker));
					transfer_time += latency/1000.;
				}
//				printf("%d/%d %s x %d time = %lf transfer_time = %lf\n", w, tp->sched_ctx_id, tp->cl->model->symbol, tp->n, times[w][t], transfer_time);
				times[w][t] += transfer_time;
			}
                }
        }
}

