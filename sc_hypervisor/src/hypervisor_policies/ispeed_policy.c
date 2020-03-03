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

#include "sc_hypervisor_policy.h"

static unsigned _get_fastest_sched_ctx(void)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	int fastest_sched_ctx = STARPU_NMAX_SCHED_CTXS;
	double curr_speed = 0.0;
	double biggest_speed = 0.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		curr_speed = sc_hypervisor_get_ctx_speed(sc_hypervisor_get_wrapper(sched_ctxs[i]));
		if( curr_speed > biggest_speed)
		{
			fastest_sched_ctx = sched_ctxs[i];
			biggest_speed = curr_speed;
		}
	}

	return fastest_sched_ctx;
}

static unsigned _get_slowest_sched_ctx(void)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	double smallest_speed = sc_hypervisor_get_ctx_speed(sc_hypervisor_get_wrapper(sched_ctxs[0]));
	unsigned slowest_sched_ctx = smallest_speed == -1.0  ? STARPU_NMAX_SCHED_CTXS : sched_ctxs[0];
	double curr_speed = 0.0;
	int i;
	for(i = 1; i < nsched_ctxs; i++)
	{
		curr_speed = sc_hypervisor_get_ctx_speed(sc_hypervisor_get_wrapper(sched_ctxs[i]));
		if((curr_speed < smallest_speed || smallest_speed == 0.0) && curr_speed != -1.0)
		{
			smallest_speed = curr_speed;
			slowest_sched_ctx = sched_ctxs[i];
		}
	}

	return slowest_sched_ctx;
}


/* get first nworkers with the highest idle time in the context */
static int* _get_slowest_workers(unsigned sched_ctx, int *nworkers, enum starpu_worker_archtype arch)
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
						double worker_speed = sc_hypervisor_get_speed_per_worker(sc_w, worker);
						if(worker_speed != -1.0)
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
								double curr_worker_speed = sc_hypervisor_get_speed_per_worker(sc_w, curr_workers[index]);
//								printf("speed[%d] = %lf speed[%d] = %lf\n", worker, worker_speed, curr_workers[index], curr_worker_speed);
								if(worker_speed < curr_worker_speed && curr_worker_speed != -1.0)
								{
									curr_workers[index] = worker;
								}
							}
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

static void ispeed_handle_poped_task(unsigned sched_ctx, int worker, __attribute__((unused))struct starpu_task *task, __attribute__((unused))uint32_t footprint)
{
	int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(sc_hypervisor_criteria_fulfilled(sched_ctx, worker))
		{
			unsigned fastest_sched_ctx = _get_fastest_sched_ctx();
			unsigned slowest_sched_ctx = _get_slowest_sched_ctx();
			if(fastest_sched_ctx != STARPU_NMAX_SCHED_CTXS && slowest_sched_ctx != STARPU_NMAX_SCHED_CTXS && fastest_sched_ctx != slowest_sched_ctx)
			{
				int nworkers_to_move = sc_hypervisor_compute_nworkers_to_move(fastest_sched_ctx);
				if(nworkers_to_move > 0)
				{
					int *workers_to_move = _get_slowest_workers(fastest_sched_ctx, &nworkers_to_move, STARPU_ANY_WORKER);
					if(nworkers_to_move > 0)
					{
						double new_speed = 0.0;
						int i;
						for(i = 0; i < nworkers_to_move; i++)
							new_speed += sc_hypervisor_get_speed_per_worker(sc_hypervisor_get_wrapper(fastest_sched_ctx), workers_to_move[i]);
						double fastest_speed = sc_hypervisor_get_ctx_speed(sc_hypervisor_get_wrapper(fastest_sched_ctx));
						double slowest_speed = sc_hypervisor_get_ctx_speed(sc_hypervisor_get_wrapper(slowest_sched_ctx));
//						printf("fast_speed(%d) %lf slow_speed(%d) %lf new speed(%d) %lf \n", fastest_sched_ctx, fastest_speed, slowest_sched_ctx,
//						       slowest_speed, workers_to_move[0], new_speed);
						if(fastest_speed != -1.0 && slowest_speed != -1.0 && (slowest_speed + new_speed) <= (fastest_speed - new_speed))
						{
							sc_hypervisor_move_workers(fastest_sched_ctx, slowest_sched_ctx, workers_to_move, nworkers_to_move, 0);
						}
					}

					free(workers_to_move);
				}

			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
	}
}

struct sc_hypervisor_policy ispeed_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = ispeed_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.init_worker = NULL,
	.custom = 0,
	.name = "ispeed"
};
