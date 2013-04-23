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

static struct sc_hypervisor_policy_task_pool *task_pools = NULL;

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

struct teft_lp_data 
{
	int nt;
	double **tasks;
	int *in_sched_ctxs;
	int *workers;
	struct sc_hypervisor_policy_task_pool *tmp_task_pools;
	unsigned size_ctxs;
};

static double _compute_workers_distrib(int ns, int nw, double final_w_in_s[ns][nw], 
			   unsigned is_integer, double tmax, void *specific_data)
{
	struct teft_lp_data *sd = (struct teft_lp_data *)specific_data;

	int nt = sd->nt;
	double **final_tasks = sd->tasks;
	int *in_sched_ctxs = sd->in_sched_ctxs;
	int *workers = sd->workers;
	struct sc_hypervisor_policy_task_pool *tmp_task_pools = sd->tmp_task_pools;
	unsigned size_ctxs = sd->size_ctxs;
		
	if(tmp_task_pools == NULL)
		return 0.0;

	double w_in_s[ns][nw];
	double tasks[nw][nt];
	double times[nw][nt];
	
	sc_hypervisor_get_tasks_times(nw, nt, times, workers, size_ctxs, task_pools);

	double res = 0.0;
#ifdef STARPU_HAVE_GLPK_H
	res = sc_hypervisor_lp_simulate_distrib_tasks(ns, nw, nt, w_in_s, tasks, times, is_integer, tmax, in_sched_ctxs, tmp_task_pools);
#endif //STARPU_HAVE_GLPK_H
	if(res != 0.0)
	{
		int s, w, t;
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
				final_w_in_s[s][w] = w_in_s[s][w];

		for(w = 0; w < nw; w++)
			for(t = 0; t < nt; t++)
				final_tasks[w][t] = tasks[w][t];
	}
	return res;
}
	
static void _size_ctxs(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	int ns = sched_ctxs == NULL ? sc_hypervisor_get_nsched_ctxs() : nsched_ctxs;
	int nw = workers == NULL ? (int)starpu_worker_get_count() : nworkers; /* Number of different workers */
	int nt = 0; /* Number of different kinds of tasks */
	starpu_pthread_mutex_lock(&mutex);
	struct sc_hypervisor_policy_task_pool * tp;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;

	double w_in_s[ns][nw];
//	double tasks[nw][nt];
	double **tasks=(double**)malloc(nw*sizeof(double*));
	int i;
	for(i = 0; i < nw; i++)
		tasks[i] = (double*)malloc(nt*sizeof(double));


	struct teft_lp_data specific_data;
	specific_data.nt = nt;
	specific_data.tasks = tasks;
	specific_data.in_sched_ctxs = sched_ctxs;
	specific_data.workers = workers;
	specific_data.tmp_task_pools = task_pools;
	specific_data.size_ctxs = 1;

	/* smallest possible tmax, difficult to obtain as we
	   compute the nr of flops and not the tasks */
	double possible_tmax = sc_hypervisor_lp_get_tmax(nw, workers);
	double smallest_tmax = possible_tmax / 3;
	double tmax = possible_tmax * ns;
	double tmin = smallest_tmax;

	unsigned found_sol = sc_hypervisor_lp_execute_dichotomy(ns, nw, w_in_s, 1, (void*)&specific_data, 
								tmin, tmax, smallest_tmax, _compute_workers_distrib);

	starpu_pthread_mutex_unlock(&mutex);
	/* if we did find at least one solution redistribute the resources */
	if(found_sol)
		sc_hypervisor_lp_place_resources_in_ctx(ns, nw, w_in_s, sched_ctxs, workers, 1);
	
	for(i = 0; i < nw; i++)
		free(tasks[i]);
	free(tasks);

}

static void size_if_required()
{
	int nsched_ctxs, nworkers;
	int *sched_ctxs, *workers;
	unsigned has_req = sc_hypervisor_get_size_req(&sched_ctxs, &nsched_ctxs, &workers, &nworkers);

	if(has_req)
	{
		struct sc_hypervisor_wrapper* sc_w = NULL;
		unsigned ready_to_size = 1;
		int s;
		starpu_pthread_mutex_lock(&act_hypervisor_mutex);
		for(s = 0; s < nsched_ctxs; s++)
		{
			sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);
			if(sc_w->submitted_flops < sc_w->total_flops)
				ready_to_size = 0;
		}

		if(ready_to_size)
			_size_ctxs(sched_ctxs, nsched_ctxs, workers, nworkers);
		starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
	}
}

static void teft_lp_handle_submitted_job(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint)
{
	/* count the tasks of the same type */
	starpu_pthread_mutex_lock(&mutex);
	sc_hypervisor_policy_add_task_to_pool(cl, sched_ctx, footprint, &task_pools);
	starpu_pthread_mutex_unlock(&mutex);

	size_if_required();
}

static void teft_lp_handle_poped_task(unsigned sched_ctx, int worker, struct starpu_task *task, uint32_t footprint)
{
	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx);

	int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(sc_w->submitted_flops < sc_w->total_flops)
		{
			starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
			return;
		}

		if(sc_hypervisor_has_velocity_gap_btw_ctxs())
		{
			int ns = sc_hypervisor_get_nsched_ctxs();
			int nw = starpu_worker_get_count(); /* Number of different workers */
			int nt = 0; /* Number of different kinds of tasks */

//			starpu_pthread_mutex_lock(&mutex);

			/* we don't take the mutex bc a correct value of the number of tasks is
			   not required but we do a copy in order to be sure
			   that the linear progr won't segfault if the list of 
			   submitted task will change during the exec */

			struct sc_hypervisor_policy_task_pool *tp = NULL;
			struct sc_hypervisor_policy_task_pool *tmp_task_pools = sc_hypervisor_policy_clone_task_pool(task_pools);

			for (tp = task_pools; tp; tp = tp->next)
				nt++;


			double w_in_s[ns][nw];
//			double tasks_per_worker[nw][nt];
			double **tasks_per_worker=(double**)malloc(nw*sizeof(double*));
			int i;
			for(i = 0; i < nw; i++)
				tasks_per_worker[i] = (double*)malloc(nt*sizeof(double));

			struct teft_lp_data specific_data;
			specific_data.nt = nt;
			specific_data.tasks = tasks_per_worker;
			specific_data.in_sched_ctxs = NULL;
			specific_data.workers = NULL;
			specific_data.tmp_task_pools = tmp_task_pools;
			specific_data.size_ctxs = 0;

			/* smallest possible tmax, difficult to obtain as we
			   compute the nr of flops and not the tasks */
			double possible_tmax = sc_hypervisor_lp_get_tmax(nw, NULL);
			double smallest_tmax = possible_tmax / 3;
			double tmax = possible_tmax * ns;
			double tmin = smallest_tmax;
			unsigned found_sol = sc_hypervisor_lp_execute_dichotomy(ns, nw, w_in_s, 1, (void*)&specific_data, 
								tmin, tmax, smallest_tmax, _compute_workers_distrib);
//			starpu_pthread_mutex_unlock(&mutex);

			/* if we did find at least one solution redistribute the resources */
			if(found_sol)
				sc_hypervisor_lp_place_resources_in_ctx(ns, nw, w_in_s, NULL, NULL, 0);

			struct sc_hypervisor_policy_task_pool *next = NULL;
			struct sc_hypervisor_policy_task_pool *tmp_tp = tmp_task_pools;
			while(tmp_task_pools)
			{
				next = tmp_tp->next;
				free(tmp_tp);
				tmp_tp = next;
				tmp_task_pools = next;
			}
			for(i = 0; i < nw; i++)
				free(tasks_per_worker[i]);
			free(tasks_per_worker);
		}
		starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
	}
	/* too expensive to take this mutex and correct value of the number of tasks is not compulsory */
//	starpu_pthread_mutex_lock(&mutex);
	sc_hypervisor_policy_remove_task_from_pool(task, footprint, &task_pools);
//	starpu_pthread_mutex_unlock(&mutex);

}


static void teft_lp_size_ctxs(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	sc_hypervisor_save_size_req(sched_ctxs, nsched_ctxs, workers, nworkers);
}

struct sc_hypervisor_policy teft_lp_policy = {
	.size_ctxs = teft_lp_size_ctxs,
	.handle_poped_task = teft_lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = teft_lp_handle_submitted_job,
	.end_ctx = NULL,
	.custom = 0,
	.name = "teft_lp"
};
