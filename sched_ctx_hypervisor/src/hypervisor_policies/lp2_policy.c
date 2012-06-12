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

#include "policy_tools.h"
#include <math.h>

static struct bound_task_pool *task_pools, *last;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void lp2_handle_submitted_job(struct starpu_task *task, unsigned footprint)
{
	pthread_mutex_lock(&mutex);
	struct bound_task_pool *tp;
	
	if (last && last->cl == task->cl && last->footprint == footprint && last->sched_ctx_id == task->sched_ctx)
		tp = last;
	else
		for (tp = task_pools; tp; tp = tp->next)
			if (tp->cl == task->cl && tp->footprint == footprint && tp->sched_ctx_id == task->sched_ctx)
					break;
	
	if (!tp)
	{
		tp = (struct bound_task_pool *) malloc(sizeof(*tp));
		tp->cl = task->cl;
		tp->footprint = footprint;
		tp->sched_ctx_id = task->sched_ctx;
		tp->n = 0;
		tp->next = task_pools;
		task_pools = tp;
	}
	
	/* One more task of this kind */
	tp->n++;
	pthread_mutex_unlock(&mutex);
}

static void _starpu_get_tasks_times(int nw, int nt, double times[nw][nt])
{
        struct bound_task_pool *tp;
        int w, t;
        for (w = 0; w < nw; w++)
        {
                for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
                {
                        enum starpu_perf_archtype arch = starpu_worker_get_perf_archtype(w);
                        double length = starpu_history_based_job_expected_perf(tp->cl->model, arch, tp->footprint);

                        if (isnan(length))
                                times[w][t] = NAN;
                       else
                                times[w][t] = length / 1000.;
			
//			printf("t%d on worker %d ctx %d: %lf \n", t, w, tp->sched_ctx_id, times[w][t]);
                }
//		printf("\n");
        }
//	printf("\n");
}

int _get_idx_sched_ctx(int sched_ctx_id)
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
	int i;
	for(i = 0; i < nsched_ctxs; i++)
		if(sched_ctxs[i] == sched_ctx_id)
			return i;
	return -1;
}

/*                                                                                                                                                                                                                  
 * GNU Linear Programming Kit backend                                                                                                                                                                               
 */
#ifdef HAVE_GLPK_H
#include <glpk.h>
static void _glp_resolve(int ns, int nw, int nt, double tasks[nw][nt])
{
	struct bound_task_pool * tp;
	int t, w, s;
	glp_prob *lp;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MIN);
	glp_set_obj_name(lp, "total execution time");

	{
		double times[nw][nt];
		int ne =
			nw * (nt+1)	/* worker execution time */
			+ nt * nw
			+ nw * (nt+ns)
			+ 1; /* glp dumbness */
		int n = 1;
		int ia[ne], ja[ne];
		double ar[ne];

		_starpu_get_tasks_times(nw, nt, times);

		/* Variables: number of tasks i assigned to worker j, and tmax */
		glp_add_cols(lp, nw*nt+1);
#define colnum(w, t) ((t)*nw+(w)+1)
		glp_set_obj_coef(lp, nw*nt+1, 1.);

		for (w = 0; w < nw; w++)
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%dt%dn", w, t);
				glp_set_col_name(lp, colnum(w, t), name);
				glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0., 0.);
			}
		glp_set_col_bnds(lp, nw*nt+1, GLP_LO, 0., 0.);

		int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();

		/* ntasks_per_worker*t_tasks < tmax */
		glp_add_rows(lp, nw*ns);
		for(s = 0; s < ns; s++)
		{
			for (w = 0; w < nw; w++)
			{
				char name[32], title[64];
				starpu_worker_get_name(w, name, sizeof(name));
				snprintf(title, sizeof(title), "worker %x ctx %x limit", w, s);
				glp_set_row_name(lp, w+(s*nw)+1, title);
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if(tp->sched_ctx_id == sched_ctxs[s])
					{
						ia[n] = w+(s*nw)+1;
						ja[n] = colnum(w, t);
						ar[n] = times[w][t];
						
						n++;
					}
				}

				/* tmax */
				ia[n] = w+(s*nw)+1;
				ja[n] = nw*nt+1;
				ar[n] = -1;
				n++;

				glp_set_row_bnds(lp, w+(s*nw)+1, GLP_UP, 0.0, 0.0);
			}
 		}

		int curr_row_idx = nw*ns;
		/* Total worker execution time */
		glp_add_rows(lp, nw*ns);
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			int someone = 0;
			for (w = 0; w < nw; w++)
				if (!isnan(times[w][t]))
					someone = 1;
			if (!someone)
			{
				/* This task does not have any performance model at all, abort */
				glp_delete_prob(lp);
				return NULL;
			}
		}
		for (w = 0; w < nw; w++)
		{
			
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "worker %s", name);
			glp_set_row_name(lp, curr_row_idx+w+1, title);
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			{
				ia[n] = curr_row_idx+w+1;
				ja[n] = colnum(w, t);
				if (isnan(times[w][t]))
					ar[n] = 1000000000.;
				else
					ar[n] = times[w][t];
				if(starpu_worker_belongs_to_sched_ctx(w, tp->sched_ctx_id))
					ar[n] = 100000;
				
				n++;
			}
			/* tmax */
			ia[n] = curr_row_idx+w+1;
			ja[n] = nw*nt+1;
			ar[n] = -1;
			n++;
			glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_UP, 0, 0);
		}

		/* Total task completion */
		glp_add_rows(lp, nt);
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "task %s key %x", tp->cl->name, (unsigned) tp->footprint);
			glp_set_row_name(lp, curr_row_idx+nw+t+1, title);
			for (w = 0; w < nw; w++)
			{
				ia[n] = curr_row_idx+nw+t+1;
				ja[n] = colnum(w, t);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, curr_row_idx+nw+t+1, GLP_FX, tp->n, tp->n);
		}


//		printf("n = %d nw*ns  = %d ne = %d\n", n, nw*ns, ne);
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
		return NULL;
	}

	double tmax = glp_get_obj_val(lp);

//        printf("Theoretical minimum execution time: %f ms\n", tmax);
	for (w = 0; w < nw; w++)
	{
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			tasks[w][t] = glp_get_col_prim(lp, colnum(w, t));
//			printf("t%d worker %d ctx %d res %lf \n", t, w, tasks[w][t]);
		}
	}

	glp_delete_prob(lp);
}

int _get_worker_having_tasks_of_this_ctx(int worker, int nw, int nt, double tasks[nw][nt], int sched_ctx)
{
	int t, w;
	struct bound_task_pool * tp;
	for(w = 0; w < nw; w++)
	{
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			if(w != worker && tasks[w][t] >= 1.0 && tp->sched_ctx_id == sched_ctx)
				return w;
	}
	return -1;
}
int _get_worker_full_of_tasks_of_this_ctx(int worker, int nw, int nt, double tasks[nw][nt], int sched_ctx)
{
	int t, w;
	struct bound_task_pool * tp;
	for(w = 0; w < nw; w++)
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			if(w != worker && tasks[w][t] > 0.3 * tp->n && tp->sched_ctx_id == sched_ctx)
				return w;
	return -1;
}

void _get_tasks_from_busiest_worker(int nw, int nt, double tasks[nw][nt], int worker)
{
	int w, t;
	double tasks_per_worker[nw];
	double max_tasks = 0.0;
	int busiest_worker = -1;
	printf("got inside \n");
	for(w = 0; w < nw; w++)
	{
		if(w != worker)
		{
			tasks_per_worker[w] = 0.0;
			for(t = 0; t < nt; t++)
			{
				tasks_per_worker[w] += tasks[w][t];
			}
			if(max_tasks < tasks_per_worker[w])
			{
				max_tasks = tasks_per_worker[w];
				busiest_worker = w;
			}
		}
	}
	for(t = 0; t < nt; t++)
	{
		if(tasks_per_worker[busiest_worker] > (max_tasks / 2))
		{
			tasks[worker][t] = tasks[busiest_worker][t];
			tasks_per_worker[busiest_worker] -= tasks[busiest_worker][t];
			tasks[busiest_worker][t] = 0.0;
		}
	}
}
void _recompute_resource_distrib(int nw, int nt, double tasks[nw][nt])
{
	int w, s, t;
	struct bound_task_pool * tp;
	for(w = 0; w < nw; w++)
	{
		int no_ctxs = 0;
		int last_ctx = -1;
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			if(tasks[w][t] >= 1.0)
			{
				if(last_ctx != -1 && tp->sched_ctx_id != last_ctx)
				{
					enum starpu_archtype arch = starpu_worker_get_type(w);
					int w2 = -1;
					if(arch == STARPU_CPU_WORKER)
						w2 = _get_worker_having_tasks_of_this_ctx(w, nw, nt, tasks, tp->sched_ctx_id);
					else if(arch == STARPU_CUDA_WORKER && tasks[w][t] < 0.3*tp->n)
						w2 = _get_worker_full_of_tasks_of_this_ctx(w, nw, nt, tasks, tp->sched_ctx_id);
					
					printf("w=%d t=%d tasks=%lf w2=%d\n", w, t, tasks[w][t], w2);
					if(w2 != -1)
					{
						tasks[w2][t] += tasks[w][t];
						tasks[w][t] = 0.0;
					}
				}
				else
					last_ctx = tp->sched_ctx_id;
			}
		}
	}

	
	for(w = 0; w < nw; w++)
	{
		unsigned empty = 1;
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			if(tasks[w][t] >= 1.0)
			{
				printf("%d: tasks %lf\n", w, tasks[w][t]);
				empty = 0;
				break;
			}
		}
		
		if(empty)
		{
			printf("worker having no task %d\n", w);
			_get_tasks_from_busiest_worker(nw, nt, tasks, w);
		}
	}
}

void _redistribute_resources_in_ctxs2(int ns, int nw, int nt, double tasks[nw][nt])
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
        struct bound_task_pool * tp;
	int s, s2, w, t;

	for(s = 0; s < ns; s++)
	{
		int workers_to_add[nw], workers_to_remove[nw];
		for(w = 0; w < nw; w++)
		{
			workers_to_add[w] = -1;
			workers_to_remove[w] = -1;
		}

		int nadd = 0, nremove = 0;

		for(w = 0; w < nw; w++)
		{
			int found = 0;
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			{
				if(tp->sched_ctx_id == sched_ctxs[s])
				{
					if(tasks[w][t] >= 1.0)
					{
						workers_to_add[nadd++] = w;
						found = 1;
						break;
					}
				}
			}
			if(!found)
				workers_to_remove[nremove++] = w;
		}

		
		unsigned nworkers_ctx = get_nworkers_ctx(sched_ctxs[s], STARPU_ALL);
	
		if(nworkers_ctx > nremove)
			sched_ctx_hypervisor_remove_workers_from_sched_ctx(workers_to_remove, nremove, sched_ctxs[s]);
	
		if(nworkers_ctx != STARPU_NMAXWORKERS)
		{
			sched_ctx_hypervisor_add_workers_to_sched_ctx(workers_to_add, nadd, sched_ctxs[s]);
			struct policy_config *new_config = sched_ctx_hypervisor_get_config(sched_ctxs[s]);
			int i;
			for(i = 0; i < nadd; i++)
				new_config->max_idle[workers_to_add[i]] = new_config->max_idle[workers_to_add[i]] != MAX_IDLE_TIME ? new_config->max_idle[workers_to_add[i]] :  new_config->new_workers_max_idle;
		}
	}
		
}
int redistrib = 0;
int done = 0;
void lp2_handle_poped_task(unsigned sched_ctx, int worker)
{
	struct sched_ctx_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	
	int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(sc_w->submitted_flops >= sc_w->total_flops && !done)
		{
			redistrib = 1;
			done = 1;
		}

		if(_velocity_gap_btw_ctxs() && redistrib)
		{
			redistrib = 0;
			int ns = sched_ctx_hypervisor_get_nsched_ctxs();
			int nw = starpu_worker_get_count(); /* Number of different workers */
			int nt = 0; /* Number of different kinds of tasks */
			struct bound_task_pool * tp;
			for (tp = task_pools; tp; tp = tp->next)
				nt++;
			
			double tasks[nw][nt];
 			int w,t;
			for(w = 0; w < nw; w++)
				for(t = 0; t < nt; t++)
					tasks[w][t] = 0.0;

			printf("###################################start to resolve \n");
			_glp_resolve(ns, nw, nt, tasks);
			for(w = 0; w < nw; w++)
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if(tasks[w][t] > 0.0)
						printf("ctx %d/worker %d/task type %d: res = %lf \n", tp->sched_ctx_id, w, t, tasks[w][t]);
				}
			printf("***************************\n");			

			_recompute_resource_distrib(nw, nt, tasks);

			for(w = 0; w < nw; w++)
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if(tasks[w][t] > 0.0)
						printf("ctx %d/worker %d/task type %d: res = %lf \n", tp->sched_ctx_id, w, t, tasks[w][t]);
				}
			

			_redistribute_resources_in_ctxs2(ns, nw, nt, tasks);
		}
		pthread_mutex_unlock(&act_hypervisor_mutex);
	}		
}

struct hypervisor_policy lp2_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = lp2_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = lp2_handle_submitted_job,
	.custom = 0,
	.name = "lp2"
};
	
#endif /* HAVE_GLPK_H */

