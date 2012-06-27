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

#include "lp_tools.h"
#include <math.h>

static struct bound_task_pool *task_pools = NULL;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void _size_ctxs(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	int ns = sched_ctxs == NULL ? sched_ctx_hypervisor_get_nsched_ctxs() : nsched_ctxs;
	int nw = workers == NULL ? starpu_worker_get_count() : nworkers; /* Number of different workers */
	int nt = 0; /* Number of different kinds of tasks */
	struct bound_task_pool * tp;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;
	
	double w_in_s[ns][nw];
	
	unsigned found_sol = _compute_task_distribution_over_ctxs(ns, nw, nt, w_in_s, sched_ctxs, workers);
	/* if we did find at least one solution redistribute the resources */
	if(found_sol)
		_redistribute_resources_in_ctxs(ns, nw, nt, w_in_s, 1, sched_ctxs, workers);
}

static void size_if_required()
{
	int nsched_ctxs, nworkers;
	int *sched_ctxs, *workers;
	unsigned has_req = sched_ctx_hypervisor_get_size_req(&sched_ctxs, &nsched_ctxs, &workers, &nworkers);	

	if(has_req)
	{
		struct sched_ctx_wrapper* sc_w = NULL;
		unsigned ready_to_size = 1;
		int s;
		pthread_mutex_lock(&act_hypervisor_mutex);
		for(s = 0; s < nsched_ctxs; s++)
		{
			sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[s]);
			if(sc_w->submitted_flops < sc_w->total_flops)
				ready_to_size = 0;
		}

		if(ready_to_size)
			_size_ctxs(sched_ctxs, nsched_ctxs, workers, nworkers);
		pthread_mutex_unlock(&act_hypervisor_mutex);
	}
}

static void lp2_handle_submitted_job(struct starpu_task *task, uint32_t footprint)
{
	/* count the tasks of the same type */
	pthread_mutex_lock(&mutex);
	struct bound_task_pool *tp = NULL;

	for (tp = task_pools; tp; tp = tp->next)
	{
		if (tp->cl == task->cl && tp->footprint == footprint && tp->sched_ctx_id == task->sched_ctx)
			break;
	}

	if (!tp)
	{
		tp = (struct bound_task_pool *) malloc(sizeof(struct bound_task_pool));
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

	size_if_required();
}

static void _starpu_get_tasks_times(int nw, int nt, double times[nw][nt], int *workers)
{
        struct bound_task_pool *tp;
        int w, t;
        for (w = 0; w < nw; w++)
        {
                for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
                {
                        enum starpu_perf_archtype arch = workers == NULL ? starpu_worker_get_perf_archtype(w) :
				starpu_worker_get_perf_archtype(workers[w]);
                        double length = starpu_history_based_job_expected_perf(tp->cl->model, arch, tp->footprint);

                        if (isnan(length))
                                times[w][t] = NAN;
                       else
                                times[w][t] = length / 1000.;	
                }
        }
}

/*                                                                                                                                                                                                                  
 * GNU Linear Programming Kit backend                                                                                                                                                                               
 */
#ifdef HAVE_GLPK_H
#include <glpk.h>
static double _glp_resolve(int ns, int nw, int nt, double tasks[nw][nt], double tmax, double w_in_s[ns][nw], int *in_sched_ctxs, int *workers)
{
	struct bound_task_pool * tp;
	int t, w, s;
	glp_prob *lp;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total execution time");

	{
		double times[nw][nt];
		int ne = nt * nw /* worker execution time */
			+ nw * ns
			+ nw * (nt + ns)
			+ 1; /* glp dumbness */
		int n = 1;
		int ia[ne], ja[ne];
		double ar[ne];

		_starpu_get_tasks_times(nw, nt, times, workers);

		/* Variables: number of tasks i assigned to worker j, and tmax */
		glp_add_cols(lp, nw*nt+ns*nw);
#define colnum(w, t) ((t)*nw+(w)+1)
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
				glp_set_obj_coef(lp, nw*nt+s*nw+w+1, 1.);

		for (w = 0; w < nw; w++)
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%dt%dn", w, t);
				glp_set_col_name(lp, colnum(w, t), name);
				glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0., 0.);
			}
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%ds%dn", w, s);
				glp_set_col_name(lp, nw*nt+s*nw+w+1, name);	
				glp_set_col_bnds(lp, nw*nt+s*nw+w+1, GLP_DB, 0.0, 1.0);
			}

		int *sched_ctxs = in_sched_ctxs == NULL ? sched_ctx_hypervisor_get_sched_ctxs() : in_sched_ctxs;

		int curr_row_idx = 0;
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
				return 0.0;
			}
		}
		/*sum(t[t][w]*n[t][w]) < x[s][w]*tmax */
		for(s = 0; s < ns; s++)
		{
			for (w = 0; w < nw; w++)
			{
				char name[32], title[64];
				starpu_worker_get_name(w, name, sizeof(name));
				snprintf(title, sizeof(title), "worker %s", name);
				glp_set_row_name(lp, curr_row_idx+s*nw+w+1, title);
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if(tp->sched_ctx_id == sched_ctxs[s])
					{
						ia[n] = curr_row_idx+s*nw+w+1;
						ja[n] = colnum(w, t);
						if (isnan(times[w][t]))
							ar[n] = 1000000000.;
						else
							ar[n] = times[w][t];
						n++;
					}
				}
				/* x[s][w] = 1 | 0 */
				ia[n] = curr_row_idx+s*nw+w+1;
				ja[n] = nw*nt+s*nw+w+1;
				ar[n] = (-1) * tmax;
				n++;
				glp_set_row_bnds(lp, curr_row_idx+s*nw+w+1, GLP_UP, 0.0, 0.0);
			}
		}

		curr_row_idx += nw*ns;

		/* Total task completion */
		glp_add_rows(lp, nt);
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "task %s key %x", tp->cl->name, (unsigned) tp->footprint);
			glp_set_row_name(lp, curr_row_idx+t+1, title);
			for (w = 0; w < nw; w++)
			{
				ia[n] = curr_row_idx+t+1;
				ja[n] = colnum(w, t);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, curr_row_idx+t+1, GLP_FX, tp->n, tp->n);
		}

		curr_row_idx += nt;

		/* sum(x[s][i]) = 1 */
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
				ja[n] = nw*nt+s*nw+w+1;
				ar[n] = 1;
				n++;
			}

			glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1.0, 1.0);
		}
		if(n != ne)
			printf("ns= %d nw = %d nt = %d n = %d ne = %d\n", ns, nw, nt, n, ne);
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

	int stat = glp_get_prim_stat(lp);
	/* if we don't have a solution return */
	if(stat == GLP_NOFEAS)
	{
		glp_delete_prob(lp);
		lp = NULL;
		return 0.0;
	}

	double res = glp_get_obj_val(lp);
	for (w = 0; w < nw; w++)
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			tasks[w][t] = glp_get_col_prim(lp, colnum(w, t));

	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
			w_in_s[s][w] = glp_get_col_prim(lp, nw*nt+s*nw+w+1);

	glp_delete_prob(lp);
	return res;
}

static void _redistribute_resources_in_ctxs(int ns, int nw, int nt, double w_in_s[ns][nw], unsigned first_time, int *in_sched_ctxs, int *workers)
{
	int *sched_ctxs = in_sched_ctxs == NULL ? sched_ctx_hypervisor_get_sched_ctxs() : in_sched_ctxs;
        struct bound_task_pool * tp;
	int s, s2, w, t;

	for(s = 0; s < ns; s++)
	{
		int workers_to_add[nw], workers_to_remove[nw];
		int destination_ctx[nw][ns];

		for(w = 0; w < nw; w++)
		{
			workers_to_add[w] = -1;
			workers_to_remove[w] = -1;
			for(s2 = 0; s2 < ns; s2++)
				destination_ctx[w][s2] = -1;
		}

		int nadd = 0, nremove = 0;

		for(w = 0; w < nw; w++)
		{
			enum starpu_perf_archtype arch = workers == NULL ? starpu_worker_get_type(w) :
				starpu_worker_get_type(workers[w]);

			if(arch == STARPU_CPU_WORKER)
			{
				if(w_in_s[s][w] >= 0.5)
				{
					workers_to_add[nadd++] = workers == NULL ? w : workers[w];
				}
				else
				{
					workers_to_remove[nremove++] = workers == NULL ? w : workers[w];
					for(s2 = 0; s2 < ns; s2++)
						if(s2 != s && w_in_s[s2][w] >= 0.5)
							destination_ctx[w][s2] = 1;
						else
							destination_ctx[w][s2] = 0;	
				}
			}
			else
			{
				if(w_in_s[s][w] >= 0.3)
				{
					workers_to_add[nadd++] = workers == NULL ? w : workers[w];
				}
				else
				{
					workers_to_remove[nremove++] = workers == NULL ? w : workers[w];
					for(s2 = 0; s2 < ns; s2++)
						if(s2 != s && w_in_s[s2][w] >= 0.3)
							destination_ctx[w][s2] = 1;
						else
							destination_ctx[w][s2] = 0;
				}
			}
	
		}

		sched_ctx_hypervisor_add_workers_to_sched_ctx(workers_to_add, nadd, sched_ctxs[s]);
		struct policy_config *new_config = sched_ctx_hypervisor_get_config(sched_ctxs[s]);
		int i;
		for(i = 0; i < nadd; i++)
			new_config->max_idle[workers_to_add[i]] = new_config->max_idle[workers_to_add[i]] != MAX_IDLE_TIME ? new_config->max_idle[workers_to_add[i]] :  new_config->new_workers_max_idle;
		
		if(!first_time)
		{
			/* do not remove workers if they can't go anywhere */
			int w2;
			unsigned found_one_dest[nremove];
			unsigned all_have_dest = 1;
			for(w2 = 0; w2 < nremove; w2++)
				found_one_dest[w2] = 0;

			for(w2 = 0; w2 < nremove; w2++)
				for(s2 = 0; s2 < ns; s2++)
				{
					/* if the worker has to be removed we should find a destination
					   otherwise we are not interested */
					if(destination_ctx[w2][s2] == -1)
						found_one_dest[w2] = -1;
					if(destination_ctx[w2][s2] == 1)// && sched_ctx_hypervisor_can_resize(sched_ctxs[s2]))
					{
						found_one_dest[w2] = 1;
						break;
					}
				}
			for(w2 = 0; w2 < nremove; w2++)
			{
				if(found_one_dest[w2] == 0)
				{
					all_have_dest = 0;
					break;
				}
			}
			if(all_have_dest)
				sched_ctx_hypervisor_remove_workers_from_sched_ctx(workers_to_remove, nremove, sched_ctxs[s], 0);
		}
	}

}

static double _find_tmax(double t1, double t2)
{
	return t1 + ((t2 - t1)/2);
}

static unsigned _compute_task_distribution_over_ctxs(int ns, int nw, int nt, double w_in_s[ns][nw], int *sched_ctxs, int *workers)
{	
	double tasks[nw][nt];
	double draft_tasks[nw][nt];
	double draft_w_in_s[ns][nw];
	
	int w,t, s;
	for(w = 0; w < nw; w++)
		for(t = 0; t < nt; t++)
		{
			tasks[w][t] = 0.0;
			draft_tasks[w][t] == 0.0;
		}
	
	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			w_in_s[s][w] = 0.0;
			draft_w_in_s[s][w] = 0.0;
		}

	/* smallest possible tmax, difficult to obtain as we 
	   compute the nr of flops and not the tasks */
	double smallest_tmax = _lp_get_tmax(nw, workers);
	double tmax = smallest_tmax * ns;
	
	double res = 1.0;
	unsigned has_sol = 0;
	double tmin = 0.0;
	double old_tmax = 0.0;
	unsigned found_sol = 0;
	/* we fix tmax and we do not treat it as an unknown
	   we just vary by dichotomy its values*/
	while(tmax > 1.0)
	{
		/* find solution and save the values in draft tables
		   only if there is a solution for the system we save them
		   in the proper table */
		res = _glp_resolve(ns, nw, nt, draft_tasks, tmax, draft_w_in_s, sched_ctxs, workers);
		if(res != 0.0)
		{
			for(w = 0; w < nw; w++)
				for(t = 0; t < nt; t++)
					tasks[w][t] = draft_tasks[w][t];
			for(s = 0; s < ns; s++)
				for(w = 0; w < nw; w++)
					w_in_s[s][w] = draft_w_in_s[s][w];
			has_sol = 1;
			found_sol = 1;
		}
		else
			has_sol = 0;
		
		/* if we have a solution with this tmax try a smaller value
		   bigger than the old min */
		if(has_sol)
		{
			if(old_tmax != 0.0 && (old_tmax - tmax) < 0.5)
				break;
			old_tmax = tmax;
		}
		else /*else try a bigger one but smaller than the old tmax */
		{
			tmin = tmax;
			if(old_tmax != 0.0)
				tmax = old_tmax;
		}
		if(tmin == tmax) break;
		tmax = _find_tmax(tmin, tmax);
		
		if(tmax < smallest_tmax)
		{
			tmax = old_tmax;
			tmin = smallest_tmax;
			tmax = _find_tmax(tmin, tmax);
		}
	}
	return found_sol;
}

static void lp2_handle_poped_task(unsigned sched_ctx, int worker)
{
	struct sched_ctx_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	
	int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{
		if(sc_w->submitted_flops < sc_w->total_flops)
		{
			pthread_mutex_unlock(&act_hypervisor_mutex);
			return;
		}

		if(_velocity_gap_btw_ctxs())
		{
			int ns = sched_ctx_hypervisor_get_nsched_ctxs();
			int nw = starpu_worker_get_count(); /* Number of different workers */
			int nt = 0; /* Number of different kinds of tasks */
			struct bound_task_pool * tp;
			for (tp = task_pools; tp; tp = tp->next)
				nt++;
			
			double w_in_s[ns][nw];

			unsigned found_sol = _compute_task_distribution_over_ctxs(ns, nw, nt, w_in_s, NULL, NULL);
			/* if we did find at least one solution redistribute the resources */
			if(found_sol)
			{
				_redistribute_resources_in_ctxs(ns, nw, nt, w_in_s, 0, NULL, NULL);
			}
		}
		pthread_mutex_unlock(&act_hypervisor_mutex);
	}		
}


static void lp2_size_ctxs(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	sched_ctx_hypervisor_save_size_req(sched_ctxs, nsched_ctxs, workers, nworkers);
}

struct hypervisor_policy lp2_policy = {
	.size_ctxs = lp2_size_ctxs,
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

