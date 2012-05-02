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
#include <math.h>
struct bound_task_pool
{
	/* Which codelet has been executed */
	struct starpu_codelet *cl;
	/* Task footprint key */
	uint32_t footprint;
	/* Context the task belongs to */
	unsigned sched_ctx_id;
	/* Number of tasks of this kind */
	unsigned long n;
	/* Other task kinds */
	struct bound_task_pool *next;
};


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
                }
        }
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
static void _glp_resolve(int ns, int nw, int nt, double res[ns][nw][nt], int integer)
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
				if (integer)
					glp_set_col_kind(lp, colnum(w, t), GLP_IV);
				glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0., 0.);
			}
		glp_set_col_bnds(lp, nw*nt+1, GLP_LO, 0., 0.);

		/* Total worker execution time */
		glp_add_rows(lp, nw);
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
			glp_set_row_name(lp, w+1, title);
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			{
				ia[n] = w+1;
				ja[n] = colnum(w, t);
				if (isnan(times[w][t]))
					ar[n] = 1000000000.;
				else
					ar[n] = times[w][t];
				n++;
			}
			/* tmax */
			ia[n] = w+1;
			ja[n] = nw*nt+1;
			ar[n] = -1;
			n++;
			glp_set_row_bnds(lp, w+1, GLP_UP, 0, 0);
		}

		/* Total task completion */
		glp_add_rows(lp, nt);
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "task %s key %x", tp->cl->name, (unsigned) tp->footprint);
			glp_set_row_name(lp, nw+t+1, title);
			for (w = 0; w < nw; w++)
			{
				ia[n] = nw+t+1;
				ja[n] = colnum(w, t);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, nw+t+1, GLP_FX, tp->n, tp->n);
		}

		int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
		/* Number of task * time > 0.3 * tmax */
		glp_add_rows(lp, nw*ns);
		for (w = 0; w < nw; w++)
		{
			for(s = 0; s < ns; s++)
			{
				char name[32], title[64];
				starpu_worker_get_name(w, name, sizeof(name));
				snprintf(title, sizeof(title), "worker %x ctx %x limit", w, s);
				glp_set_row_name(lp, nw+nt+w+(s*nw)+1, title);
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if(tp->sched_ctx_id == sched_ctxs[s])
					{
						ia[n] = nw+nt+w+(s*nw)+1;
						ja[n] = colnum(w, t);
						ar[n] = 1;
						n++;
					}
				}

				/* tmax */
				ia[n] = nw+nt+w+(s*nw)+1;
				ja[n] = nw*nt+1;
				ar[n] = -0.3;
				n++;

				glp_set_row_bnds(lp, nw+nt+w+(s*nw)+1, GLP_UP, 0.0, 0.0);
			}
		}

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

        if (integer)
        {
                glp_iocp iocp;
                glp_init_iocp(&iocp);
                iocp.msg_lev = GLP_MSG_OFF;
                glp_intopt(lp, &iocp);
        }


	double tmax = glp_get_obj_val(lp);

        printf("Theoretical minimum execution time: %f ms\n", tmax);

	for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
	{
		for (w = 0; w < nw; w++)
		{
			s = _get_idx_sched_ctx(tp->sched_ctx_id);
			res[s][w][t] = glp_get_col_prim(lp, colnum(w, t));
		}
	}

	glp_delete_prob(lp);
}

void _redistribute_resources_in_ctxs2(int ns, int nw, int nt, double res[ns][nw][nt])
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
					if(res[s][w][t] >= 1.0)
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
		if(nadd != nworkers_ctx)
		{
			printf("%d: add %d \n", sched_ctxs[s], nadd);
			printf("%d: remove %d \n", sched_ctxs[s], nremove);
			sched_ctx_hypervisor_add_workers_to_sched_ctx(workers_to_add, nadd, sched_ctxs[s]);
			sched_ctx_hypervisor_remove_workers_from_sched_ctx(workers_to_remove, nremove, sched_ctxs[s]);

			struct policy_config *new_config = sched_ctx_hypervisor_get_config(sched_ctxs[s]);
			int i;
			for(i = 0; i < nadd; i++)
				new_config->max_idle[workers_to_add[i]] = new_config->max_idle[workers_to_add[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_add[i]] :  new_config->new_workers_max_idle;
		}
	}
}

void lp2_handle_poped_task(unsigned sched_ctx, int worker)
{
	if(_velocity_gap_btw_ctxs())
	{
		int ns = sched_ctx_hypervisor_get_nsched_ctxs();
		int nw = starpu_worker_get_count(); /* Number of different workers */
		int nt = 0; /* Number of different kinds of tasks */
		struct bound_task_pool * tp;
		for (tp = task_pools; tp; tp = tp->next)
			nt++;
		
       		double res[ns][nw][nt];

		int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{
			_glp_resolve(ns, nw, nt, res, 0);
/* 			int i, j, k; */
/* 			for( i = 0; i < ns; i++) */
/* 				for(j = 0; j < nw; j++) */
/* 					for(k = 0; k < nt; k++) */
/* 					{ */
/* 						printf("ctx %d/worker %d/task type %d: res = %lf \n", i, j, k, res[i][j][k]); */
/* 					} */
		
			_redistribute_resources_in_ctxs2(ns, nw, nt, res);
			pthread_mutex_unlock(&act_hypervisor_mutex);
		}
	}		
}

struct hypervisor_policy lp2_policy = {
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

