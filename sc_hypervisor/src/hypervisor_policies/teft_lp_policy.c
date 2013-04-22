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

static void _get_tasks_times(int nw, int nt, double times[nw][nt], int *workers, unsigned size_ctxs)
{
        struct sc_hypervisor_policy_task_pool *tp;
        int w, t;
        for (w = 0; w < nw; w++)
        {
                for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
                {
			int worker = workers == NULL ? w : workers[w];
                        enum starpu_perf_archtype arch = starpu_worker_get_perf_archtype(worker);
                        double length = starpu_history_based_expected_perf(tp->cl->model, arch, tp->footprint);

                        if (isnan(length))
                                times[w][t] = NAN;
			else
			{
                                times[w][t] = length / 1000.;

				double transfer_time = 0.0;
				enum starpu_archtype arch = starpu_worker_get_type(worker);
				if(arch == STARPU_CUDA_WORKER)
				{
					unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, tp->sched_ctx_id);
					if(!worker_in_ctx && !size_ctxs)
					{
						double transfer_velocity = starpu_get_bandwidth_RAM_CUDA(worker);
						transfer_time +=  (tp->footprint / transfer_velocity) / 1000. ;
					}
					double latency = starpu_get_latency_RAM_CUDA(worker);
					transfer_time += latency/1000.;

				}
//				printf("%d/%d %s x %d time = %lf transfer_time = %lf\n", w, tp->sched_ctx_id, tp->cl->model->symbol, tp->n, times[w][t], transfer_time);
				times[w][t] += transfer_time;
			}
                }
        }
}



/*
 * GNU Linear Programming Kit backend
 */
#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
static double _glp_resolve(int ns, int nw, double final_w_in_s[ns][nw], 
			   unsigned is_integer, double tmax, void *specific_data)
{
	struct teft_lp_data *sd = (struct teft_lp_data *)specific_data;

	int nt = sd->nt;
	double **final_tasks = sd->tasks;
	int *in_sched_ctxs = sd->in_sched_ctxs;
	int *workers = sd->workers;
	struct sc_hypervisor_policy_task_pool *tmp_task_pools = sd->tmp_task_pools;
	unsigned size_ctxs = sd->size_ctxs;
	
	double w_in_s[ns][nw];
	double tasks[nw][nt];
	
	if(tmp_task_pools == NULL)
		return 0.0;
	struct sc_hypervisor_policy_task_pool * tp;
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

		_get_tasks_times(nw, nt, times, workers, size_ctxs);

		/* Variables: number of tasks i assigned to worker j, and tmax */
		glp_add_cols(lp, nw*nt+ns*nw);
#define colnum(w, t) ((t)*nw+(w)+1)
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
				glp_set_obj_coef(lp, nw*nt+s*nw+w+1, 1.);

		for (w = 0; w < nw; w++)
			for (t = 0; t < nt; t++)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%dt%dn", w, t);
				glp_set_col_name(lp, colnum(w, t), name);
/* 				if (integer) */
/*                                 { */
/*                                         glp_set_col_kind(lp, colnum(w, t), GLP_IV); */
/* 					glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0, 0); */
/*                                 } */
/* 				else */
					glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0.0, 0.0);
			}
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%ds%dn", w, s);
				glp_set_col_name(lp, nw*nt+s*nw+w+1, name);
				if (is_integer)
                                {
                                        glp_set_col_kind(lp, nw*nt+s*nw+w+1, GLP_IV);
                                        glp_set_col_bnds(lp, nw*nt+s*nw+w+1, GLP_DB, 0, 1);
                                }
                                else
					glp_set_col_bnds(lp, nw*nt+s*nw+w+1, GLP_DB, 0.0, 1.0);
			}

		int *sched_ctxs = in_sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : in_sched_ctxs;

		int curr_row_idx = 0;
		/* Total worker execution time */
		glp_add_rows(lp, nw*ns);
		for (t = 0; t < nt; t++)
		{
			int someone = 0;
			for (w = 0; w < nw; w++)
				if (!isnan(times[w][t]))
					someone = 1;
			if (!someone)
			{
				/* This task does not have any performance model at all, abort */
				printf("NO PERF MODELS\n");
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
				for (t = 0, tp = tmp_task_pools; tp; t++, tp = tp->next)
				{
					if((int)tp->sched_ctx_id == sched_ctxs[s])
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
		for (t = 0, tp = tmp_task_pools; tp; t++, tp = tp->next)
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
			if(is_integer)
                                glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1, 1);
			else
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

/* 	char str[50]; */
/* 	sprintf(str, "outpu_lp_%g", tmax); */

/* 	glp_print_sol(lp, str); */

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
//		printf("no_sol in tmax = %lf\n", tmax);
		lp = NULL;
		return 0.0;
	}


	if (is_integer)
        {
                glp_iocp iocp;
                glp_init_iocp(&iocp);
                iocp.msg_lev = GLP_MSG_OFF;
		glp_intopt(lp, &iocp);
		int stat = glp_mip_status(lp);
		/* if we don't have a solution return */
		if(stat == GLP_NOFEAS)
		{
//			printf("no int sol in tmax = %lf\n", tmax);
			glp_delete_prob(lp);
			lp = NULL;
			return 0.0;
		}
	}

	double res = glp_get_obj_val(lp);
	for (w = 0; w < nw; w++)
		for (t = 0; t < nt; t++)
/* 			if (integer) */
/* 				tasks[w][t] = (double)glp_mip_col_val(lp, colnum(w, t)); */
/*                         else */
				tasks[w][t] = glp_get_col_prim(lp, colnum(w, t));
	
//	printf("for tmax %lf\n", tmax);
	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			if (is_integer)
				w_in_s[s][w] = (double)glp_mip_col_val(lp, nw*nt+s*nw+w+1);
                        else
				w_in_s[s][w] = glp_get_col_prim(lp, nw*nt+s*nw+w+1);
//			printf("w_in_s[%d][%d]=%lf\n", s, w, w_in_s[s][w]);
		}
//	printf("\n");

	glp_delete_prob(lp);
	if(res != 0.0)
	{
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
	double tasks[nw][nt];

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
								tmin, tmax, smallest_tmax, _glp_resolve);

	starpu_pthread_mutex_unlock(&mutex);
	/* if we did find at least one solution redistribute the resources */
	if(found_sol)
		sc_hypervisor_lp_place_resources_in_ctx(ns, nw, w_in_s, sched_ctxs, workers, 1);
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
								tmin, tmax, smallest_tmax, _glp_resolve);
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

#endif /* STARPU_HAVE_GLPK_H */
