/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * Record which kinds of tasks have been executed, to later on compute an upper
 * bound of the performance that could have theoretically been achieved
 */

#include <starpu.h>
#include <starpu_config.h>
#include <profiling/bound.h>
#include <core/jobs.h>

#ifdef HAVE_GLPK_H
#include <glpk.h>
#endif /* HAVE_GLPK_H */

/* TODO: output duration between starpu_bound_start and starpu_bound_stop */

struct task_pool {
	/* Which codelet has been executed */
	struct starpu_codelet_t *cl;
	/* Task footprint key */
	uint32_t footprint;
	/* Number of tasks of this kind */
	unsigned long n;
	/* Other tasks */
	struct task_pool *next;
};

static struct task_pool *task_pools, *last;
static int recording;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void starpu_bound_start(void)
{
	struct task_pool *tp;

	PTHREAD_MUTEX_LOCK(&mutex);
	tp = task_pools;
	task_pools = NULL;
	last = NULL;
	recording = 1;
	PTHREAD_MUTEX_UNLOCK(&mutex);

	for ( ; tp; tp = tp->next)
		free(tp);
}

void _starpu_bound_record(starpu_job_t j)
{
	struct task_pool *tp;

	if (!recording)
		return;

	/* No codelet, nothing to measure */
	if (!j->task->cl)
		return;
	/* No performance model, no time duration estimation */
	if (!j->task->cl->model)
		return;
	/* Only support history based */
	if (j->task->cl->model->type != STARPU_HISTORY_BASED)
		return;

	PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!recording) {
		PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	if (STARPU_UNLIKELY(!j->footprint_is_computed))
		_starpu_compute_buffers_footprint(j);

	if (last && last->cl == j->task->cl && last->footprint == j->footprint)
		tp = last;
	else
		for (tp = task_pools; tp; tp = tp->next)
			if (tp->cl == j->task->cl && tp->footprint == j->footprint)
				break;

	if (!tp) {
		tp = malloc(sizeof(*tp));
		tp->cl = j->task->cl;
		tp->footprint = j->footprint;
		tp->n = 0;
		tp->next = task_pools;
		task_pools = tp;
	}

	/* One more task of this kind */
	tp->n++;

	PTHREAD_MUTEX_UNLOCK(&mutex);
}

void starpu_bound_stop(void)
{
	PTHREAD_MUTEX_LOCK(&mutex);
	recording = 0;
	PTHREAD_MUTEX_UNLOCK(&mutex);
}

static void _starpu_get_tasks_times(int nw, int nt, double times[nw][nt]) {
	struct task_pool *tp;
	int w, t;
	for (w = 0; w < nw; w++) {
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			struct starpu_job_s j = {
				.footprint = tp->footprint,
				.footprint_is_computed = 1,
			};
			enum starpu_perf_archtype arch = starpu_worker_get_perf_archtype(w);
			times[w][t] = _starpu_history_based_job_expected_length(tp->cl->model, arch, &j) / 1000.;
		}
	}
}

/*
 * lp_solve format
 */
void starpu_bound_print_lp(FILE *output)
{
	struct task_pool *tp;
	int nt; /* Number of different kinds of tasks */
	int nw; /* Number of different workers */
	int t, w;

	PTHREAD_MUTEX_LOCK(&mutex);

	nw = starpu_worker_get_count();
	nt = 0;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;

	{
		double times[nw][nt];

		_starpu_get_tasks_times(nw, nt, times);

		fprintf(output, "/* StarPU upper bound linear programming problem, to be run in lp_solve. */\n\n");
		fprintf(output, "/* We want to minimize total execution time (ms) */\n");
		fprintf(output, "min: tmax;\n\n");

		fprintf(output, "/* Which is the maximum of all worker execution times (ms) */\n");
		for (w = 0; w < nw; w++) {
			char name[32];
			starpu_worker_get_name(w, name, sizeof(name));
			fprintf(output, "/* worker %s */\n", name);
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				fprintf(output, "\t%+f * w%dt%dn", (float) times[w][t], w, t);
			fprintf(output, " <= tmax;\n");
		}
		fprintf(output, "\n");

		fprintf(output, "/* And we have to have computed exactly all tasks */\n");
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			fprintf(output, "/* task %s key %x */\n", tp->cl->model->symbol, (unsigned) tp->footprint);
			for (w = 0; w < nw; w++)
				fprintf(output, "\t+w%dt%dn", w, t);
			fprintf(output, " = %ld;\n", tp->n);
			/* Show actual values */
			fprintf(output, "/*");
			for (w = 0; w < nw; w++)
				fprintf(output, "\t+%ld", tp->cl->per_worker_stats[w]);
			fprintf(output, "\t*/\n\n");
		}

		fprintf(output, "/* Optionally tell that tasks can not be divided */\n");
		fprintf(output, "/* int ");
		int first = 1;
		for (w = 0; w < nw; w++)
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
				if (!first)
					fprintf(output, ",");
				else
					first = 0;
				fprintf(output, "w%dt%dn", w, t);
			}
		fprintf(output, "; */\n");
	}

	PTHREAD_MUTEX_UNLOCK(&mutex);
}

/*
 * MPS output format
 */
void starpu_bound_print_mps(FILE *output)
{
	struct task_pool * tp;
	int nt; /* Number of different kinds of tasks */
	int nw; /* Number of different workers */
	int t, w;

	PTHREAD_MUTEX_LOCK(&mutex);

	nw = starpu_worker_get_count();
	nt = 0;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;

	{
		double times[nw][nt];

		_starpu_get_tasks_times(nw, nt, times);

		fprintf(output, "NAME           StarPU theoretical bound\n");

		fprintf(output, "\nROWS\n");

		fprintf(output, "* We want to minimize total execution time (ms)\n");
		fprintf(output, " N  TMAX\n");

		fprintf(output, "\n* Which is the maximum of all worker execution times (ms)\n");
		for (w = 0; w < nw; w++) {
			char name[32];
			starpu_worker_get_name(w, name, sizeof(name));
			fprintf(output, "* worker %s\n", name);
			fprintf(output, " L  W%d\n", w);
		}

		fprintf(output, "\n* And we have to have computed exactly all tasks\n");
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			fprintf(output, "* task %s key %x\n", tp->cl->model->symbol, (unsigned) tp->footprint);
			fprintf(output, " E  T%d\n", t);
		}

		fprintf(output, "\nCOLUMNS\n");

		fprintf(output, "\n* Execution times and completion of all tasks\n");
		for (w = 0; w < nw; w++)
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
				char name[9];
				snprintf(name, sizeof(name), "W%dT%d", w, t);
				fprintf(stderr,"    %-8s  W%-7d  %12f\n", name, w, times[w][t]);
				fprintf(stderr,"    %-8s  T%-7d  %12u\n", name, t, 1);
			}

		fprintf(output, "\n* Total execution time\n");
		for (w = 0; w < nw; w++)
			fprintf(stderr,"    TMAX      W%-2d       %12d\n", w, -1);
		fprintf(stderr,"    TMAX      TMAX      %12d\n", 1);

		fprintf(output, "\nRHS\n");

		fprintf(output, "\n* Total number of tasks\n");
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			fprintf(stderr,"    NT%-2d      T%-7d  %12lu\n", t, t, tp->n);

		fprintf(output, "ENDATA\n");
	}

	PTHREAD_MUTEX_UNLOCK(&mutex);
}

/*
 * GNU Linear Programming Kit backend
 */
#ifdef HAVE_GLPK_H
static glp_prob *_starpu_bound_glp_resolve(void)
{
	struct task_pool * tp;
	int nt; /* Number of different kinds of tasks */
	int nw; /* Number of different workers */
	int t, w;
	glp_prob *lp;
	int ret;

	nw = starpu_worker_get_count();
	nt = 0;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MIN);
	glp_set_obj_name(lp, "total execution time");

	{
		double times[nw][nt];
		int ne =
			nw * (nt+1)	/* worker execution time */
			+ nt * nw
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
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
				char name[32];
				snprintf(name, sizeof(name), "w%dt%dn", w, t);
				glp_set_col_name(lp, colnum(w, t), name);
				glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0., 0.);
			}
		glp_set_col_bnds(lp, nw*nt+1, GLP_LO, 0., 0.);

		/* Total worker execution time */
		glp_add_rows(lp, nw);
		for (w = 0; w < nw; w++) {
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "worker %s", name);
			glp_set_row_name(lp, w+1, title);
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
				ia[n] = w+1;
				ja[n] = colnum(w, t);
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
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "task %s key %x", tp->cl->model->symbol, (unsigned) tp->footprint);
			glp_set_row_name(lp, nw+t+1, title);
			for (w = 0; w < nw; w++) {
				ia[n] = nw+t+1;
				ja[n] = colnum(w, t);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, nw+t+1, GLP_FX, tp->n, tp->n);
		}

		STARPU_ASSERT(n == ne);

		glp_load_matrix(lp, ne-1, ia, ja, ar);
	}

	glp_adv_basis(lp, 0);
	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	ret = glp_simplex(lp, &parm);
	if (ret) {
		glp_delete_prob(lp);
		lp = NULL;
	}

	return lp;
}
#endif /* HAVE_GLPK_H */

void starpu_bound_print(FILE *output) {
#ifdef HAVE_GLPK_H
	PTHREAD_MUTEX_LOCK(&mutex);
	glp_prob *lp = _starpu_bound_glp_resolve();
	if (lp) {
		struct task_pool * tp;
		int t, w;
		int nw; /* Number of different workers */
		double tmax;

		nw = starpu_worker_get_count();

		tmax = glp_get_obj_val(lp);

		fprintf(output, "Theoretical minimum execution time: %f ms\n", tmax);

		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			fprintf(output, "%s key %x\n", tp->cl->model->symbol, (unsigned) tp->footprint);
			for (w = 0; w < nw; w++)
				fprintf(output, "\tw%dt%d %f", w, t, glp_get_col_prim(lp, colnum(w, t)));
			fprintf(output, "\n");
		}

		glp_delete_prob(lp);
	} else {
		fprintf(stderr, "Simplex failed\n");
	}
	PTHREAD_MUTEX_UNLOCK(&mutex);
#else /* HAVE_GLPK_H */
	fprintf(output, "Please rebuild StarPU with glpk installed.\n");
#endif /* HAVE_GLPK_H */
}

void starpu_bound_compute(double *res) {
#ifdef HAVE_GLPK_H
	double ret;
	PTHREAD_MUTEX_LOCK(&mutex);
	glp_prob *lp = _starpu_bound_glp_resolve();
	if (lp) {
		ret = glp_get_obj_val(lp);
		glp_delete_prob(lp);
	} else
		ret = 0.;
	PTHREAD_MUTEX_UNLOCK(&mutex);
	*res = ret;
#else /* HAVE_GLPK_H */
	*res = 0.;
#endif /* HAVE_GLPK_H */
}
