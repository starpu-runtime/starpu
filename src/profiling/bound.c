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
#include <profiling/bound.h>
#include <core/jobs.h>

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

		/* And we have to have computed exactly all tasks */
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			fprintf(output, "/* task %s key %lx */\n", tp->cl->model->symbol, (unsigned long) tp->footprint);
			for (w = 0; w < nw; w++)
				fprintf(output, "\t+w%dt%dn", w, t);
			fprintf(output, " = %ld;\n", tp->n);
			/* Show actual values */
			fprintf(output, "/*");
			for (w = 0; w < nw; w++)
				fprintf(output, "\t+%ld", tp->cl->per_worker_stats[w]);
			fprintf(output, "\t*/\n\n");
		}

		/* Optionally tell that tasks can not be divided */
		fprintf(output, "int ");
		int first = 1;
		for (w = 0; w < nw; w++)
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
				if (!first)
					fprintf(output, ",");
				else
					first = 0;
				fprintf(output, "w%dt%dn", w, t);
			}
		fprintf(output, ";\n");
	}

	PTHREAD_MUTEX_UNLOCK(&mutex);
}

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

		fprintf(output, "NAME           StarPU theoretical bound");
		fprintf(output, "ROWS");
		fprintf(output, "TODO");
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next) {
			for (w = 0; w < nw; w++)
				;
		}
	}

	PTHREAD_MUTEX_UNLOCK(&mutex);
}

void starpu_bound_print(FILE *output)
{
	fprintf(output, "TODO: use glpk");
}
