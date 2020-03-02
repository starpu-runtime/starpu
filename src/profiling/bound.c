/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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

/*
 * Record which kinds of tasks have been executed, to later on compute an upper
 * bound of the performance that could have theoretically been achieved
 */

#include <starpu.h>
#include <starpu_config.h>
#include <profiling/bound.h>
#include <core/jobs.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
#endif /* STARPU_HAVE_GLPK_H */

/* TODO: output duration between starpu_bound_start and starpu_bound_stop */

/* TODO: compute critical path and introduce it in the LP */

/*
 * Record without dependencies: just count each kind of task
 *
 * The linear programming problem will just have as variables:
 * - the number of tasks of kind `t' executed by worker `w'
 * - the total duration
 *
 * and the constraints will be:
 * - the time taken by each worker to complete its assigned tasks is lower than
 *   the total duration.
 * - the total numer of tasks of a given kind is equal to the number run by the
 *   application.
 */
struct bound_task_pool
{
	/* Which codelet has been executed */
	struct starpu_codelet *cl;
	/* Task footprint key (for history-based perfmodel) */
	uint32_t footprint;
	/* Number of tasks of this kind */
	unsigned long n;
	/* Other task kinds */
	struct bound_task_pool *next;
};

/*
 * Record with dependencies: each task is recorded separately
 *
 * The linear programming problem will have as variables:
 * - The start time of each task
 * - The completion time of each tag
 * - The total duration
 * - For each task and for each worker, whether the task is executing on that worker.
 * - For each pair of task, which task is scheduled first.
 *
 * and the constraints will be:
 * - All task start time plus duration are less than total duration
 * - Each task is executed on exactly one worker.
 * - Each task starts after all its task dependencies finish.
 * - Each task starts after all its tag dependencies finish.
 * - For each task pair and each worker, if both tasks are executed by that worker,
 *   one is started after the other's completion.
 */
struct task_dep
{
	/* Task this depends on */
	struct bound_task *dep;
	/* Data transferred between tasks (i.e. implicit data dep size) */
	size_t size;
};
struct bound_task
{
	/* Unique ID */
	unsigned long id;
	/* Tag ID, if any */
	starpu_tag_t tag_id;
	int use_tag;
	/* Which codelet has been executed */
	struct starpu_codelet *cl;
	/* Task footprint key */
	uint32_t footprint;
	/* Task priority */
	int priority;
	/* Tasks this one depends on */
	struct task_dep *deps;
	int depsn;

	/* Estimated duration */
	double** duration[STARPU_NARCH];

	/* Other tasks */
	struct bound_task *next;
};

struct bound_tag_dep
{
	starpu_tag_t tag;
	starpu_tag_t dep_tag;
	struct bound_tag_dep *next;
};

static struct bound_task_pool *task_pools, *last;
static struct bound_task *tasks;
static struct bound_tag_dep *tag_deps;
int _starpu_bound_recording;
static int recorddeps;
static int recordprio;

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

static void _starpu_bound_clear(int record, int deps, int prio)
{
	struct bound_task_pool *tp;
	struct bound_task *t;
	struct bound_tag_dep *td;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);

	tp = task_pools;
	task_pools = NULL;
	last = NULL;

	t = tasks;
	tasks = NULL;

	td = tag_deps;
	tag_deps = NULL;

	_starpu_bound_recording = record;
	recorddeps = deps;
	recordprio = prio;

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	while (tp != NULL)
	{
		struct bound_task_pool *next = tp->next;
		free(tp);
		tp = next;
	}

	while (t != NULL)
	{
		struct bound_task *next = t->next;
		unsigned i,j;
		for (i = 0; i < STARPU_NARCH; i++)
		{
			if (t->duration[i])
			{
				for (j = 0; t->duration[i][j]; j++)
					free(t->duration[i][j]);
				free(t->duration[i]);
			}
		}
		free(t->deps);
		free(t);
		t = next;
	}

	while (td != NULL)
	{
		struct bound_tag_dep *next = td->next;
		free(td);
		td = next;
	}
}

void starpu_bound_clear(void)
{
	_starpu_bound_clear(0, 0, 0);
}

/* Initialization */
void starpu_bound_start(int deps, int prio)
{
	_starpu_bound_clear(1, deps, prio);
}

/* Whether we will include it in the computation */
static int good_job(struct _starpu_job *j)
{
	/* No codelet, nothing to measure */
	if (j->exclude_from_dag)
		return 0;
	if (!j->task->cl)
		return 0;
	/* No performance model, no time duration estimation */
	if (!j->task->cl->model)
		return 0;
	/* Only support history based */
	if (j->task->cl->model->type != STARPU_HISTORY_BASED
	 && j->task->cl->model->type != STARPU_NL_REGRESSION_BASED)
		return 0;
	return 1;
}
static double** initialize_arch_duration(int maxdevid, unsigned* maxncore_table)
{
	int devid, maxncore;
	double ** arch_model;
	_STARPU_MALLOC(arch_model, sizeof(*arch_model)*(maxdevid+1));
	arch_model[maxdevid] = NULL;
	for(devid=0; devid<maxdevid; devid++)
	{
		if(maxncore_table != NULL)
			maxncore = maxncore_table[devid];
		else
			maxncore = 1;
		_STARPU_CALLOC(arch_model[devid], maxncore+1,sizeof(*arch_model[devid]));
	}
	return arch_model;
}

static void initialize_duration(struct bound_task *task)
{
	struct _starpu_machine_config *conf = _starpu_get_machine_config();
	task->duration[STARPU_CPU_WORKER] = initialize_arch_duration(1,&conf->topology.nhwcpus);
	task->duration[STARPU_CUDA_WORKER] = initialize_arch_duration(conf->topology.nhwcudagpus,NULL);
	task->duration[STARPU_OPENCL_WORKER] = initialize_arch_duration(conf->topology.nhwopenclgpus,NULL);
	task->duration[STARPU_MIC_WORKER] = initialize_arch_duration(conf->topology.nhwmicdevices,conf->topology.nmiccores);
}

static struct starpu_perfmodel_device device =
{
	.type = STARPU_CPU_WORKER,
	.devid = 0,
	.ncores = 1,
};
static struct starpu_perfmodel_arch dumb_arch =
{
	.ndevices = 1,
	.devices = &device,
};

/* Create a new task (either because it has just been submitted, or a
 * dependency was added before submission) */
static void new_task(struct _starpu_job *j)
{
	struct bound_task *t;

	if (j->bound_task)
		return;

	_STARPU_CALLOC(t, 1, sizeof(*t));
	t->id = j->job_id;
	t->tag_id = j->task->tag_id;
	t->use_tag = j->task->use_tag;
	t->cl = j->task->cl;
	t->footprint = _starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, &dumb_arch, 0, j);
	t->priority = j->task->priority;
	t->deps = NULL;
	t->depsn = 0;
	initialize_duration(t);
	t->next = tasks;
	j->bound_task = t;
	tasks = t;
}

/* A new task was submitted, record it */
void _starpu_bound_record(struct _starpu_job *j)
{
	if (!_starpu_bound_recording)
		return;

	if (!good_job(j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	if (recorddeps)
	{
		new_task(j);
	}
	else
	{
		struct bound_task_pool *tp;

		_starpu_compute_buffers_footprint(j->task->cl?j->task->cl->model:NULL, NULL, 0, j);

		if (last && last->cl == j->task->cl && last->footprint == j->footprint)
			tp = last;
		else
			for (tp = task_pools; tp; tp = tp->next)
				if (tp->cl == j->task->cl && tp->footprint == j->footprint)
					break;

		if (!tp)
		{
			_STARPU_MALLOC(tp, sizeof(*tp));
			tp->cl = j->task->cl;
			tp->footprint = j->footprint;
			tp->n = 0;
			tp->next = task_pools;
			task_pools = tp;
		}

		/* One more task of this kind */
		tp->n++;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/* A tag dependency was emitted, record it */
void _starpu_bound_tag_dep(starpu_tag_t id, starpu_tag_t dep_id)
{
	struct bound_tag_dep *td;

	if (!_starpu_bound_recording || !recorddeps)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording || !recorddeps)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	_STARPU_MALLOC(td, sizeof(*td));
	td->tag = id;
	td->dep_tag = dep_id;
	td->next = tag_deps;
	tag_deps = td;
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/* A task dependency was emitted, record it */
void _starpu_bound_task_dep(struct _starpu_job *j, struct _starpu_job *dep_j)
{
	struct bound_task *t;
	int i;

	if (!_starpu_bound_recording || !recorddeps)
		return;

	if (!good_job(j) || !good_job(dep_j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording || !recorddeps)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	new_task(j);
	new_task(dep_j);
	t = j->bound_task;
	for (i = 0; i < t->depsn; i++)
		if (t->deps[i].dep == dep_j->bound_task)
			break;
	if (i == t->depsn)
	{
		/* Not already there, add */
		_STARPU_REALLOC(t->deps, ++t->depsn * sizeof(t->deps[0]));
		t->deps[t->depsn-1].dep = dep_j->bound_task;
		t->deps[t->depsn-1].size = 0; /* We don't have data information in that case */
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/* Look for job with id ID among our tasks */
static struct bound_task *find_job(unsigned long id)
{
	struct bound_task *t;

	for (t = tasks; t; t = t->next)
		if (t->id == id)
			return t;
	return NULL;
}

/* Job J depends on previous job of id ID (which is already finished) */
void _starpu_bound_job_id_dep_size(size_t size, struct _starpu_job *j, unsigned long id)
{
	struct bound_task *t, *dep_t;
	int i;

	if (!_starpu_bound_recording || !recorddeps)
		return;

	if (!good_job(j))
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	/* Re-check, this time with mutex held */
	if (!_starpu_bound_recording || !recorddeps)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	new_task(j);
	dep_t = find_job(id);
	if (!dep_t)
	{
		_STARPU_MSG("dependency %lu not found !\n", id);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}
	t = j->bound_task;
	for (i = 0; i < t->depsn; i++)
		if (t->deps[i].dep == dep_t)
		{
			/* Found, just add size */
			t->deps[i].size += size;
			break;
		}
	if (i == t->depsn)
	{
		/* Not already there, add */
		_STARPU_REALLOC(t->deps, ++t->depsn * sizeof(t->deps[0]));
		t->deps[t->depsn-1].dep = dep_t;
		t->deps[t->depsn-1].size = size;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

void _starpu_bound_job_id_dep(starpu_data_handle_t handle, struct _starpu_job *j, unsigned long id)
{
	if (!_starpu_bound_recording || !recorddeps)
		return;

	if (!good_job(j))
		return;

	_starpu_bound_job_id_dep_size(_starpu_data_get_size(handle), j, id);
}

void starpu_bound_stop(void)
{
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	_starpu_bound_recording = 0;
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/* Compute all tasks times on all workers */
static void _starpu_get_tasks_times(int nw, int nt, double *times)
{
	struct bound_task_pool *tp;
	int w, t;
	for (w = 0; w < nw; w++)
	{
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			struct _starpu_job j =
			{
				.footprint = tp->footprint,
				.footprint_is_computed = 1,
			};
			struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(w, STARPU_NMAX_SCHED_CTXS);
			double length = _starpu_history_based_job_expected_perf(tp->cl->model, arch, &j, j.nimpl);
			if (isnan(length))
				times[w*nt+t] = NAN;
			else
				times[w*nt+t] = length / 1000.;
		}
	}
}

/* Return whether PARENT is an ancestor of CHILD */
static int ancestor(struct bound_task *child, struct bound_task *parent)
{
	int i;
	for (i = 0; i < child->depsn; i++)
	{
		if (parent == child->deps[i].dep)
			return 1;
		if (ancestor(child->deps[i].dep, parent))
			return -1;
	}
	return 0;
}

/* Print bound recording in .dot format */
void starpu_bound_print_dot(FILE *output)
{
	struct bound_task *t;
	struct bound_tag_dep *td;
	int i;

	if (!recorddeps)
	{
		fprintf(output, "Not supported\n");
		return;
	}
	fprintf(output, "strict digraph bounddeps {\n");
	for (t = tasks; t; t = t->next)
	{
		fprintf(output, "\"t%lu\" [label=\"%lu: %s\"]\n", t->id, t->id, _starpu_codelet_get_model_name(t->cl));
		for (i = 0; i < t->depsn; i++)
			fprintf(output, "\"t%lu\" -> \"t%lu\"\n", t->deps[i].dep->id, t->id);
	}
	for (td = tag_deps; td; td = td->next)
		fprintf(output, "\"tag%lu\" -> \"tag%lu\";\n", (unsigned long) td->dep_tag, (unsigned long) td->tag);
	fprintf(output, "}\n");
}

/*
 * Print bound system in lp_solve format
 *
 * When dependencies are enabled, you can check the set of tasks and deps that
 * were recorded by using tools/lp2paje and vite.
 */
void starpu_bound_print_lp(FILE *output)
{
	int nt; /* Number of different kinds of tasks */
	int nw; /* Number of different workers */
	int t;
	int w, w2; /* worker */
	unsigned n, n2;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	nw = starpu_worker_get_count();
	if (!nw)
		/* Make llvm happy about the VLA below */
		return;

	if (recorddeps)
	{
		struct bound_task *t1, *t2;
		struct bound_tag_dep *td;
		int i;

		nt = 0;
		for (t1 = tasks; t1; t1 = t1->next)
		{
			if (t1->cl->model->type != STARPU_HISTORY_BASED &&
			    t1->cl->model->type != STARPU_NL_REGRESSION_BASED)
				/* TODO: */
				_STARPU_MSG("Warning: task %s uses a perf model which is neither history nor non-linear regression-based, support for such model is not implemented yet, system will not be solvable.\n", _starpu_codelet_get_model_name(t1->cl));

			struct _starpu_job j =
			{
				.footprint = t1->footprint,
				.footprint_is_computed = 1,
			};
			for (w = 0; w < nw; w++)
			{
				struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(w, STARPU_NMAX_SCHED_CTXS);
				if (_STARPU_IS_ZERO(t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores]))
				{
					double length = _starpu_history_based_job_expected_perf(t1->cl->model, arch, &j,j.nimpl);
					if (isnan(length))
						/* Avoid problems with binary coding of doubles */
						t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores] = NAN;
					else
						t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores] = length / 1000.;
				}
			}
			nt++;
		}
		if (!nt)
			return;
		fprintf(output, "/* StarPU upper bound linear programming problem, to be run in lp_solve. */\n\n");
		fprintf(output, "/* !! This is a big system, it will be long to solve !! */\n\n");

		fprintf(output, "/* We want to minimize total execution time (ms) */\n");
		fprintf(output, "min: tmax;\n\n");

		fprintf(output, "/* Number of tasks */\n");
		fprintf(output, "nt = %d;\n", nt);
		fprintf(output, "/* Number of workers */\n");
		fprintf(output, "nw = %d;\n", nw);

		fprintf(output, "/* The total execution time is the maximum of all task completion times (ms) */\n");
		for (t1 = tasks; t1; t1 = t1->next)
			fprintf(output, "c%lu <= tmax;\n", t1->id);

		fprintf(output, "\n/* We have tasks executing on workers, exactly one worker executes each task */\n");
		for (t1 = tasks; t1; t1 = t1->next)
		{
			for (w = 0; w < nw; w++)
			{
				struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(w, STARPU_NMAX_SCHED_CTXS);
				if (!isnan(t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores]))
					fprintf(output, " +t%luw%d", t1->id, w);
			}
			fprintf(output, " = 1;\n");
		}

		fprintf(output, "\n/* Completion time is start time plus computation time */\n");
		fprintf(output, "/* According to where the task is indeed executed */\n");
		for (t1 = tasks; t1; t1 = t1->next)
		{
			fprintf(output, "/* %s %x */\tc%lu = s%lu", _starpu_codelet_get_model_name(t1->cl), (unsigned) t1->footprint, t1->id, t1->id);
			for (w = 0; w < nw; w++)
			{
				struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(w, STARPU_NMAX_SCHED_CTXS);
				if (!isnan(t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores]))
					fprintf(output, " + %f t%luw%d", t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores], t1->id, w);
			}
			fprintf(output, ";\n");
		}

		fprintf(output, "\n/* Each task starts after all its task dependencies finish and data is transferred. */\n");
		fprintf(output, "/* Note that the dependency finish time depends on the worker where it's working */\n");
		for (t1 = tasks; t1; t1 = t1->next)
			for (i = 0; i < t1->depsn; i++)
			{
				fprintf(output, "/* %lu bytes transferred */\n", (unsigned long) t1->deps[i].size);
				fprintf(output, "s%lu >= c%lu", t1->id, t1->deps[i].dep->id);
				/* Transfer time: pick up one source node and a worker on it */
				for (n = 0; n < starpu_memory_nodes_get_count(); n++)
				for (w = 0; w < nw; w++)
				if (starpu_worker_get_memory_node(w) == n)
				{
					/* pick up another destination node and a worker on it */
					for (n2 = 0; n2 < starpu_memory_nodes_get_count(); n2++)
					if (n2 != n)
					{
						for (w2 = 0; w2 < nw; w2++)
						if (starpu_worker_get_memory_node(w2) == n2)
						{
							/* If predecessor is on worker w and successor
							 * on worker w2 on different nodes, we need to
							 * transfer the data. */
							fprintf(output, " + d_t%luw%dt%luw%d", t1->deps[i].dep->id, w, t1->id, w2);

						}
					}
				}
				fprintf(output, ";\n");
				/* Transfer time: pick up one source node and a worker on it */
				for (n = 0; n < starpu_memory_nodes_get_count(); n++)
				for (w = 0; w < nw; w++)
				if (starpu_worker_get_memory_node(w) == n)
				{
					/* pick up another destination node and a worker on it */
					for (n2 = 0; n2 < starpu_memory_nodes_get_count(); n2++)
					if (n2 != n)
					{
						for (w2 = 0; w2 < nw; w2++)
						if (starpu_worker_get_memory_node(w2) == n2)
						{
							/* The data transfer is at least 0ms */
							fprintf(output, "d_t%luw%dt%luw%d >= 0;\n", t1->deps[i].dep->id, w, t1->id, w2);
							/* The data transfer from w to w2 only happens if tasks run there */
							fprintf(output, "d_t%luw%dt%luw%d >= %f - 2e5 + 1e5 t%luw%d + 1e5 t%luw%d;\n",
									t1->deps[i].dep->id, w, t1->id, w2,
									starpu_transfer_predict(n, n2, t1->deps[i].size)/1000.,
									t1->deps[i].dep->id, w, t1->id, w2);
						}
					}
				}
			}


		fprintf(output, "\n/* Each tag finishes when its corresponding task finishes */\n");
		for (t1 = tasks; t1; t1 = t1->next)
			if (t1->use_tag)
			{
				for (w = 0; w < nw; w++)
					fprintf(output, "c%lu = tag%lu;\n", t1->id, (unsigned long) t1->tag_id);
			}

		fprintf(output, "\n/* tags start after all their tag dependencies finish. */\n");
		for (td = tag_deps; td; td = td->next)
			fprintf(output, "tag%lu >= tag%lu;\n", (unsigned long) td->tag, (unsigned long) td->dep_tag);

/* TODO: factorize ancestor calls */
		fprintf(output, "\n/* For each task pair and each worker, if both tasks are executed by the same worker,\n");
		fprintf(output, "   one is started after the other's completion */\n");
		for (t1 = tasks; t1; t1 = t1->next)
		{
			for (t2 = t1->next; t2; t2 = t2->next)
			{
				if (!ancestor(t1, t2) && !ancestor(t2, t1))
				{
					for (w = 0; w < nw; w++)
					{
						struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(w, STARPU_NMAX_SCHED_CTXS);
						if (!isnan(t1->duration[arch->devices[0].type][arch->devices[0].devid][arch->devices[0].ncores]))
						{
							fprintf(output, "s%lu - c%lu >= -3e5 + 1e5 t%luw%d + 1e5 t%luw%d + 1e5 t%luafter%lu;\n",
									t1->id, t2->id, t1->id, w, t2->id, w, t1->id, t2->id);
							fprintf(output, "s%lu - c%lu >= -2e5 + 1e5 t%luw%d + 1e5 t%luw%d - 1e5 t%luafter%lu;\n",
									t2->id, t1->id, t1->id, w, t2->id, w, t1->id, t2->id);
						}
					}
				}
			}
		}

#if 0
/* Doesn't help at all to actually express what "after" means */
		for (t1 = tasks; t1; t1 = t1->next)
			for (t2 = t1->next; t2; t2 = t2->next)
				if (!ancestor(t1, t2) && !ancestor(t2, t1))
				{
					fprintf(output, "s%lu - s%lu >= -1e5 + 1e5 t%luafter%lu;\n", t1->id, t2->id, t1->id, t2->id);
					fprintf(output, "s%lu - s%lu >= -1e5 t%luafter%lu;\n", t2->id, t1->id, t1->id, t2->id);
				}
#endif

		if (recordprio)
		{
			fprintf(output, "\n/* For StarPU, a priority means given schedulable tasks it will consider the\n");
			fprintf(output, " * more prioritized first */\n");
			for (t1 = tasks; t1; t1 = t1->next)
			{
				for (t2 = t1->next; t2; t2 = t2->next)
				{
					if (!ancestor(t1, t2) && !ancestor(t2, t1)
					     && t1->priority != t2->priority)
					{
						if (t1->priority > t2->priority)
						{
							/* Either t2 is scheduled before t1, but then it
							   needs to be scheduled before some t dep finishes */

							/* One of the t1 deps to give the maximum start time for t2 */
							if (t1->depsn > 1)
							{
								for (i = 0; i < t1->depsn; i++)
									fprintf(output, " + t%lut%lud%d", t2->id, t1->id, i);
								fprintf(output, " = 1;\n");
							}

							for (i = 0; i < t1->depsn; i++)
							{
								fprintf(output, "c%lu - s%lu >= ", t1->deps[i].dep->id, t2->id);
								if (t1->depsn > 1)
									/* Only checks this when it's this dependency that is chosen */
									fprintf(output, "-2e5 + 1e5 t%lut%lud%d", t2->id, t1->id, i);
								else
									fprintf(output, "-1e5");
								/* Only check this if t1 is after t2 */
								fprintf(output, " + 1e5 t%luafter%lu", t1->id, t2->id);
								fprintf(output, ";\n");
							}

							/* Or t2 is scheduled after t1 is.  */
							fprintf(output, "s%lu - s%lu >= -1e5 t%luafter%lu;\n", t2->id, t1->id, t1->id, t2->id);
						}
						else
						{
							/* Either t1 is scheduled before t2, but then it
							   needs to be scheduled before some t2 dep finishes */

							/* One of the t2 deps to give the maximum start time for t1 */
							if (t2->depsn > 1)
							{
								for (i = 0; i < t2->depsn; i++)
									fprintf(output, " + t%lut%lud%d", t1->id, t2->id, i);
								fprintf(output, " = 1;\n");
							}

							for (i = 0; i < t2->depsn; i++)
							{
								fprintf(output, "c%lu - s%lu >= ", t2->deps[i].dep->id, t1->id);
								if (t2->depsn > 1)
									/* Only checks this when it's this dependency that is chosen */
									fprintf(output, "-1e5 + 1e5 t%lut%lud%d", t1->id, t2->id, i);
								/* Only check this if t2 is after t1 */
								fprintf(output, " - 1e5 t%luafter%lu;\n", t1->id, t2->id);
							}

							/* Or t1 is scheduled after t2 is.  */
							fprintf(output, "s%lu - s%lu >= -1e5 + 1e5 t%luafter%lu;\n", t1->id, t2->id, t1->id, t2->id);
						}
					}
				}
			}
		}


		for (t1 = tasks; t1; t1 = t1->next)
			for (t2 = t1->next; t2; t2 = t2->next)
				if (!ancestor(t1, t2) && !ancestor(t2, t1))
				{
					fprintf(output, "bin t%luafter%lu;\n", t1->id, t2->id);
					if (recordprio && t1->priority != t2->priority)
					{
						if (t1->priority > t2->priority)
						{
							if (t1->depsn > 1)
								for (i = 0; i < t1->depsn; i++)
									fprintf(output, "bin t%lut%lud%d;\n", t2->id, t1->id, i);
						}
						else
						{
							if (t2->depsn > 1)
								for (i = 0; i < t2->depsn; i++)
									fprintf(output, "bin t%lut%lud%d;\n", t1->id, t2->id, i);
						}
					}
				}

		for (t1 = tasks; t1; t1 = t1->next)
			for (w = 0; w < nw; w++)
				fprintf(output, "bin t%luw%d;\n", t1->id, w);
	}
	else
	{
		struct bound_task_pool *tp;
		nt = 0;
		for (tp = task_pools; tp; tp = tp->next)
			nt++;
		if (!nt)
			return;

		{
			double times[nw*nt];

			_starpu_get_tasks_times(nw, nt, times);

			fprintf(output, "/* StarPU upper bound linear programming problem, to be run in lp_solve. */\n\n");
			fprintf(output, "/* We want to minimize total execution time (ms) */\n");
			fprintf(output, "min: tmax;\n\n");

			fprintf(output, "/* Which is the maximum of all worker execution times (ms) */\n");
			for (w = 0; w < nw; w++)
			{
				char name[32];
				starpu_worker_get_name(w, name, sizeof(name));
				fprintf(output, "/* worker %s */\n0", name);
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if (!isnan(times[w*nt+t]))
						fprintf(output, "\t%+f * w%dt%dn", (float) times[w*nt+t], w, t);
				}
				fprintf(output, " <= tmax;\n");
			}
			fprintf(output, "\n");

			fprintf(output, "/* And we have to have computed exactly all tasks */\n");
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			{
				int got_one = 0;
				fprintf(output, "/* task %s key %x */\n0", _starpu_codelet_get_model_name(tp->cl), (unsigned) tp->footprint);
				for (w = 0; w < nw; w++)
				{
					if (isnan(times[w*nt+t]))
						_STARPU_MSG("Warning: task %s has no performance measurement for worker %d.\n", _starpu_codelet_get_model_name(tp->cl), w);
					else
					{
						got_one = 1;
						fprintf(output, "\t+w%dt%dn", w, t);
					}
				}
				fprintf(output, " = %lu;\n", tp->n);
				if (!got_one)
					_STARPU_MSG("Warning: task %s has no performance measurement for any worker, system will not be solvable!\n", _starpu_codelet_get_model_name(tp->cl));
				/* Show actual values */
				fprintf(output, "/*");
				for (w = 0; w < nw; w++)
					fprintf(output, "\t+%lu", tp->cl->per_worker_stats[w]);
				fprintf(output, "\t*/\n\n");
			}

			fprintf(output, "/* Optionally tell that tasks can not be divided */\n");
			fprintf(output, "/* int ");
			int first = 1;
			for (w = 0; w < nw; w++)
				for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				{
					if (!first)
						fprintf(output, ",");
					else
						first = 0;
					fprintf(output, "w%dt%dn", w, t);
				}
			fprintf(output, "; */\n");
		}
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/*
 * Print bound system in MPS output format
 */
void starpu_bound_print_mps(FILE *output)
{
	struct bound_task_pool * tp;
	int nt; /* Number of different kinds of tasks */
	int nw; /* Number of different workers */
	int t, w;

	if (recorddeps)
	{
		fprintf(output, "Not supported\n");
		return;
	}

	nw = starpu_worker_get_count();
	if (!nw)
		/* Make llvm happy about the VLA below */
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	nt = 0;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;
	if (!nt)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return;
	}

	{
		double times[nw*nt];

		_starpu_get_tasks_times(nw, nt, times);

		fprintf(output, "NAME           StarPU theoretical bound\n");

		fprintf(output, "*\nROWS\n");

		fprintf(output, "* We want to minimize total execution time (ms)\n");
		fprintf(output, " N  TMAX\n");

		fprintf(output, "* Which is the maximum of all worker execution times (ms)\n");
		for (w = 0; w < nw; w++)
		{
			char name[32];
			starpu_worker_get_name(w, name, sizeof(name));
			fprintf(output, "* worker %s\n", name);
			fprintf(output, " L  W%d\n", w);
		}

		fprintf(output, "*\n* And we have to have computed exactly all tasks\n*\n");
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			fprintf(output, "* task %s key %x\n", _starpu_codelet_get_model_name(tp->cl), (unsigned) tp->footprint);
			fprintf(output, " E  T%d\n", t);
		}

		fprintf(output, "*\nCOLUMNS\n*\n");

		fprintf(output, "*\n* Execution times and completion of all tasks\n*\n");
		for (w = 0; w < nw; w++)
			for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
				if (!isnan(times[w*nt+t]))
				{
					char name[23];
					snprintf(name, sizeof(name), "W%dT%d", w, t);
					fprintf(output,"    %-8s  W%-7d  %12f\n", name, w, times[w*nt+t]);
					fprintf(output,"    %-8s  T%-7d  %12d\n", name, t, 1);
				}

		fprintf(output, "*\n* Total execution time\n*\n");
		for (w = 0; w < nw; w++)
			fprintf(output,"    TMAX      W%-2d       %12d\n", w, -1);
		fprintf(output,"    TMAX      TMAX      %12d\n", 1);

		fprintf(output, "*\nRHS\n*\n");

		fprintf(output, "*\n* Total number of tasks\n*\n");
		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
			fprintf(output,"    NT%-2d      T%-7d  %12lu\n", t, t, tp->n);

		fprintf(output, "ENDATA\n");
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

/*
 * Solve bound system thanks to GNU Linear Programming Kit backend
 */
#ifdef STARPU_HAVE_GLPK_H
static glp_prob *_starpu_bound_glp_resolve(int integer)
{
	struct bound_task_pool * tp;
	int nt; /* Number of different kinds of tasks */
	int nw; /* Number of different workers */
	int t, w;
	glp_prob *lp;
	int ret;

	nw = starpu_worker_get_count();
	if (!nw)
		/* Make llvm happy about the VLA below */
		return NULL;
	nt = 0;
	for (tp = task_pools; tp; tp = tp->next)
		nt++;
	if (!nt)
		return NULL;

	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MIN);
	glp_set_obj_name(lp, "total execution time");

	{
		double times[nw*nt];
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
				if (!isnan(times[w*nt+t]))
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
				if (isnan(times[w*nt+t]))
					ar[n] = 1000000000.;
				else
					ar[n] = times[w*nt+t];
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
			snprintf(title, sizeof(title), "task %s key %x", _starpu_codelet_get_model_name(tp->cl), (unsigned) tp->footprint);
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

		STARPU_ASSERT(n == ne);

		glp_load_matrix(lp, ne-1, ia, ja, ar);
	}

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	ret = glp_simplex(lp, &parm);
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

	return lp;
}
#endif /* STARPU_HAVE_GLPK_H */

/* Print the computed bound as well as the optimized distribution of tasks */
void starpu_bound_print(FILE *output, int integer)
{
#ifdef STARPU_HAVE_GLPK_H
	if (recorddeps)
	{
		fprintf(output, "Not supported\n");
		return;
	}

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	glp_prob *lp = _starpu_bound_glp_resolve(integer);
	if (lp)
	{
		struct bound_task_pool * tp;
		int t, w;
		int nw; /* Number of different workers */
		double tmax;

		nw = starpu_worker_get_count();

		if (integer)
			tmax = glp_mip_obj_val(lp);
		else
			tmax = glp_get_obj_val(lp);

		fprintf(output, "Theoretical minimum execution time: %f ms\n", tmax);

		for (t = 0, tp = task_pools; tp; t++, tp = tp->next)
		{
			fprintf(output, "%s key %x\n", _starpu_codelet_get_model_name(tp->cl), (unsigned) tp->footprint);
			for (w = 0; w < nw; w++)
				if (integer)
					fprintf(output, "\tw%dt%dn %f", w, t, glp_mip_col_val(lp, colnum(w, t)));
				else
					fprintf(output, "\tw%dt%dn %f", w, t, glp_get_col_prim(lp, colnum(w, t)));
			fprintf(output, "\n");
		}

		glp_delete_prob(lp);
	}
	else
	{
		_STARPU_MSG("Simplex failed\n");
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
#else /* STARPU_HAVE_GLPK_H */
	(void) integer;
	fprintf(output, "Please rebuild StarPU with glpk installed.\n");
#endif /* STARPU_HAVE_GLPK_H */
}

/* Compute and return the bound */
void starpu_bound_compute(double *res, double *integer_res, int integer)
{
#ifdef STARPU_HAVE_GLPK_H
	double ret;

	if (recorddeps)
	{
		*res = 0.;
		return;
	}

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	glp_prob *lp = _starpu_bound_glp_resolve(integer);
	if (lp)
	{
		ret = glp_get_obj_val(lp);
		if (integer)
			*integer_res = glp_mip_obj_val(lp);
		glp_delete_prob(lp);
	}
	else
		ret = 0.;
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	*res = ret;
#else /* STARPU_HAVE_GLPK_H */
	(void) integer_res;
	(void) integer;
	*res = 0.;
#endif /* STARPU_HAVE_GLPK_H */
}
