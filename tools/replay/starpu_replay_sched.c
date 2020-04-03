/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2017       Erwan Leria
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
 * This reads a sched.rec file and mangles submitted tasks according to the hint
 * from that file.
 */

#include <starpu.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <common/uthash.h>
#include <common/list.h>
#include <common/utils.h>
#include <limits.h>

/*
 sched.rec files look like this:

 SubmitOrder: 1234
 Priority: 12
 SpecificWorker: 1
 Workers: 0 1 2
 DependsOn: 1235

 Prefetch: 1234
 DependsOn: 1233
 MemoryNode: 1
 Parameters: 1
 */


#define CPY(src, dst, n) memcpy(dst, src, n * sizeof(*dst))

#if 0
#define debug(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#else
#define debug(fmt, ...) (void)0
#endif

static unsigned long submitorder; /* Also use as prefetchtag */
static int priority;
static int eosw;
static unsigned workerorder;
static int memnode;
/* FIXME: MAXs */
static uint32_t workers[STARPU_NMAXWORKERS/32];
static unsigned nworkers;
static unsigned dependson[STARPU_NMAXBUFS];
static unsigned ndependson;
static unsigned params[STARPU_NMAXBUFS];
static unsigned nparams;

static enum sched_type
{
	NormalTask,
	PrefetchTask,
} sched_type;

static struct starpu_codelet cl_prefetch =
{
        .where = STARPU_NOWHERE,
        .nbuffers = 1,
        .modes = { STARPU_R },
};

static struct task
{
	UT_hash_handle hh;

	unsigned long submitorder;
	int priority;
	int memnode;
	unsigned dependson[STARPU_NMAXBUFS];
	unsigned ndependson;
	struct starpu_task *depends_tasks[STARPU_NMAXBUFS];

	/* For real tasks */
	int eosw;
	unsigned workerorder;
	uint32_t workers[STARPU_NMAXWORKERS/32];
	unsigned nworkers;

	/* For prefetch tasks */
	unsigned params[STARPU_NMAXBUFS];
	unsigned nparams;
	struct starpu_task *pref_task; /* Actual prefetch task */
} *mangled_tasks, *prefetch_tasks;

LIST_TYPE(dep,
	struct task *task;
	unsigned i;
);

struct deps
{
	UT_hash_handle hh;
	unsigned long submitorder;
	struct dep_list list;
} *dependencies = NULL;

static void reset(void)
{
	submitorder = 0;
	priority = INT_MIN;
	eosw = -1;
	memset(&workers, 0, sizeof(workers));
	nworkers = 0;
	ndependson = 0;
	sched_type = NormalTask;
	nparams = 0;
	memnode = -1;
	workerorder = 0;
}

/* TODO : respecter l'ordre de soumission des tâches SubmitOrder */


static void checkField(char * s)
{
	/* Record various information */
#define TEST(field) (!strncmp(s, field": ", strlen(field) + 2))

	if (TEST("SubmitOrder"))
	{
		s = s + strlen("SubmitOrder: ");
		submitorder = strtol(s, NULL, 10);
	}

	else if (TEST("Priority"))
	{
		s = s + strlen("Priority: ");
		priority = strtol(s, NULL, 10);
	}

	else if (TEST("SpecificWorker"))
	{
		s = s + strlen("SpecificWorker: ");
		eosw = strtol(s, NULL, 10);
	}

	else if (TEST("Workers"))
	{
		s = s + strlen("Workers: ");
		char * delim = " ";
		char * token = strtok(s, delim);
		int i = 0;

		while (token != NULL)
		{
			int k = strtol(token, NULL, 10);
			STARPU_ASSERT_MSG(k < STARPU_NMAXWORKERS, "%d is bigger than maximum %d\n", k, STARPU_NMAXWORKERS);
			workers[k/(sizeof(*workers)*8)] |= (1 << (k%(sizeof(*workers)*8)));
			i++;
			token = strtok(NULL, delim);
		}

		nworkers = i;
	}

	else if (TEST("DependsOn"))
	{
		/* NOTE : dependsons (in the sched.rec)  should be the submit orders of the dependencies,
		   otherwise it can occur an undefined behaviour
		   (contrary to the tasks.rec where dependencies are jobids */
		unsigned i = 0;
		char * delim = " ";
		char * token = strtok(s+strlen("DependsOn: "), delim);

		while (token != NULL)
		{
			dependson[i] = strtol(token, NULL, 10);
			i++;
			token = strtok(NULL, delim);
		}
		ndependson = i;
	}

	else if (TEST("Prefetch"))
	{
		s = s + strlen("Prefetch: ");
		submitorder = strtol(s, NULL, 10);
		sched_type = PrefetchTask;
	}

	else if (TEST("Parameters"))
	{
		s = s + strlen("Parameters: ");
		char * delim = " ";
		char * token = strtok(s, delim);
		int i = 0;

		while (token != NULL)
		{
			params[i] = strtol(token, NULL, 10);
			i++;
			token = strtok(NULL, delim);
		}
		nparams = i;
	}

	else if (TEST("MemoryNode"))
	{
		s = s + strlen("MemoryNode: ");
		memnode = strtol(s, NULL, 10);
	}

	else if (TEST("Workerorder"))
	{
		s = s + strlen("Workerorder: ");
		workerorder = strtol(s, NULL, 10);
	}
}


void schedRecInit(const char * filename)
{
	FILE * f = fopen(filename, "r");

	if(f == NULL)
	{
		fprintf(stderr,"unable to open file %s: %s\n", filename, strerror(errno));
		return;
	}

	size_t lnsize = 128;
	char *s;
	_STARPU_MALLOC(s, sizeof(*s) * lnsize);
	int eof = 0;

	reset();

	while(!eof && !feof(f))
	{
		char *ln;

		/* Get the line */
		if (!fgets(s, lnsize, f))
		{
			eof = 1;
		}
		while (!(ln = strchr(s, '\n')))
		{
			_STARPU_REALLOC(s, lnsize * 2);
			if (!fgets(s + lnsize-1, lnsize+1, f))
			{
				eof = 1;
				break;
			}
			lnsize *= 2;
		}

		if ((ln == s || eof) && submitorder)
		{
			/* Empty line, doit */
			struct task * task;
			unsigned i;

			_STARPU_MALLOC(task, sizeof(*task));
			task->submitorder = submitorder;
			task->priority = priority;
			task->memnode = memnode;
			CPY(dependson, task->dependson, ndependson);
			task->ndependson = ndependson;

			/* Also record submitorder of tasks that this one will need to depend on */
			for (i = 0; i < ndependson; i++)
			{
				struct dep *dep;
				struct starpu_task *starpu_task;
				_STARPU_MALLOC(dep, sizeof(*dep));
				dep->task = task;
				dep->i = i;

				struct deps *deps;
				HASH_FIND(hh, dependencies, &task->dependson[i], sizeof(submitorder), deps);
				if (!deps)
				{
					/* No task depends on this one yet, add a cell for it */
					_STARPU_MALLOC(deps, sizeof(*deps));
					dep_list_init(&deps->list);
					deps->submitorder = task->dependson[i];
					HASH_ADD(hh, dependencies, submitorder, sizeof(submitorder), deps);
				}
				dep_list_push_back(&deps->list, dep);

				/* Create the intermediate task */
				starpu_task = dep->task->depends_tasks[i] = starpu_task_create();
				starpu_task->cl = NULL;
				starpu_task->destroy = 0;
				starpu_task->no_submitorder = 1;
			}

			switch (sched_type)
			{
			case NormalTask:
				/* A new task to mangle, record what needs to be done */
				task->eosw = eosw;
				task->workerorder = workerorder;
				CPY(workers, task->workers, STARPU_NMAXWORKERS/32);
				task->nworkers = nworkers;
				STARPU_ASSERT(nparams == 0);

				debug("adding mangled task %lu\n", submitorder);
				HASH_ADD(hh, mangled_tasks, submitorder, sizeof(submitorder), task);
				break;

			case PrefetchTask:
				STARPU_ASSERT(memnode >= 0);
				STARPU_ASSERT(eosw == -1);
				STARPU_ASSERT(workerorder == 0);
				STARPU_ASSERT(nworkers == 0);
				CPY(params, task->params, nparams);
				task->nparams = nparams;
				/* TODO: more params */
				STARPU_ASSERT_MSG(nparams == 1, "only supports one parameter at a time");

				debug("adding prefetch task for %lu\n", submitorder);
				HASH_ADD(hh, prefetch_tasks, submitorder, sizeof(submitorder), task);
				break;
			default:
				STARPU_ASSERT(0);
				break;
			}

			reset();
		}
		else checkField(s);
	}

	fclose(f);
}

static void do_prefetch(void *arg)
{
	unsigned node = (uintptr_t) arg;
	starpu_data_idle_prefetch_on_node(starpu_task_get_current()->handles[0], node, 1);
}

void applySchedRec(struct starpu_task *starpu_task, unsigned long submit_order)
{
	struct task *task;
	struct deps *deps;
	int ret;

	HASH_FIND(hh, dependencies, &submit_order, sizeof(submit_order), deps);
	if (deps)
	{
		struct dep *dep;
		for (dep  = dep_list_begin(&deps->list);
		     dep != dep_list_end(&deps->list);
		     dep =  dep_list_next(dep))
		{
			debug("task %lu is %d-th dep for %lu\n", submit_order, dep->i, dep->task->submitorder);
			/* Some task will depend on this one, make the dependency */
			starpu_task_declare_deps_array(dep->task->depends_tasks[dep->i], 1, &starpu_task);
			ret = starpu_task_submit(dep->task->depends_tasks[dep->i]);
			STARPU_ASSERT(ret == 0);
		}
	}

	HASH_FIND(hh, prefetch_tasks, &submit_order, sizeof(submit_order), task);
	if (task)
	{
		/* We want to submit a prefetch for this task */
		debug("task %lu has a prefetch for parameter %d to node %d\n", submit_order, task->params[0], task->memnode);
		struct starpu_task *pref_task;
		pref_task = task->pref_task = starpu_task_create();
		pref_task->cl = &cl_prefetch;
		pref_task->destroy = 1;
		pref_task->no_submitorder = 1;
		pref_task->callback_arg = (void*)(uintptr_t) task->memnode;
		pref_task->callback_func = do_prefetch;

		/* TODO: more params */
		pref_task->handles[0] = starpu_task->handles[task->params[0]];
		/* Make it depend on intermediate tasks */
		if (task->ndependson)
		{
			debug("%u dependencies\n", task->ndependson);
			starpu_task_declare_deps_array(pref_task, task->ndependson, task->depends_tasks);
		}
		ret = starpu_task_submit(pref_task);
		STARPU_ASSERT(ret == 0);
	}

	HASH_FIND(hh, mangled_tasks, &submit_order, sizeof(submit_order), task);
       	if (task == NULL)
		/* Nothing to do for this */
		return;

	debug("mangling task %lu\n", submit_order);
	if (task->eosw >= 0)
	{
		debug("execute on a specific worker %d\n", task->eosw);
		starpu_task->workerid = task->eosw;
		starpu_task->execute_on_a_specific_worker = 1;
	}
	if (task->workerorder > 0)
	{
		debug("workerorder %d\n", task->workerorder);
		starpu_task->workerorder = task->workerorder;
	}
	if (task->priority != INT_MIN)
	{
		debug("priority %d\n", task->priority);
		starpu_task->priority = task->priority;
	}
	if (task->nworkers)
	{
		debug("%u workers %x\n", task->nworkers, task->workers[0]);
		starpu_task->workerids_len = sizeof(task->workers) / sizeof(task->workers[0]);
		_STARPU_MALLOC(starpu_task->workerids, task->nworkers * sizeof(*starpu_task->workerids));
		CPY(task->workers, starpu_task->workerids, STARPU_NMAXWORKERS/32);
	}

	if (task->ndependson)
	{
		debug("%u dependencies\n", task->ndependson);
		starpu_task_declare_deps_array(starpu_task, task->ndependson, task->depends_tasks);
	}

	/* And now, let it go!  */
}
