/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2017  Université de Bordeaux
 * Copyright (C) 2017  Erwan Leria
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
#include <common/utils.h>

/*
 sched.rec files look like this:

 Tag: 1234
 Priority: 12
 ExecuteOnSpecificWorker: 1
 Workers: 0 1 2
 DependsOn: 1235

 Prefetch: 1234
 DependsOn: 1233
 */


#define CPY(src, dst, n) memcpy(dst, src, n * sizeof(*dst))

static unsigned long submitorder; /* Also use as prefetchtag */
static int priority;
static int eosw;
static int workerorder;
static int memnode;
static unsigned workers[STARPU_NMAXWORKERS];
static unsigned nworkers;
static unsigned dependson[STARPU_NMAXBUFS];
static unsigned ndependson;
static unsigned params[STARPU_NMAXBUFS];
static unsigned nparams;

static enum sched_type {
	NormalTask,
	PrefetchTask,
} sched_type;

static struct starpu_codelet cl_prefetch = {
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
	int workerorder;
	unsigned workers[STARPU_NMAXWORKERS];
	unsigned nworkers;

	/* For prefetch tasks */
	unsigned params[STARPU_NMAXBUFS];
	unsigned nparams;
	struct starpu_task *pref_task; /* Actual prefetch task */
} *mangled_tasks, *prefetch_tasks;

static struct dep {
	UT_hash_handle hh;
	unsigned long submitorder;
	struct task *task;
	unsigned i;
} *dependences;

static void reset(void) {
	submitorder = 0;
	priority = 0;
	eosw = -1;
	nworkers = 0;
	ndependson = 0;
	sched_type = NormalTask;
	nparams = 0;
	memnode = -1;
	workerorder = -1;
}

/* TODO : respecter l'ordre de soumission des tâches SubmitOrder */

// TODO: call SchedRecInit


static void checkField(char * s)
{
	/* Record various information */
#define TEST(field) (!strncmp(s, field": ", strlen(field) + 2))

	if (TEST("SubmitOrder"))
	{
		s = s + sizeof("SubmitOrder: ");
		submitorder = strtol(s, NULL, 16);
	}

	else if (TEST("Priority"))
	{
		s = s + sizeof("Priority: ");
		priority = strtol(s, NULL, 10);
	}

	else if (TEST("ExecuteOnSpecificWorker"))
	{
		eosw = strtol(s, NULL, 10);
	}

	else if (TEST("Workers"))
	{
		s = s + sizeof("Workers: ");
		char * delim = " ";
		char * token = strtok(s, delim);
		int i = 0;
		 
		while (token != NULL)
		{
			int k = strtol(token, NULL, 10);
			workers[k/sizeof(*workers)] |= (1 << (k%(sizeof(*workers))));
			i++;
		}

		nworkers = i;
	}

	else if (TEST("DependsOn"))
	{
		/* NOTE : dependsons (in the sched.rec)  should be the submit orders of the dependences, 
		   otherwise it can occur an undefined behaviour
		   (contrary to the tasks.rec where dependences are jobids */
		unsigned i = 0;
		char * delim = " ";
		char * token = strtok(s+sizeof("DependsOn: "), delim);
		
		while (token != NULL)
		{
			dependson[i] = strtol(token, NULL, 10);
			i++;
		}
		ndependson = i;
	}

	else if (TEST("Prefetch"))
	{
		s = s + sizeof("Prefetch: ");
		submitorder = strtol(s, NULL, 10);
		sched_type = PrefetchTask;
	}

	else if (TEST("Parameters"))
	{
		s = s + sizeof("Parameters: ");
		char * delim = " ";
		char * token = strtok(s, delim);
		int i = 0;
		 
		while (token != NULL)
		{
			params[i] = strtol(token, NULL, 10);
			i++;
		}
		nparams = i;
	}

	else if (TEST("MemoryNode"))
	{
		s = s + sizeof("MemoryNode: ");
		memnode = strtol(s, NULL, 10);
	}
	
	else if (TEST("Workerorder"))
	{
		s = s + sizeof("Workerorder: ");
		workerorder = strtol(s, NULL, 10);
	}
}


void schedRecInit(const char * filename)
{
	FILE * f = fopen(filename, "r");
	
	if(f == NULL)
	{
		return;
	}

	size_t lnsize = 128;
	char * s = malloc(sizeof(*s) * lnsize);
	
	reset();

	while(!feof(f))
	{
		char *ln;

		/* Get the line */
		if (!fgets(s, lnsize, f))
		{
			return;
		}
		while (!(ln = strchr(s, '\n')))
		{
			_STARPU_REALLOC(s, lnsize * 2);
			if (!fgets(s + lnsize-1, lnsize+1, f))
			{
				return;
			}
			lnsize *= 2;
		}

		if (ln == s)
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
			for (i = 0; i < ndependson; i++) {
				struct dep *dep;
				struct starpu_task *starpu_task;
				_STARPU_MALLOC(dep, sizeof(*dep));
				dep->task = task;
				dep->i = i;
				dep->submitorder = task->dependson[i];
				HASH_ADD(hh, dependences, submitorder, sizeof(submitorder), dep);

				/* Create the intermediate task */
				starpu_task = dep->task->depends_tasks[i] = starpu_task_create();
				starpu_task->cl = NULL;
				starpu_task->destroy = 0;
				starpu_task->no_submitorder = 1;
			}
			break;

			switch (sched_type)
			{
			case NormalTask:
				/* A new task to mangle, record what needs to be done */
				task->eosw = eosw;
				task->workerorder = workerorder;
				CPY(workers, task->workers, nworkers);
				task->nworkers = nworkers;
				STARPU_ASSERT(nparams == 0);
				break;

			case PrefetchTask:
				STARPU_ASSERT(eosw == -1);
				STARPU_ASSERT(workerorder == -1);
				STARPU_ASSERT(nworkers == 0);
				CPY(params, task->params, nparams);
				task->nparams = nparams;
				break;
			}

			HASH_ADD(hh, mangled_tasks, submitorder, sizeof(submitorder), task);

			reset();
		}
		else checkField(s);
	}
}

static void do_prefetch(void *arg)
{
	unsigned node = (uintptr_t) arg;
	starpu_data_idle_prefetch_on_node(starpu_task_get_current()->handles[0], node, 1);
}

void applySchedRec(struct starpu_task * starpu_task, unsigned long submit_order)
{
	struct task *task;
	struct dep *dep;
	int ret;

	HASH_FIND(hh, dependences, &submit_order, sizeof(submit_order), dep);
	if (dep)
	{
		/* Some task will depend on this one, make the dependency */
		starpu_task_declare_deps_array(dep->task->depends_tasks[dep->i], 1, &starpu_task);
		ret = starpu_task_submit(dep->task->depends_tasks[dep->i]);
		STARPU_ASSERT(ret == 0);
	}

	HASH_FIND(hh, prefetch_tasks, &submit_order, sizeof(submit_order), task);
	if (task) {
		/* We want to submit a prefetch for this task */
		struct starpu_task *pref_task;
		pref_task = task->pref_task = starpu_task_create();
		pref_task->cl = &cl_prefetch;
		pref_task->destroy = 1;
		pref_task->no_submitorder = 1;
		pref_task->callback_arg = (void*)(uintptr_t) task->memnode;
		pref_task->callback_func = do_prefetch;

		/* TODO: more params */
		pref_task->handles[0] = starpu_task->handles[0];
		/* Make it depend on intermediate tasks */
		if (task->ndependson)
			starpu_task_declare_deps_array(pref_task, task->ndependson, task->depends_tasks);
		ret = starpu_task_submit(pref_task);
		STARPU_ASSERT(ret == 0);
	}

	HASH_FIND(hh, mangled_tasks, &submit_order, sizeof(submit_order), task);
       	if (task == NULL)
		/* Nothing to do for this */
		return;

	starpu_task->workerorder = task->workerorder;
	starpu_task->priority = task->priority;
	starpu_task->workerids_len = task->nworkers;
	_STARPU_MALLOC(starpu_task->workerids, task->nworkers * sizeof(*starpu_task->workerids));
	CPY(task->workers, starpu_task->workerids, task->nworkers);

	if (task->ndependson)
		starpu_task_declare_deps_array(starpu_task, task->ndependson, task->depends_tasks);

	/* And now, let it go!  */
}
