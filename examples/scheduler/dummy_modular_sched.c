/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This is an example of an application-defined scheduler.
 * This is a mere eager scheduler with a centralized list of tasks to schedule:
 * when a task becomes ready (push) it is put on the list. When a device
 * becomes ready (pop), a task is taken from the list.
 */
#include <starpu.h>
#include <starpu_scheduler.h>
#include <starpu_sched_component.h>

#ifdef STARPU_QUICK_CHECK
#define NTASKS	320
#elif !defined(STARPU_LONG_CHECK)
#define NTASKS	3200
#else
#define NTASKS	32000
#endif
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct dummy_sched_params
{
	int verbose;
};

struct dummy_sched_data
{
	int verbose;
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

static void dummy_deinit_data(struct starpu_sched_component * component)
{
	struct dummy_sched_data *data = component->data;

	STARPU_ASSERT(starpu_task_list_empty(&data->sched_list));

	if (data->verbose)
		fprintf(stderr, "Destroying Dummy scheduler\n");

	STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);
	free(data);
}

static int dummy_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct dummy_sched_data *data = component->data;
	if (data->verbose)
		fprintf(stderr, "pushing task %p\n", task);

	/* NB: In this simplistic strategy, we assume that the context in which
	   we push task has at least one worker*/

	/* lock all workers when pushing tasks on a list where all
	   of them would pop for tasks */
        STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	starpu_task_list_push_front(&data->sched_list, task);

	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	/* Tell below that they can now pull */
	component->can_pull(component);

	return 0;
}

static struct starpu_task *dummy_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct dummy_sched_data *data = component->data;
	if (data->verbose)
		fprintf(stderr, "%p pulling for a task\n", to);

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_task_list_empty(&data->sched_list))
		return NULL;
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	struct starpu_task *task = NULL;
	if (!starpu_task_list_empty(&data->sched_list))
		task = starpu_task_list_pop_back(&data->sched_list);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	return task;
}

static int dummy_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct dummy_sched_data *data = component->data;
	int didwork = 0;

	if (data->verbose)
		fprintf(stderr, "%p tells me I can push to him\n", to);

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		if (data->verbose)
			fprintf(stderr, "oops, %p couldn't take our task\n", to);
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_task_list_push_back(&data->sched_list, task);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	}
	else
	{
		if (data->verbose)
		{
			if (didwork)
				fprintf(stderr, "pushed some tasks to %p\n", to);
			else
				fprintf(stderr, "I didn't have anything for %p\n", to);
		}
	}

	/* There is room now */
	return didwork || starpu_sched_component_can_push(component, to);
}

static int dummy_can_pull(struct starpu_sched_component * component)
{
	struct dummy_sched_data *data = component->data;

	if (data->verbose)
		fprintf(stderr,"telling below they can pull\n");

	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *dummy_create(struct starpu_sched_tree *tree, struct dummy_sched_params *params)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dummy");
	struct dummy_sched_data *data = malloc(sizeof(*data));

	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	/* Create a linked-list of tasks and a condition variable to protect it */
	starpu_task_list_init(&data->sched_list);
	data->verbose = params->verbose;

	component->data = data;
	component->push_task = dummy_push_task;
	component->pull_task = dummy_pull_task;
	component->can_push = dummy_can_push;
	component->can_pull = dummy_can_pull;
	component->deinit_data = dummy_deinit_data;

	return component;
}

static void init_dummy_sched(unsigned sched_ctx_id)
{
	FPRINTF(stderr, "Initialising Dummy scheduler\n");

	struct dummy_sched_params params =
	{
		.verbose = 0,
	};

	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) dummy_create, &params,
			STARPU_SCHED_SIMPLE_DECIDE_WORKERS |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO,
			sched_ctx_id);
}

static void deinit_dummy_sched(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(t);
}

static struct starpu_sched_policy dummy_sched_policy =
{
	.init_sched = init_dummy_sched,
	.deinit_sched = deinit_dummy_sched,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "dummy",
	.policy_description = "dummy modular scheduling strategy",
	.worker_type = STARPU_WORKER_LIST,
};

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.cuda_funcs = {dummy_func},
        .opencl_funcs = {dummy_func},
	.model = &starpu_perfmodel_nop,
	.nbuffers = 0,
	.name = "dummy",
};


int main(void)
{
	int ntasks = NTASKS;
	int ret;
	struct starpu_conf conf;

	char *sched = getenv("STARPU_SCHED");
	if (sched && sched[0])
		/* Testing a specific scheduler, no need to run this */
		return 77;

	starpu_conf_init(&conf);
	conf.sched_policy = &dummy_sched_policy,
	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_QUICK_CHECK
	ntasks /= 100;
#endif

	int i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &dummy_codelet;
		task->cl_arg = NULL;

		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();

	starpu_shutdown();

	return 0;
}
