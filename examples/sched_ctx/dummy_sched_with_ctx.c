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
 * This is an example of an application-defined scheduler run inside a
 * scheduling context.
 * This is a mere eager scheduler with a centralized list of tasks to schedule:
 * when a task becomes ready (push) it is put on the list. When a device
 * becomes ready (pop), a task is taken from the list.
 */
#include <starpu.h>
#include <starpu_scheduler.h>

#ifdef STARPU_QUICK_CHECK
#define NTASKS	320
#elif !defined(STARPU_LONG_CHECK)
#define NTASKS	3200
#else
#define NTASKS	32000
#endif
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct dummy_sched_data
{
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

static void init_dummy_sched(unsigned sched_ctx_id)
{
	struct dummy_sched_data *data = (struct dummy_sched_data*)malloc(sizeof(struct dummy_sched_data));


	/* Create a linked-list of tasks and a condition variable to protect it */
	starpu_task_list_init(&data->sched_list);

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);

	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	FPRINTF(stderr, "Initialising Dummy scheduler\n");
}

static void deinit_dummy_sched(unsigned sched_ctx_id)
{
	struct dummy_sched_data *data = (struct dummy_sched_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_ASSERT(starpu_task_list_empty(&data->sched_list));

	STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);

	free(data);

	FPRINTF(stderr, "Destroying Dummy scheduler\n");
}

static int push_task_dummy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct dummy_sched_data *data = (struct dummy_sched_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	/* NB: In this simplistic strategy, we assume that the context in which
	   we push task has at least one worker*/


	/* lock all workers when pushing tasks on a list where all
	   of them would pop for tasks */
        STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	starpu_task_list_push_front(&data->sched_list, task);

	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);


        /*if there are no tasks block */
        /* wake people waiting for a task */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

        struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
        {
		unsigned worker;
                worker = workers->get_next(workers, &it);
		starpu_pthread_mutex_t *sched_mutex;
                starpu_pthread_cond_t *sched_cond;
                starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
		STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
                STARPU_PTHREAD_COND_SIGNAL(sched_cond);
                STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
        }

	return 0;
}

/* The mutex associated to the calling worker is already taken by StarPU */
static struct starpu_task *pop_task_dummy(unsigned sched_ctx_id)
{
	/* NB: In this simplistic strategy, we assume that all workers are able
	 * to execute all tasks, otherwise, it would have been necessary to go
	 * through the entire list until we find a task that is executable from
	 * the calling worker. So we just take the head of the list and give it
	 * to the worker. */
	struct dummy_sched_data *data = (struct dummy_sched_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
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

static struct starpu_sched_policy dummy_sched_policy =
{
	.init_sched = init_dummy_sched,
	.add_workers = NULL,
	.remove_workers = NULL,
	.deinit_sched = deinit_dummy_sched,
	.push_task = push_task_dummy,
	.pop_task = pop_task_dummy,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "dummy",
	.policy_description = "dummy scheduling strategy",
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
	.model = NULL,
	.nbuffers = 0,
	.name = "dummy",
};


int main(void)
{
	int ntasks = NTASKS;
	int ret;
/* 	struct starpu_conf conf; */

/* 	starpu_conf_init(&conf); */
/* 	conf.sched_policy = &dummy_sched_policy, */
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned sched_ctx = starpu_sched_ctx_create(NULL, -1, "dummy", STARPU_SCHED_CTX_POLICY_STRUCT, &dummy_sched_policy, 0);
#ifdef STARPU_QUICK_CHECK
	ntasks /= 100;
#endif

	starpu_sched_ctx_set_context(&sched_ctx);
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
