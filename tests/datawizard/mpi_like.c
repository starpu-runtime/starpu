/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <errno.h>
#include <common/thread.h>
#include "../helper.h"
#include "../variable/increment.h"

/*
 * Mimic the behavior of libstarpumpi, tested by a ring of threads which
 * increment the same variable one after the other.
 * This is the synchronous version: the threads wait for completion of each
 * step before continuing.
 */

#define NTHREADS	4
#define NITER		2

//static starpu_pthread_cond_t cond;
//static starpu_pthread_mutex_t mutex;

struct thread_data
{
	unsigned index;
	unsigned val;
	starpu_data_handle_t handle;
	starpu_pthread_t thread;

	starpu_pthread_cond_t recv_cond;
	starpu_pthread_mutex_t recv_mutex;
	unsigned recv_flag; // set when a message is received
	unsigned recv_buf;
	struct thread_data *neighbour;
};

static struct thread_data problem_data[NTHREADS];

/* We implement some ring transfer, every thread will try to receive a piece of
 * data from its neighbour and increment it before transmitting it to its
 * successor. */

static void increment_handle(struct thread_data *thread_data)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &increment_cl;

	task->handles[0] = thread_data->handle;

	task->cl_arg = thread_data;
	task->cl_arg_size = sizeof(thread_data);

	task->destroy = 1;
	task->detach = 0;

	int ret = starpu_task_submit(task);
	if (ret == -ENODEV)
		exit(STARPU_TEST_SKIPPED);
	STARPU_ASSERT(!ret);

	ret = starpu_task_wait(task);
	STARPU_ASSERT(!ret);
}

static void recv_handle(struct thread_data *thread_data)
{
	starpu_data_acquire(thread_data->handle, STARPU_W);
	STARPU_PTHREAD_MUTEX_LOCK(&thread_data->recv_mutex);

	/* We wait for the previous thread to notify that the data is available */
	while (!thread_data->recv_flag)
		STARPU_PTHREAD_COND_WAIT(&thread_data->recv_cond, &thread_data->recv_mutex);

	/* We overwrite thread's data with the received value */
	thread_data->val = thread_data->recv_buf;

	/* Notify that we read the value */
	thread_data->recv_flag = 0;
	STARPU_PTHREAD_COND_SIGNAL(&thread_data->recv_cond);

//	FPRINTF(stderr, "Thread %d received value %d from thread %d\n", thread_data->index, thread_data->val, (thread_data->index - 1)%NTHREADS);

	STARPU_PTHREAD_MUTEX_UNLOCK(&thread_data->recv_mutex);
	starpu_data_release(thread_data->handle);
}

static void send_handle(struct thread_data *thread_data)
{
	struct thread_data *neighbour_data = thread_data->neighbour;

	starpu_data_acquire(thread_data->handle, STARPU_R);

//	FPRINTF(stderr, "Thread %d sends value %d to thread %d\n", thread_data->index, thread_data->val, neighbour_data->index);
	/* send the message */
	STARPU_PTHREAD_MUTEX_LOCK(&neighbour_data->recv_mutex);
	neighbour_data->recv_buf = thread_data->val;
	neighbour_data->recv_flag = 1;
	STARPU_PTHREAD_COND_SIGNAL(&neighbour_data->recv_cond);

	/* wait until it's received (ie. neighbour's recv_flag is set back to 0) */
	while (neighbour_data->recv_flag)
		STARPU_PTHREAD_COND_WAIT(&neighbour_data->recv_cond, &neighbour_data->recv_mutex);

	STARPU_PTHREAD_MUTEX_UNLOCK(&neighbour_data->recv_mutex);

	starpu_data_release(thread_data->handle);
}

static void *thread_func(void *arg)
{
	unsigned iter;
	struct thread_data *thread_data = (struct thread_data *) arg;
	unsigned index = thread_data->index;

	starpu_variable_data_register(&thread_data->handle, STARPU_MAIN_RAM, (uintptr_t)&thread_data->val, sizeof(unsigned));

	for (iter = 0; iter < NITER; iter++)
	{
		/* The first thread initiates the first transfer */
		if (!((index == 0) && (iter == 0)))
		{
			recv_handle(thread_data);
		}

		increment_handle(thread_data);

		if (!((index == (NTHREADS - 1)) && (iter == (NITER - 1))))
		{
			send_handle(thread_data);
		}
	}

	return NULL;
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	increment_load_opencl();

	unsigned t;
	for (t = 0; t < NTHREADS; t++)
	{
		problem_data[t].index = t;
		problem_data[t].val = 0;
		STARPU_PTHREAD_COND_INIT(&problem_data[t].recv_cond, NULL);
		STARPU_PTHREAD_MUTEX_INIT(&problem_data[t].recv_mutex, NULL);
		problem_data[t].recv_flag = 0;
		problem_data[t].neighbour = &problem_data[(t+1)%NTHREADS];
	}

	for (t = 0; t < NTHREADS; t++)
	{
		STARPU_PTHREAD_CREATE(&problem_data[t].thread, NULL, thread_func, &problem_data[t]);
	}

	for (t = 0; t < NTHREADS; t++)
	{
		void *retval;
		STARPU_PTHREAD_JOIN(problem_data[t].thread, &retval);
		STARPU_ASSERT(retval == NULL);
	}

	/* We check that the value in the "last" thread is valid */
	starpu_data_handle_t last_handle = problem_data[NTHREADS - 1].handle;
	starpu_data_acquire(last_handle, STARPU_R);
	starpu_data_release(last_handle);

	for (t = 0; t < NTHREADS; t++)
	{
		starpu_data_unregister(problem_data[t].handle);
	}

	increment_unload_opencl();
	starpu_shutdown();

	ret = EXIT_SUCCESS;
	if (problem_data[NTHREADS - 1].val != (NTHREADS * NITER))
	{
		FPRINTF(stderr, "Final value : %u should be %d\n", problem_data[NTHREADS - 1].val, (NTHREADS * NITER));
		ret = EXIT_FAILURE;
	}
	else
		FPRINTF(stderr, "Final value : %u (niter %u nthread %u)\n", problem_data[NTHREADS - 1].val, NITER, NTHREADS);

	return ret;
}
