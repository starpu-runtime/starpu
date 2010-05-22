/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <starpu.h>
#include <pthread.h>

#define NTHREADS	4
#define NITER		128

static pthread_cond_t cond;
static pthread_mutex_t mutex;

struct thread_data {
	unsigned index;
	unsigned val;
	starpu_data_handle handle;
	pthread_t thread;

	pthread_cond_t recv_cond;
	pthread_mutex_t recv_mutex;
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
	starpu_data_sync_with_mem(thread_data->handle, STARPU_RW);
	thread_data->val++;
	starpu_data_release_from_mem(thread_data->handle);
}

static void recv_handle(struct thread_data *thread_data)
{
	starpu_data_sync_with_mem(thread_data->handle, STARPU_W);
	pthread_mutex_lock(&thread_data->recv_mutex);

	/* We wait for the previous thread to notify that the data is available */
	while (!thread_data->recv_flag)
		pthread_cond_wait(&thread_data->recv_cond, &thread_data->recv_mutex);

	/* We overwrite thread's data with the received value */
	thread_data->val = thread_data->recv_buf;

	/* Notify that we read the value */
	thread_data->recv_flag = 0;
	pthread_cond_signal(&thread_data->recv_cond);

//	fprintf(stderr, "Thread %d received value %d from thread %d\n", thread_data->index, thread_data->val, (thread_data->index - 1)%NTHREADS);

	pthread_mutex_unlock(&thread_data->recv_mutex);
	starpu_data_release_from_mem(thread_data->handle);
}

static void send_handle(struct thread_data *thread_data)
{
	struct thread_data *neighbour_data = thread_data->neighbour;

	starpu_data_sync_with_mem(thread_data->handle, STARPU_R);

//	fprintf(stderr, "Thread %d sends value %d to thread %d\n", thread_data->index, thread_data->val, neighbour_data->index);
	/* send the message */
	pthread_mutex_lock(&neighbour_data->recv_mutex);
	neighbour_data->recv_buf = thread_data->val;
	neighbour_data->recv_flag = 1;
	pthread_cond_signal(&neighbour_data->recv_cond);

	/* wait until it's received (ie. neighbour's recv_flag is set back to 0) */
	while (neighbour_data->recv_flag)
		pthread_cond_wait(&neighbour_data->recv_cond, &neighbour_data->recv_mutex);
	
	pthread_mutex_unlock(&neighbour_data->recv_mutex);

	starpu_data_release_from_mem(thread_data->handle);
}

static void *thread_func(void *arg)
{
	unsigned iter;
	struct thread_data *thread_data = arg;
	unsigned index = thread_data->index;

//	fprintf(stderr, "Hello from thread %d\n", thread_data->index);

	starpu_variable_data_register(&thread_data->handle, 0, (uintptr_t)&thread_data->val, sizeof(unsigned));

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

//	starpu_data_sync_with_mem(thread_data->handle, STARPU_R);
//	fprintf(stderr, "Final value on thread %d: %d\n", thread_data->index, thread_data->val);
//	starpu_data_release_from_mem(thread_data->handle);

	return NULL;
}

int main(int argc, char **argv)
{
	int ret;

	starpu_init(NULL);

	unsigned t;
	for (t = 0; t < NTHREADS; t++)
	{
		problem_data[t].index = t;
		problem_data[t].val = 0;
		pthread_cond_init(&problem_data[t].recv_cond, NULL);
		pthread_mutex_init(&problem_data[t].recv_mutex, NULL);
		problem_data[t].recv_flag = 0;
		problem_data[t].neighbour = &problem_data[(t+1)%NTHREADS];
	}

	for (t = 0; t < NTHREADS; t++)
	{
		ret = pthread_create(&problem_data[t].thread, NULL, thread_func, &problem_data[t]);
		STARPU_ASSERT(!ret);
	}

	for (t = 0; t < NTHREADS; t++)
	{
		void *retval;
		ret = pthread_join(problem_data[t].thread, &retval);
		STARPU_ASSERT(retval == NULL);
	}

	/* We check that the value in the "last" thread is valid */
	starpu_data_handle last_handle = problem_data[NTHREADS - 1].handle;
	starpu_data_sync_with_mem(last_handle, STARPU_R);
	if (problem_data[NTHREADS - 1].val != (NTHREADS * NITER))
		STARPU_ABORT();
	starpu_data_release_from_mem(last_handle);

	starpu_shutdown();

	return 0;
}
