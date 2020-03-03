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

#include <starpu.h>
#include "../helper.h"
#include <common/thread.h>

/*
 * Mimic the behavior of libstarpumpi, tested by a ring of threads which
 * increment the same variable one after the other.
 * This is the asynchronous version: the threads submit the series of
 * synchronizations and tasks.
 */

#ifdef STARPU_QUICK_CHECK
#  define NTHREADS_DEFAULT	4
#  define NITER_DEFAULT		8
#else
#  define NTHREADS_DEFAULT	16
#  define NITER_DEFAULT		128
#endif

static unsigned nthreads = NTHREADS_DEFAULT;
static unsigned niter = NITER_DEFAULT;

//#define DEBUG_MESSAGES	1

//static starpu_pthread_cond_t cond;
//static starpu_pthread_mutex_t mutex;

struct thread_data
{
	unsigned index;
	unsigned val;
	starpu_data_handle_t handle;
	starpu_pthread_t thread;

	starpu_pthread_mutex_t recv_mutex;
	unsigned recv_flag; // set when a message is received
	unsigned recv_buf;
	struct thread_data *neighbour;
};

struct data_req
{
	int (*test_func)(void *);
	void *test_arg;
	struct data_req *next;
};

static starpu_pthread_mutex_t data_req_mutex;
static starpu_pthread_cond_t data_req_cond;
struct data_req *data_req_list;
unsigned progress_thread_running;

static struct thread_data problem_data[NTHREADS_DEFAULT];

/* We implement some ring transfer, every thread will try to receive a piece of
 * data from its neighbour and increment it before transmitting it to its
 * successor. */

#ifdef STARPU_USE_CUDA
void cuda_codelet_unsigned_inc(void *descr[], void *cl_arg);
#endif
#ifdef STARPU_USE_OPENCL
void opencl_codelet_unsigned_inc(void *buffers[], void *cl_arg);
#endif

void increment_handle_cpu_kernel(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*val += 1;

//	FPRINTF(stderr, "VAL %d (&val = %p)\n", *val, val);
}

static struct starpu_codelet increment_handle_cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {increment_handle_cpu_kernel},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_codelet_unsigned_inc},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = { opencl_codelet_unsigned_inc},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"increment_handle_cpu_kernel"},
	.nbuffers = 1
};

static void increment_handle_async(struct thread_data *thread_data)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &increment_handle_cl;

	task->handles[0] = thread_data->handle;

	task->detach = 1;
	task->destroy = 1;

	int ret = starpu_task_submit(task);
	if (ret == -ENODEV)
		exit(STARPU_TEST_SKIPPED);
	STARPU_ASSERT(!ret);
}

static int test_recv_handle_async(void *arg)
{
//	FPRINTF(stderr, "test_recv_handle_async\n");

	int ret;
	struct thread_data *thread_data = (struct thread_data *) arg;

	STARPU_PTHREAD_MUTEX_LOCK(&thread_data->recv_mutex);

	ret = (thread_data->recv_flag == 1);

	if (ret)
	{
		thread_data->recv_flag = 0;
		thread_data->val = thread_data->recv_buf;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&thread_data->recv_mutex);

	if (ret)
	{
#ifdef DEBUG_MESSAGES
		FPRINTF(stderr, "Thread %u received value %u from thread %d\n",
			thread_data->index, thread_data->val, (thread_data->index - 1)%nthreads);
#endif
		starpu_data_release(thread_data->handle);
	}

	return ret;
}

static void recv_handle_async(void *_thread_data)
{
	struct thread_data *thread_data = (struct thread_data *) _thread_data;

	struct data_req *req = (struct data_req *) malloc(sizeof(struct data_req));
	req->test_func = test_recv_handle_async;
	req->test_arg = thread_data;

	STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);
	req->next = data_req_list;
	data_req_list = req;
	STARPU_PTHREAD_COND_SIGNAL(&data_req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_req_mutex);
}

static int test_send_handle_async(void *arg)
{
	int ret;
	struct thread_data *thread_data = (struct thread_data *) arg;
	struct thread_data *neighbour_data = thread_data->neighbour;

	STARPU_PTHREAD_MUTEX_LOCK(&neighbour_data->recv_mutex);
	ret = (neighbour_data->recv_flag == 0);
	STARPU_PTHREAD_MUTEX_UNLOCK(&neighbour_data->recv_mutex);

	if (ret)
	{
#ifdef DEBUG_MESSAGES
		FPRINTF(stderr, "Thread %u sends value %u to thread %u\n", thread_data->index, thread_data->val, neighbour_data->index);
#endif
		starpu_data_release(thread_data->handle);
	}

	return ret;
}

static void send_handle_async(void *_thread_data)
{
	struct thread_data *thread_data = (struct thread_data *) _thread_data;
	struct thread_data *neighbour_data = thread_data->neighbour;

//	FPRINTF(stderr, "send_handle_async\n");

	/* send the message */
	STARPU_PTHREAD_MUTEX_LOCK(&neighbour_data->recv_mutex);
	neighbour_data->recv_buf = thread_data->val;
	neighbour_data->recv_flag = 1;
	STARPU_PTHREAD_MUTEX_UNLOCK(&neighbour_data->recv_mutex);

	struct data_req *req = (struct data_req *) malloc(sizeof(struct data_req));
	req->test_func = test_send_handle_async;
	req->test_arg = thread_data;

	STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);
	req->next = data_req_list;
	data_req_list = req;
	STARPU_PTHREAD_COND_SIGNAL(&data_req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_req_mutex);
}

static void *progress_func(void *arg)
{
	(void)arg;

	STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);

	progress_thread_running = 1;
	STARPU_PTHREAD_COND_SIGNAL(&data_req_cond);

	while (progress_thread_running)
	{
		struct data_req *req;

		if (data_req_list == NULL)
			STARPU_PTHREAD_COND_WAIT(&data_req_cond, &data_req_mutex);

		req = data_req_list;

		if (req)
		{
			data_req_list = req->next;
			req->next = NULL;

			STARPU_PTHREAD_MUTEX_UNLOCK(&data_req_mutex);

			int ret = req->test_func(req->test_arg);

			if (ret)
			{
				free(req);
				STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);
			}
			else
			{
				/* ret = 0 : the request is not finished, we put it back at the end of the list */
				STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);

				struct data_req *req_aux = data_req_list;
				if (!req_aux)
				{
					/* The list is empty */
					data_req_list = req;
				}
				else
				{
					while (req_aux)
					{
						if (req_aux->next == NULL)
						{
							req_aux->next = req;
							break;
						}

						req_aux = req_aux->next;
					}
				}
			}
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_req_mutex);

	return NULL;
}

static void *thread_func(void *arg)
{
	unsigned iter;
	struct thread_data *thread_data = (struct thread_data *) arg;
	unsigned index = thread_data->index;
	int ret;

	starpu_variable_data_register(&thread_data->handle, STARPU_MAIN_RAM, (uintptr_t)&thread_data->val, sizeof(unsigned));

	for (iter = 0; iter < niter; iter++)
	{
		/* The first thread initiates the first transfer */
		if (!((index == 0) && (iter == 0)))
		{
			starpu_data_acquire_cb(
				thread_data->handle, STARPU_W,
				recv_handle_async, thread_data
			);
		}

		increment_handle_async(thread_data);

		if (!((index == (nthreads - 1)) && (iter == (niter - 1))))
		{
			starpu_data_acquire_cb(
				thread_data->handle, STARPU_R,
				send_handle_async, thread_data
			);
		}
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	return NULL;
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(int argc, char **argv)
{
	int ret;
	void *retval;

#ifdef STARPU_QUICK_CHECK
	niter /= 16;
	nthreads /= 4;
#endif

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/datawizard/opencl_codelet_unsigned_inc_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	/* Create a thread to perform blocking calls */
	starpu_pthread_t progress_thread;
	STARPU_PTHREAD_MUTEX_INIT(&data_req_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&data_req_cond, NULL);
	data_req_list = NULL;
	progress_thread_running = 0;

	unsigned t;
	for (t = 0; t < nthreads; t++)
	{
		problem_data[t].index = t;
		problem_data[t].val = 0;
		STARPU_PTHREAD_MUTEX_INIT(&problem_data[t].recv_mutex, NULL);
		problem_data[t].recv_flag = 0;
		problem_data[t].neighbour = &problem_data[(t+1)%nthreads];
	}

	STARPU_PTHREAD_CREATE(&progress_thread, NULL, progress_func, NULL);

	STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);
	while (!progress_thread_running)
		STARPU_PTHREAD_COND_WAIT(&data_req_cond, &data_req_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_req_mutex);

	for (t = 0; t < nthreads; t++)
	{
		STARPU_PTHREAD_CREATE(&problem_data[t].thread, NULL, thread_func, &problem_data[t]);
	}

	for (t = 0; t < nthreads; t++)
	{
		STARPU_PTHREAD_JOIN(problem_data[t].thread, &retval);
		STARPU_ASSERT(retval == NULL);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&data_req_mutex);
	progress_thread_running = 0;
	STARPU_PTHREAD_COND_SIGNAL(&data_req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_req_mutex);

	STARPU_PTHREAD_JOIN(progress_thread, &retval);
	STARPU_ASSERT(retval == NULL);

	/* We check that the value in the "last" thread is valid */
	starpu_data_handle_t last_handle = problem_data[nthreads - 1].handle;
	starpu_data_acquire(last_handle, STARPU_R);

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif

	ret = EXIT_SUCCESS;
	if (problem_data[nthreads - 1].val != (nthreads * niter))
	{
		FPRINTF(stderr, "Final value : %u should be %u\n", problem_data[nthreads - 1].val, (nthreads * niter));
		ret = EXIT_FAILURE;
	}
	starpu_data_release(last_handle);

	for (t = 0; t < nthreads; t++)
	{
		starpu_data_unregister(problem_data[t].handle);
	}

	starpu_shutdown();

	return ret;
}
