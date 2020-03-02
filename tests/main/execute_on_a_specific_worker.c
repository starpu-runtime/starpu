/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"
#include <common/thread.h>

/*
 * Test binding tasks on specific workers
 */

#ifdef STARPU_QUICK_CHECK
  #define N 10
#elif !defined(STARPU_LONG_CHECK)
  #define N 100
#else
  #define N 1000
#endif

#define VECTORSIZE	1024

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

static unsigned finished = 0;

static unsigned cnt;

starpu_data_handle_t v_handle;
static unsigned *v;

static void callback(void *arg)
{
	(void)arg;

	unsigned res = STARPU_ATOMIC_ADD(&cnt, -1);
	ANNOTATE_HAPPENS_BEFORE(&cnt);

	if (res == 0)
	{
		ANNOTATE_HAPPENS_AFTER(&cnt);
		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		finished = 1;
		STARPU_PTHREAD_COND_SIGNAL(&cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	}
}

void codelet_null(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
//	int id = starpu_worker_get_id();
//	FPRINTF(stderr, "worker #%d\n", id);
}

static struct starpu_codelet cl_r =
{
	.cpu_funcs = {codelet_null},
	.cuda_funcs = {codelet_null},
        .opencl_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

static struct starpu_codelet cl_w =
{
	.cpu_funcs = {codelet_null},
	.cuda_funcs = {codelet_null},
        .opencl_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static struct starpu_codelet cl_rw =
{
	.cpu_funcs = {codelet_null},
	.cuda_funcs = {codelet_null},
        .opencl_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

static struct starpu_codelet *select_codelet_with_random_mode(void)
{
	int r = rand();

	switch (r % 3)
	{
		case 0:
			return &cl_r;
		case 1:
			return &cl_w;
		case 2:
			return &cl_rw;
	};
	return &cl_rw;
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_malloc((void **)&v, VECTORSIZE*sizeof(unsigned));
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");

	starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned nworker = starpu_worker_get_count();

	cnt = nworker*N;

	unsigned iter, worker;
	for (iter = 0; iter < N; iter++)
	{
		for (worker = 0; worker < nworker; worker++)
		{
			/* execute a task on that worker */
			struct starpu_task *task = starpu_task_create();

			task->handles[0] = v_handle;
			task->cl = select_codelet_with_random_mode();

			task->callback_func = callback;
			task->callback_arg = NULL;

			task->execute_on_a_specific_worker = 1;
			task->workerid = worker;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	while (!finished)
		STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	starpu_data_unregister(v_handle);
	starpu_free(v);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(v_handle);
	starpu_free(v);
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
