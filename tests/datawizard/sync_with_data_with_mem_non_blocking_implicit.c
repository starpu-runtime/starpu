/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Mix submitting tasks and asynchronously acquiring the corresponding
 * data, but without implicit dependencies.
 */

#define NBUFFERS_DEF	64
#define NITER_DEF	128
#define VECTORSIZE_DEF	1024

static unsigned nbuffers = NBUFFERS_DEF;
static unsigned niter = NITER_DEF;
static unsigned vectorsize = VECTORSIZE_DEF;

float *buffer[NBUFFERS_DEF];

starpu_data_handle_t v_handle[NBUFFERS_DEF];

void dummy_codelet(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

static struct starpu_codelet cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {dummy_codelet},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dummy_codelet},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {dummy_codelet},
#endif
	.cpu_funcs_name = {"dummy_codelet"},
	.nbuffers = 1
};

static
int use_handle(starpu_data_handle_t handle)
{
	int ret;
	struct starpu_task *task;

	task = starpu_task_create();
		task->cl = &cl;
		task->handles[0] = handle;

	ret = starpu_task_submit(task);
	return ret;
}

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;
static unsigned n_synced_buffers;

static
void callback_sync_data(void *arg)
{
	(void)arg;
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);

	n_synced_buffers++;

	if (n_synced_buffers == nbuffers)
		STARPU_PTHREAD_COND_SIGNAL(&cond);

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

int main(int argc, char **argv)
{
	int ret;

#ifdef STARPU_QUICK_CHECK
	nbuffers /= 4;
	niter /= 4;
	vectorsize /= 8;
#endif

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Allocate all buffers and register them to StarPU */
	unsigned b;
	for (b = 0; b < nbuffers; b++)
	{
		ret = starpu_malloc((void **)&buffer[b], vectorsize);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
		starpu_vector_data_register(&v_handle[b], STARPU_MAIN_RAM,
				(uintptr_t)buffer[b], vectorsize, sizeof(char));
	}

	unsigned iter;
	for (iter = 0; iter < niter; iter++)
	{
		/* Use the buffers on the different workers so that it may not
		 * be in main memory anymore */
		for (b = 0; b < nbuffers; b++)
		{
			ret = use_handle(v_handle[b]);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		n_synced_buffers = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

		/* Grab the different pieces of data into main memory */
		for (b = 0; b < nbuffers; b++)
		{
			ret = starpu_data_acquire_cb(v_handle[b], STARPU_RW,
						     callback_sync_data, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire_cb");
		}

		starpu_do_schedule();
		/* Wait for all buffers to be available */
		STARPU_PTHREAD_MUTEX_LOCK(&mutex);

		while (n_synced_buffers != nbuffers)
			STARPU_PTHREAD_COND_WAIT(&cond, &mutex);

		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

		/* Release them */
		for (b = 0; b < nbuffers; b++)
			starpu_data_release(v_handle[b]);
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	/* do some cleanup */
	for (b = 0; b < nbuffers; b++)
	{
		starpu_data_unregister(v_handle[b]);
		starpu_free_noflag(buffer[b], vectorsize);
	}

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
