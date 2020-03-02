/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Try to mix starpu_data_invalidate and starpu_data_invalidate_submit
 * calls with task insertions
 */

#ifdef STARPU_QUICK_CHECK
#  define NLOOPS		100
#else
#  define NLOOPS		1000
#endif
#define VECTORSIZE	1024

static starpu_data_handle_t v_handle;

/*
 *	Memset
 */

#ifdef STARPU_USE_CUDA
static void cuda_memset_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemsetAsync(buf, 42, length, starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
static void opencl_memset_codelet(void *buffers[], void *args)
{
	(void) args;

	cl_command_queue queue;
	int id = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(id);
	starpu_opencl_get_queue(devid, &queue);

	cl_mem buffer = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	unsigned length = STARPU_VECTOR_GET_NX(buffers[0]);
	char *v = malloc(length);
	STARPU_ASSERT(v != NULL);
	memset(v, 42, length);

	clEnqueueWriteBuffer(queue,
			     buffer,
			     CL_FALSE,
			     0,      /* offset */
			     length, /* sizeof (char) */
			     v,
			     0,      /* num_events_in_wait_list */
			     NULL,   /* event_wait_list */
			     NULL    /* event */);
}
#endif /* !STARPU_USE_OPENCL */

void cpu_memset_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	memset(buf, 42, length * sizeof(*buf));
}

static struct starpu_codelet memset_cl =
{
	.cpu_funcs = {cpu_memset_codelet},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_memset_codelet},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_memset_codelet},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"cpu_memset_codelet"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

/*
 *	Check content
 */

void cpu_check_content_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		if (buf[i] != 42)
		{
			FPRINTF(stderr, "buf[%u] is '%c' while it should be '%c'\n", i, buf[i], 42);
			exit(-1);
		}
	}
}

static struct starpu_codelet check_content_cl =
{
	.cpu_funcs = {cpu_check_content_codelet},
	.cpu_funcs_name = {"cpu_check_content_codelet"},
	.nbuffers = 1,
	.modes = {STARPU_R}
};


int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if(starpu_cpu_worker_get_count() == 0) return STARPU_TEST_SKIPPED;

	/* The buffer should never be explicitely allocated */
	starpu_vector_data_register(&v_handle, (uint32_t)-1, (uintptr_t)NULL, VECTORSIZE, sizeof(char));

	unsigned loop;
	for (loop = 0; loop < NLOOPS; loop++)
	{
		struct starpu_task *memset_task;
		struct starpu_task *check_content_task;

		memset_task = starpu_task_create();
		memset_task->cl = &memset_cl;
		memset_task->handles[0] = v_handle;
		memset_task->detach = 0;

		ret = starpu_task_submit(memset_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait(memset_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

		check_content_task = starpu_task_create();
		check_content_task->cl = &check_content_cl;
		check_content_task->handles[0] = v_handle;
		check_content_task->detach = 0;

		ret = starpu_task_submit(check_content_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait(check_content_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

		starpu_data_invalidate(v_handle);
	}

	for (loop = 0; loop < NLOOPS; loop++)
	{
		struct starpu_task *memset_task;
		struct starpu_task *check_content_task;

		memset_task = starpu_task_create();
		memset_task->cl = &memset_cl;
		memset_task->handles[0] = v_handle;

		ret = starpu_task_submit(memset_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		check_content_task = starpu_task_create();
		check_content_task->cl = &check_content_cl;
		check_content_task->handles[0] = v_handle;

		ret = starpu_task_submit(check_content_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		starpu_data_invalidate_submit(v_handle);
	}

	/* this should get rid of automatically allocated buffers */
	starpu_data_unregister(v_handle);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(v_handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
