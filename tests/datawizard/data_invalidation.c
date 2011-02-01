/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#define NLOOPS		1000
#define VECTORSIZE	1024

static starpu_data_handle v_handle;

/*
 *	Memset
 */

#ifdef STARPU_USE_CUDA
static void cuda_memset_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemset(buf, 42, length);
	cudaThreadSynchronize();
}
#endif

static void cpu_memset_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	memset(buf, 42, length);
}

static starpu_codelet memset_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = cpu_memset_codelet,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_memset_codelet,
#endif
	.nbuffers = 1
};

/*
 *	Check content
 */

static void cpu_check_content_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		if (buf[i] != 42)
		{
			fprintf(stderr, "buf[%d] is %c while it should be %c\n", i, buf[i], 42);
			exit(-1);
		}
	}
}

static starpu_codelet check_content_cl = {
	.where = STARPU_CPU,
	.cpu_func = cpu_check_content_codelet,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	int ret;

	starpu_init(NULL);

	/* The buffer should never be explicitely allocated */
	starpu_vector_data_register(&v_handle, (uint32_t)-1, (uintptr_t)NULL, VECTORSIZE, sizeof(char));

	unsigned loop;
	for (loop = 0; loop < NLOOPS; loop++)
	{
		struct starpu_task *memset_task;
		struct starpu_task *check_content_task;

		memset_task = starpu_task_create();
		memset_task->cl = &memset_cl;
		memset_task->buffers[0].handle = v_handle;
		memset_task->buffers[0].mode = STARPU_W;
		memset_task->detach = 0;
	
		ret = starpu_task_submit(memset_task);
		if (ret == -ENODEV)
				goto enodev;
	
		ret = starpu_task_wait(memset_task);
		if (ret)
			exit(-1);
		
		check_content_task = starpu_task_create();
		check_content_task->cl = &check_content_cl;
		check_content_task->buffers[0].handle = v_handle;
		check_content_task->buffers[0].mode = STARPU_R;
		check_content_task->detach = 0;
	
		ret = starpu_task_submit(check_content_task);
		if (ret == -ENODEV)
				goto enodev;
	
		ret = starpu_task_wait(check_content_task);
		if (ret)
			exit(-1);

		starpu_data_invalidate(v_handle);
	}

	/* this should get rid of automatically allocated buffers */
	starpu_data_unregister(v_handle);

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
