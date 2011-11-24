/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include "../common/helper.h"

#define NLOOPS		128
#define VECTORSIZE	1024

static unsigned *A;
starpu_data_handle_t A_handle, B_handle;

//static unsigned var = 0;

#ifdef STARPU_USE_CUDA
extern void cuda_f(void *descr[], __attribute__ ((unused)) void *_args);
#endif

static void cpu_f(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *v = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned *tmp = (unsigned *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(descr[0]);

	memcpy(tmp, v, nx*elemsize);

	unsigned i;
	for (i = 0; i < nx; i++)
	{
		v[i] = tmp[i] + 1;
	}
}

static struct starpu_codelet cl_f = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = cpu_f,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_f,
#endif
	.nbuffers = 2
};

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	A = (unsigned *) calloc(VECTORSIZE, sizeof(unsigned));

	starpu_vector_data_register(&A_handle, 0, (uintptr_t)A, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&B_handle, -1, (uintptr_t)NULL, VECTORSIZE, sizeof(unsigned));

	unsigned loop;
	for (loop = 0; loop < NLOOPS; loop++)
	{
		struct starpu_task *task_f = starpu_task_create();
		task_f->cl = &cl_f;
		task_f->buffers[0].handle = A_handle;
		task_f->buffers[0].mode = STARPU_RW;
		task_f->buffers[1].handle = B_handle;
		task_f->buffers[1].mode = STARPU_SCRATCH;

		ret = starpu_task_submit(task_f);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	/* Make sure that data A is in main memory */
	ret = starpu_data_acquire(A_handle, STARPU_R);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

	/* Check result */
	unsigned i;
	for (i = 0; i < VECTORSIZE; i++)
	{
		STARPU_ASSERT(A[i] == NLOOPS);
	}

	starpu_data_release(A_handle);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_shutdown();
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	fprintf(stderr, "WARNING: No one can execute this task\n");
	return 77;
}
