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

#define NLOOPS		128
#define VECTORSIZE	1024

static unsigned *A;
starpu_data_handle A_handle, B_handle;

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

static starpu_codelet cl_f = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = cpu_f,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_f,
#endif
	.nbuffers = 2
};

int main(int argc, char **argv)
{
	starpu_init(NULL);

	A = calloc(VECTORSIZE, sizeof(unsigned));

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

		int ret = starpu_task_submit(task_f);
		if (ret == -ENODEV)
			goto enodev;
	}

	starpu_task_wait_for_all();

	/* Make sure that data A is in main memory */
	starpu_data_acquire(A_handle, STARPU_R);	

	/* Check result */
	unsigned i;
	for (i = 0; i < VECTORSIZE; i++)
	{
		STARPU_ASSERT(A[i] == NLOOPS);
	}

	starpu_data_release(A_handle);

	starpu_shutdown();

	return 0;

enodev:
	/* No one can execute that task, this is not a bug, just an incomplete
	 * test :) */
	return 0;
}
