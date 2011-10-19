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

#define VECTORSIZE	1024

static unsigned *A, *B, *C, *D;
starpu_data_handle A_handle, B_handle, C_handle, D_handle;

static unsigned var = 0;

static void f(void *descr[], __attribute__ ((unused)) void *_args)
{
	usleep(200000);
}

static starpu_codelet cl_f = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = f,
	.cuda_func = f,
	.nbuffers = 2
};

static void g(void *descr[], __attribute__ ((unused)) void *_args)
{
	usleep(100000);
	var = 42;
}

static starpu_codelet cl_g = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = g,
	.cuda_func = g,
	.nbuffers = 2
};

static void h(void *descr[], __attribute__ ((unused)) void *_args)
{
	FPRINTF(stderr, "VAR %u (should be 42)\n", var);
	STARPU_ASSERT(var == 42);
}

static starpu_codelet cl_h = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = h,
	.cuda_func = h,
	.nbuffers = 2
};

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	A = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));
	B = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));
	C = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));
	D = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));

	starpu_vector_data_register(&A_handle, 0, (uintptr_t)A, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&B_handle, 0, (uintptr_t)B, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&C_handle, 0, (uintptr_t)C, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&D_handle, 0, (uintptr_t)D, VECTORSIZE, sizeof(unsigned));

#if 0
	starpu_data_set_sequential_consistency_flag(A_handle, 0);
	starpu_data_set_sequential_consistency_flag(B_handle, 0);
	starpu_data_set_sequential_consistency_flag(C_handle, 0);
	starpu_data_set_sequential_consistency_flag(D_handle, 0);
#endif

	/* 	f(Ar, Brw): sleep 
	 *	g(Br; Crw); sleep, var = 42
	 *	h(Cr; Drw); check that var == 42
	 */
	struct starpu_task *task_f = starpu_task_create();
	task_f->cl = &cl_f;
	task_f->buffers[0].handle = A_handle;
	task_f->buffers[0].mode = STARPU_R;
	task_f->buffers[1].handle = B_handle;
	task_f->buffers[1].mode = STARPU_RW;
	ret = starpu_task_submit(task_f);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	struct starpu_task *task_g = starpu_task_create();
	task_g->cl = &cl_g;
	task_g->buffers[0].handle = B_handle;
	task_g->buffers[0].mode = STARPU_R;
	task_g->buffers[1].handle = C_handle;
	task_g->buffers[1].mode = STARPU_RW;
	ret = starpu_task_submit(task_g);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	struct starpu_task *task_h = starpu_task_create();
	task_h->cl = &cl_h;
	task_h->buffers[0].handle = C_handle;
	task_h->buffers[0].mode = STARPU_R;
	task_h->buffers[1].handle = D_handle;
	task_h->buffers[1].mode = STARPU_RW;
	ret = starpu_task_submit(task_h);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	free(A);
	free(B);
	free(C);
	free(D);

	starpu_shutdown();

	return 0;

enodev:
	free(A);
	free(B);
	free(C);
	free(D);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return 77;
}
