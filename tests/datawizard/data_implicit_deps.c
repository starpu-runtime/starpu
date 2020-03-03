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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Test that implicit dependencies get properly computed
 */

#define VECTORSIZE	1024

static unsigned *A, *B, *C, *D;
starpu_data_handle_t A_handle, B_handle, C_handle, D_handle;

static unsigned var = 0;
starpu_data_handle_t var_handle;

void f(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;

	STARPU_SKIP_IF_VALGRIND;

	usleep(200000);
}

static struct starpu_codelet cl_f =
{
	.modes = { STARPU_RW, STARPU_R, STARPU_RW },
	.cpu_funcs = {f},
	.cuda_funcs = {f},
	.opencl_funcs = {f},
	.cpu_funcs_name = {"f"},
	.nbuffers = 3,
};

void g(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *val = (unsigned *) STARPU_VARIABLE_GET_PTR(descr[0]);

	usleep(100000);
	*val = 42;
}

#ifdef STARPU_USE_CUDA
void g_cuda(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *val = (unsigned *) STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned value = 42;

	usleep(100000);
	cudaMemcpyAsync(val, &value, sizeof(value), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
void g_opencl(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	cl_mem val = (cl_mem) STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned value = 42;

	usleep(100000);
	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueWriteBuffer(queue, val, CL_TRUE, 0, sizeof(unsigned), (void *)&value, 0, NULL, NULL);
	clFinish(queue);
}
#endif

static struct starpu_codelet cl_g =
{
	.modes = { STARPU_RW, STARPU_R, STARPU_RW },
	.cpu_funcs = {g},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {g_cuda},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {g_opencl},
#endif
	.cpu_funcs_name = {"g"},
	.nbuffers = 3,
};

void h(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *val = (unsigned *) STARPU_VARIABLE_GET_PTR(descr[0]);

	FPRINTF(stderr, "VAR %u (should be 42)\n", *val);
	STARPU_ASSERT(*val == 42);
}

#ifdef STARPU_USE_CUDA
void h_cuda(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *val = (unsigned *) STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned value;

	cudaMemcpyAsync(&value, val, sizeof(value), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
	FPRINTF(stderr, "VAR %u (should be 42)\n", value);
	STARPU_ASSERT(value == 42);
}
#endif

#ifdef STARPU_USE_OPENCL
void h_opencl(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	cl_mem val = (cl_mem) STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned value = 0;

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueReadBuffer(queue, val, CL_TRUE, 0, sizeof(unsigned), (void *)&value, 0, NULL, NULL);
	clFinish(queue);

	FPRINTF(stderr, "VAR %u (should be 42)\n", value);
	STARPU_ASSERT(value == 42);
}
#endif

static struct starpu_codelet cl_h =
{
	.modes = { STARPU_RW, STARPU_R, STARPU_RW },
	.cpu_funcs = {h},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {h_cuda},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {h_opencl},
#endif
	.cpu_funcs_name = {"h"},
	.nbuffers = 3
};

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	A = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));
	B = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));
	C = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));
	D = (unsigned *) malloc(VECTORSIZE*sizeof(unsigned));

	starpu_vector_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C, VECTORSIZE, sizeof(unsigned));
	starpu_vector_data_register(&D_handle, STARPU_MAIN_RAM, (uintptr_t)D, VECTORSIZE, sizeof(unsigned));

	starpu_variable_data_register(&var_handle, STARPU_MAIN_RAM, (uintptr_t)(&var), sizeof(var));

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
	task_f->handles[0] = var_handle;
	task_f->handles[1] = A_handle;
	task_f->handles[2] = B_handle;
	ret = starpu_task_submit(task_f);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	struct starpu_task *task_g = starpu_task_create();
	task_g->cl = &cl_g;
	task_g->handles[0] = var_handle;
	task_g->handles[1] = B_handle;
	task_g->handles[2] = C_handle;
	ret = starpu_task_submit(task_g);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	struct starpu_task *task_h = starpu_task_create();
	task_h->cl = &cl_h;
	task_h->handles[0] = var_handle;
	task_h->handles[1] = C_handle;
	task_h->handles[2] = D_handle;
	ret = starpu_task_submit(task_h);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);
	starpu_data_unregister(D_handle);

	starpu_data_unregister(var_handle);

	free(A);
	free(B);
	free(C);
	free(D);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);
	starpu_data_unregister(D_handle);

	starpu_data_unregister(var_handle);

	free(A);
	free(B);
	free(C);
	free(D);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
