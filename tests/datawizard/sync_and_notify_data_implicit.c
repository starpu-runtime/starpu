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

#include "../helper.h"

/*
 * Mix tasks with implicit dependencies and data acquisitions
 */

#define N_DEF	100
#define K_DEF	256

static unsigned n=N_DEF;
static unsigned k=K_DEF;

/*
 * In this test, we maintain a vector v = (a,b,c).
 *
 * Each iteration consists of:
 *  - increment a n times
 *  - sync v in ram
 *  - incrementer b
 *  - notify the modification of v
 *  - incrementer c n times
 *  - sync v 
 *
 * At the end, we have to make sure that if we did k iterations,
 *  v == (kn, k, kn)
 */

#ifdef STARPU_USE_CUDA
void cuda_codelet_incA(void *descr[], void *_args);
void cuda_codelet_incC(void *descr[], void *_args);
#endif

#ifdef STARPU_USE_OPENCL
void opencl_codelet_incA(void *descr[], void *_args);
void opencl_codelet_incC(void *descr[], void *_args);
struct starpu_opencl_program opencl_code;
#endif

#define VECTORSIZE	16

starpu_data_handle_t v_handle;
static unsigned v[VECTORSIZE] STARPU_ATTRIBUTE_ALIGNED(128) = {0, 0, 0, 0};

void cpu_codelet_incA(void *descr[], void *arg)
{
	(void)arg;
	unsigned *val = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	val[0]++;
}

void cpu_codelet_incC(void *descr[], void *arg)
{
	(void)arg;
	unsigned *val = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	val[2]++;
}

/* increment a = v[0] */
static struct starpu_codelet cl_inc_a =
{
	.cpu_funcs = {cpu_codelet_incA},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_codelet_incA},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_codelet_incA},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"cpu_codelet_incA"},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

/* increment c = v[2] */
struct starpu_codelet cl_inc_c =
{
	.cpu_funcs = {cpu_codelet_incC},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_codelet_incC},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_codelet_incC},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"cpu_codelet_incC"},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int main(int argc, char **argv)
{
	int ret;

#ifdef STARPU_QUICK_CHECK
	n /= 10;
#endif
#ifndef STARPU_LONG_CHECK
	k /= 8;
#endif

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_load_opencl_from_file("tests/datawizard/sync_and_notify_data_opencl_codelet.cl", &opencl_code, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

        starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned iter;
	for (iter = 0; iter < k; iter++)
	{
		unsigned ind;
		for (ind = 0; ind < n; ind++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_a;
			task->handles[0] = v_handle;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		/* synchronize v in RAM */
		ret = starpu_data_acquire(v_handle, STARPU_RW);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

		/* increment b */
		v[1]++;

		starpu_data_release(v_handle);

		for (ind = 0; ind < n; ind++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_c;
			task->handles[0] = v_handle;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

	}

	ret = starpu_data_acquire(v_handle, STARPU_RW);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

	FPRINTF(stderr, "V = {%u, %u, %u}\n", v[0], v[1], v[2]);

	starpu_data_release(v_handle);
	starpu_data_unregister(v_handle);
	starpu_shutdown();

	ret = EXIT_SUCCESS;
	if ((v[0] != n*k) || (v[1] != k) || (v[2] != n*k))
	{
		FPRINTF(stderr, "Incorrect result\n");
		ret = EXIT_FAILURE;
	}
	return ret;

enodev:
	starpu_data_unregister(v_handle);
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
