/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

/*
 * This creates two dumb vectors, splits them into chunks, and for each pair of
 * chunk, run axpy on them.
 */

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <common/blas.h>

#ifdef STARPU_USE_CUDA
#include <starpu_cublas_v2.h>
#endif

#include "axpy.h"

#define AXPY	STARPU_SAXPY
#define CUBLASAXPY	cublasSaxpy

#define N	(16*1024*1024)

#define NBLOCKS	8

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define EPSILON 1e-6

TYPE *_vec_x, *_vec_y;
TYPE _alpha = 3.41;

/* descriptors for StarPU */
starpu_data_handle_t _handle_y, _handle_x;

void axpy_cpu(void *descr[], void *arg)
{
	TYPE alpha = *((TYPE *)arg);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	TYPE *block_x = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *block_y = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	AXPY((int)n, alpha, block_x, 1, block_y, 1);
}

#ifdef STARPU_USE_CUDA
void axpy_gpu(void *descr[], void *arg)
{
	TYPE alpha = *((TYPE *)arg);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	TYPE *block_x = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *block_y = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	cublasStatus_t status = CUBLASAXPY(starpu_cublas_get_local_handle(), (int)n, &alpha, block_x, 1, block_y, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

#ifdef STARPU_USE_OPENCL
extern void axpy_opencl(void *buffers[], void *args);
#endif

static struct starpu_perfmodel axpy_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "axpy"
};

static struct starpu_codelet axpy_cl =
{
	.cpu_funcs = {axpy_cpu},
	.cpu_funcs_name = {"axpy_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {axpy_gpu},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {axpy_opencl},
#elif defined(STARPU_SIMGRID)
	.opencl_funcs = {(void*)1},
#endif
	.opencl_flags = {STARPU_OPENCL_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.name = "axpy",
	.model = &axpy_model
};

static int
check(void)
{
	int i;
	for (i = 0; i < N; i++)
	{
		TYPE expected_value = _alpha * _vec_x[i] + 4.0;
		if (fabs(_vec_y[i] - expected_value) > expected_value * EPSILON)
		{
			FPRINTF(stderr,"at %d, %f*%f+%f=%f, expected %f\n", i, _alpha, _vec_x[i], 4.0, _vec_y[i], expected_value);
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(void)
{
	int ret, exit_value = 0;

	/* Initialize StarPU */
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("examples/axpy/axpy_opencl_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	starpu_cublas_init();

	/* This is equivalent to
		vec_a = malloc(N*sizeof(TYPE));
		vec_b = malloc(N*sizeof(TYPE));
	*/
	starpu_malloc((void **)&_vec_x, N*sizeof(TYPE));
	assert(_vec_x);

	starpu_malloc((void **)&_vec_y, N*sizeof(TYPE));
	assert(_vec_y);

	unsigned i;
	for (i = 0; i < N; i++)
	{
		_vec_x[i] = 1.0f; /*(TYPE)starpu_drand48(); */
		_vec_y[i] = 4.0f; /*(TYPE)starpu_drand48(); */
	}

	FPRINTF(stderr, "BEFORE x[0] = %2.2f\n", _vec_x[0]);
	FPRINTF(stderr, "BEFORE y[0] = %2.2f\n", _vec_y[0]);

	/* Declare the data to StarPU */
	starpu_vector_data_register(&_handle_x, STARPU_MAIN_RAM, (uintptr_t)_vec_x, N, sizeof(TYPE));
	starpu_vector_data_register(&_handle_y, STARPU_MAIN_RAM, (uintptr_t)_vec_y, N, sizeof(TYPE));

	/* Divide the vector into blocks */
	struct starpu_data_filter block_filter =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = NBLOCKS
	};

	starpu_data_partition(_handle_x, &block_filter);
	starpu_data_partition(_handle_y, &block_filter);

	double start;
	double end;

	start = starpu_timing_now();

	unsigned b;
	for (b = 0; b < NBLOCKS; b++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &axpy_cl;

		task->cl_arg = &_alpha;
		task->cl_arg_size = sizeof(_alpha);

		task->handles[0] = starpu_data_get_sub_data(_handle_x, 1, b);
		task->handles[1] = starpu_data_get_sub_data(_handle_y, 1, b);

		task->tag_id = b;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV)
		{
			exit_value = 77;
			goto enodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();

enodev:
	starpu_data_unpartition(_handle_x, STARPU_MAIN_RAM);
	starpu_data_unpartition(_handle_y, STARPU_MAIN_RAM);
	starpu_data_unregister(_handle_x);
	starpu_data_unregister(_handle_y);

	end = starpu_timing_now();
        double timing = end - start;

	FPRINTF(stderr, "timing -> %2.2f us %2.2f MB/s\n", timing, 3*N*sizeof(TYPE)/timing);

	FPRINTF(stderr, "AFTER y[0] = %2.2f (ALPHA = %2.2f)\n", _vec_y[0], _alpha);

	if (exit_value != 77)
		exit_value = check();

	starpu_free((void *)_vec_x);
	starpu_free((void *)_vec_y);

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
	/* Stop StarPU */
	starpu_shutdown();

	return exit_value;
}
