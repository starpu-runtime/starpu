/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2012  Université de Bordeaux 1
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
 * This examples shows how to submit a pipeline to StarPU with limited buffer
 * use, and avoiding submitted all the tasks at once.
 *
 * This is a dumb example pipeline, depicted here:
 *
 * x--\
 *     >==axpy-->sum
 * y--/
 *
 * x and y produce vectors full of x and y values, axpy multiplies them, and sum
 * sums it up. We thus have 3 temporary buffers
 */

#include <starpu.h>
#include <stdint.h>
#include <semaphore.h>
#include <common/blas.h>

#ifdef STARPU_USE_CUDA
#include <cublas.h>
#endif

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

/* Vector size */
#ifdef STARPU_SLOW_MACHINE
#define N 16
#else
#define N 1048576
#endif

/* Number of iteration buffers, and thus overlapped pipeline iterations */
#define K 16

/* Number of concurrently submitted pipeline iterations */
#define C 64

/* Number of iterations */
#define L 256

/* X / Y codelets */
void pipeline_cpu_x(void *descr[], void *args)
{
	float x;
	float *val = (float *) STARPU_VECTOR_GET_PTR(descr[0]);
	int n = STARPU_VECTOR_GET_NX(descr[0]);
	int i;

	starpu_codelet_unpack_args(args, &x);
	for (i = 0; i < n ; i++)
		val[i] = x;
}

static struct starpu_perfmodel pipeline_model_x =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "pipeline_model_x"
};

static struct starpu_codelet pipeline_codelet_x =
{
	.where = STARPU_CPU,
	.cpu_funcs = {pipeline_cpu_x, NULL},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.model = &pipeline_model_x
};

/* axpy codelets */
void pipeline_cpu_axpy(void *descr[], void *arg)
{
	float *x = (float *) STARPU_VECTOR_GET_PTR(descr[0]);
	float *y = (float *) STARPU_VECTOR_GET_PTR(descr[1]);
	int n = STARPU_VECTOR_GET_NX(descr[0]);

	SAXPY(n, 1., x, 1, y, 1);
}

#ifdef STARPU_USE_CUDA
void pipeline_cublas_axpy(void *descr[], void *arg)
{
	float *x = (float *) STARPU_VECTOR_GET_PTR(descr[0]);
	float *y = (float *) STARPU_VECTOR_GET_PTR(descr[1]);
	int n = STARPU_VECTOR_GET_NX(descr[0]);

	cublasSaxpy(n, 1., x, 1, y, 1);
}
#endif

static struct starpu_perfmodel pipeline_model_axpy =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "pipeline_model_axpy"
};

static struct starpu_codelet pipeline_codelet_axpy =
{
	.where = STARPU_CPU
#ifdef STARPU_USE_CUDA
		| STARPU_CUDA
#endif
		,
	.cpu_funcs = {pipeline_cpu_axpy, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {pipeline_cublas_axpy, NULL},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &pipeline_model_axpy
};

/* sum codelet */
void pipeline_cpu_sum(void *descr[], void *_args)
{
	float *x = (float *) STARPU_VECTOR_GET_PTR(descr[0]);
	int n = STARPU_VECTOR_GET_NX(descr[0]);
	float y;

	y = SASUM(n, x, 1);

	FPRINTF(stderr,"CPU finished with %f\n", y);
}

#ifdef STARPU_USE_CUDA
void pipeline_cublas_sum(void *descr[], void *arg)
{
	float *x = (float *) STARPU_VECTOR_GET_PTR(descr[0]);
	int n = STARPU_VECTOR_GET_NX(descr[0]);
	float y;

	y = cublasSasum(n, x, 1);

	FPRINTF(stderr,"CUBLAS finished with %f\n", y);
}
#endif

static struct starpu_perfmodel pipeline_model_sum =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "pipeline_model_sum"
};

static struct starpu_codelet pipeline_codelet_sum =
{
	.where = STARPU_CPU
#ifdef STARPU_USE_CUDA
		| STARPU_CUDA
#endif
		,
	.cpu_funcs = {pipeline_cpu_sum, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {pipeline_cublas_sum, NULL},
#endif
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &pipeline_model_sum
};

int main(void)
{
	int ret = 0;
	int k, l, c;
	starpu_data_handle_t buffersX[K], buffersY[K], buffersP[K];
	sem_t sems[C];

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_helper_cublas_init();

	/* Initialize the K temporary buffers. No need to allocate it ourselves
	 * Since it's the X and Y kernels which will fill the initial values. */
	for (k = 0; k < K; k++)
	{
		starpu_vector_data_register(&buffersX[k], -1, 0, N, sizeof(float));
		starpu_vector_data_register(&buffersY[k], -1, 0, N, sizeof(float));
		starpu_vector_data_register(&buffersP[k], -1, 0, N, sizeof(float));
	}

	/* Initialize way to wait for the C previous concurrent stages */
	for (c = 0; c < C; c++)
		sem_init(&sems[c], 0, 0);

	/* Submits the l pipeline stages */
	for (l = 0; l < L; l++)
	{
		float x = l;
		float y = 2*l;
		/* First wait for the C previous concurrent stages */
		if (l >= C)
			sem_wait(&sems[l%C]);

		/* Now submit the next stage */
		ret = starpu_insert_task(&pipeline_codelet_x,
				STARPU_W, buffersX[l%K],
				STARPU_VALUE, &x, sizeof(x),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task x");

		ret = starpu_insert_task(&pipeline_codelet_x,
				STARPU_W, buffersY[l%K],
				STARPU_VALUE, &y, sizeof(y),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task y");

		ret = starpu_insert_task(&pipeline_codelet_axpy,
				STARPU_R, buffersX[l%K],
				STARPU_RW, buffersY[l%K],
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task axpy");

		ret = starpu_insert_task(&pipeline_codelet_sum,
				STARPU_R, buffersY[l%K],
				STARPU_CALLBACK_WITH_ARG, (void (*)(void*))sem_post, &sems[l%C],
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task sum");
	}
	starpu_task_wait_for_all();

enodev:
	for (k = 0; k < K; k++)
	{
		starpu_data_unregister(buffersX[k]);
		starpu_data_unregister(buffersY[k]);
		starpu_data_unregister(buffersP[k]);
	}
	starpu_shutdown();

	return (ret == -ENODEV ? 77 : 0);
}
