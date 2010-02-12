/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>


#include <common/blas.h>

#define TYPE	float
#define AXPY	SAXPY
#define CUBLASAXPY	cublasSaxpy

#define N	(16*1024*1024)

#define NBLOCKS	8

TYPE *vec_x, *vec_y;

/* descriptors for StarPU */
starpu_data_handle handle_y, handle_x;

void axpy_cpu(void *descr[], __attribute__((unused)) void *arg)
{
	TYPE alpha = *((TYPE *)arg);

	unsigned n = STARPU_GET_VECTOR_NX(descr[0]);

	TYPE *block_x = (TYPE *)STARPU_GET_VECTOR_PTR(descr[0]);
	TYPE *block_y = (TYPE *)STARPU_GET_VECTOR_PTR(descr[1]);

	AXPY((int)n, alpha, block_x, 1, block_y, 1);
}

#ifdef USE_CUDA
void axpy_gpu(void *descr[], __attribute__((unused)) void *arg)
{
	TYPE alpha = *((TYPE *)arg);

	unsigned n = STARPU_GET_VECTOR_NX(descr[0]);

	TYPE *block_x = (TYPE *)STARPU_GET_VECTOR_PTR(descr[0]);
	TYPE *block_y = (TYPE *)STARPU_GET_VECTOR_PTR(descr[1]);

	CUBLASAXPY((int)n, alpha, block_x, 1, block_y, 1);
	cudaThreadSynchronize();
}
#endif

static starpu_codelet axpy_cl = {
        .where =
#ifdef USE_CUDA
                STARPU_CUDA|
#endif
                STARPU_CPU,

	.cpu_func = axpy_cpu,
#ifdef USE_CUDA
	.cuda_func = axpy_gpu,
#endif
	.nbuffers = 2
};

int main(int argc, char **argv)
{
	/* Initialize StarPU */
	starpu_init(NULL);

	starpu_helper_init_cublas();

	/* This is equivalent to 
		vec_a = malloc(N*sizeof(TYPE));
		vec_b = malloc(N*sizeof(TYPE));
	*/
	starpu_malloc_pinned_if_possible((void **)&vec_x, N*sizeof(TYPE));
	assert(vec_x);

	starpu_malloc_pinned_if_possible((void **)&vec_y, N*sizeof(TYPE));
	assert(vec_y);

	unsigned i;
	for (i = 0; i < N; i++)
	{
		vec_x[i] = 1.0f;//(TYPE)drand48();
		vec_y[i] = 4.0f;//(TYPE)drand48();
	}

	fprintf(stderr, "BEFORE x[0] = %2.2f\n", vec_x[0]);
	fprintf(stderr, "BEFORE y[0] = %2.2f\n", vec_y[0]);

	/* Declare the data to StarPU */
	starpu_register_vector_data(&handle_x, 0, (uintptr_t)vec_x, N, sizeof(TYPE));
	starpu_register_vector_data(&handle_y, 0, (uintptr_t)vec_y, N, sizeof(TYPE));

	/* Divide the vector into blocks */
	starpu_filter block_filter = {
		.filter_func = starpu_block_filter_func_vector,
		.filter_arg = NBLOCKS
	};

	starpu_partition_data(handle_x, &block_filter);
	starpu_partition_data(handle_y, &block_filter);

	TYPE alpha = 3.41;

	struct timeval start;
	struct timeval end;
	
	gettimeofday(&start, NULL);

	unsigned b;
	for (b = 0; b < NBLOCKS; b++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &axpy_cl;

		task->cl_arg = &alpha;

		task->buffers[0].handle = starpu_get_sub_data(handle_x, 1, b);
		task->buffers[0].mode = STARPU_R;
		
		task->buffers[1].handle = starpu_get_sub_data(handle_y, 1, b);
		task->buffers[1].mode = STARPU_RW;
		
		starpu_submit_task(task);
	}

	starpu_wait_all_tasks();

	starpu_unpartition_data(handle_y, 0);
	starpu_delete_data(handle_y);

	gettimeofday(&end, NULL);
        double timing = (double)((end.tv_sec - start.tv_sec)*1000000 +
                                        (end.tv_usec - start.tv_usec));

	fprintf(stderr, "timing -> %2.2f us %2.2f MB/s\n", timing, 3*N*sizeof(TYPE)/timing);

	fprintf(stderr, "AFTER y[0] = %2.2f (ALPHA = %2.2f)\n", vec_y[0], alpha);

	/* Stop StarPU */
	starpu_shutdown();

	return 0;
}
