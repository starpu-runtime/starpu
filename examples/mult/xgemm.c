/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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

#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <starpu.h>

#include <common/blas.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cublas.h>
#include <starpu_cuda.h>
#endif

static unsigned niter = 100;
static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
static unsigned xdim = 256;
static unsigned ydim = 256;
static unsigned zdim = 64;
static unsigned check = 0;

static TYPE *A, *B, *C;
static starpu_data_handle A_handle, B_handle, C_handle;

static void check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", ydim, xdim, zdim, (TYPE)-1.0f, A, ydim, B, zdim, (TYPE)1.0f, C, ydim);
		
	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(xdim*ydim, C, 1);

	if (err < xdim*ydim*0.001) {
		fprintf(stderr, "Results are OK\n");
	}
	else {
		int max;
		max = CPU_IAMAX(xdim*ydim, C, 1);

		fprintf(stderr, "There were errors ... err = %f\n", err);
		fprintf(stderr, "Max error : %e\n", C[max]);
	}
}

static void init_problem_data(void)
{
	unsigned i,j;

	starpu_data_malloc_pinned_if_possible((void **)&A, zdim*ydim*sizeof(TYPE));
	starpu_data_malloc_pinned_if_possible((void **)&B, xdim*zdim*sizeof(TYPE));
	starpu_data_malloc_pinned_if_possible((void **)&C, xdim*ydim*sizeof(TYPE));

	/* fill the A and B matrices */
	for (j=0; j < ydim; j++) {
		for (i=0; i < zdim; i++) {
			A[j+i*ydim] = (TYPE)(starpu_drand48());
		}
	}

	for (j=0; j < zdim; j++) {
		for (i=0; i < xdim; i++) {
			B[j+i*zdim] = (TYPE)(starpu_drand48());
		}
	}

	for (j=0; j < ydim; j++) {
		for (i=0; i < xdim; i++) {
			C[j+i*ydim] = (TYPE)(0);
		}
	}
}

static void partition_mult_data(void)
{
	starpu_matrix_data_register(&A_handle, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(TYPE));

	struct starpu_data_filter f;
	memset(&f, 0, sizeof(f));
	f.filter_func = starpu_vertical_block_filter_func;
	f.nchildren = nslicesx;
		
	struct starpu_data_filter f2;
	memset(&f2, 0, sizeof(f2));
	f2.filter_func = starpu_block_filter_func;
	f2.nchildren = nslicesy;
		
	starpu_data_partition(B_handle, &f);
	starpu_data_partition(A_handle, &f2);

	starpu_data_map_filters(C_handle, 2, &f, &f2);
}

static void mult_kernel_common(void *descr[], int type)
{
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

	if (type == STARPU_CPU) {
		int worker_size = starpu_combined_worker_get_size();

		if (worker_size == 1)
		{
			/* Sequential CPU task */
			CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB, (TYPE)0.0, subC, ldC);
		}
		else {
			/* Parallel CPU task */
			int rank = starpu_combined_worker_get_rank();
		
			int block_size = (nyC + worker_size - 1)/worker_size;
			int new_nyC = STARPU_MIN(nyC, block_size*(rank+1)) - block_size*rank;

			TYPE *new_subA = &subA[block_size*rank];
			TYPE *new_subC = &subC[block_size*rank];

			CPU_GEMM("N", "N", nxC, new_nyC, nyA, (TYPE)1.0, new_subA, ldA, subB, ldB, (TYPE)0.0, new_subC, ldC);
		}
	}
#ifdef STARPU_USE_CUDA
	else {
		CUBLAS_GEMM('n', 'n', nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB,
					     (TYPE)0.0, subC, ldC);
		cudaStreamSynchronize(starpu_cuda_get_local_stream());
	}
#endif
}

#ifdef STARPU_USE_CUDA
static void cublas_mult(void *descr[], __attribute__((unused)) void *arg)
{
	mult_kernel_common(descr, STARPU_CUDA);
}
#endif

static void cpu_mult(void *descr[], __attribute__((unused))  void *arg)
{
	mult_kernel_common(descr, STARPU_CPU);
}

static struct starpu_perfmodel_t starpu_gemm_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = STARPU_GEMM_STR(gemm)
};

static starpu_codelet cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_func = cpu_mult,
#ifdef STARPU_USE_CUDA
	.cuda_func = cublas_mult,
#endif
	.nbuffers = 3,
	.model = &starpu_gemm_model
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
		}

		if (strcmp(argv[i], "-nblocksx") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocksy") == 0) {
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-x") == 0) {
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-y") == 0) {
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-z") == 0) {
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-iter") == 0) {
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-check") == 0) {
			check = 1;
		}

		if (strcmp(argv[i], "-spmd") == 0) {
			cl.type = STARPU_SPMD;
		}
	}
}

int main(int argc, char **argv)
{
	struct timeval start;
	struct timeval end;

	parse_args(argc, argv);

	starpu_init(NULL);
	starpu_helper_cublas_init();

	init_problem_data();
	partition_mult_data();

	gettimeofday(&start, NULL);

	unsigned x, y, iter;
	for (iter = 0; iter < niter; iter++)
	{
		for (x = 0; x < nslicesx; x++) 
		for (y = 0; y < nslicesy; y++)
		{
			struct starpu_task *task = starpu_task_create();
	
			task->cl = &cl;
	
			task->buffers[0].handle = starpu_data_get_sub_data(A_handle, 1, y);
			task->buffers[0].mode = STARPU_R;
			task->buffers[1].handle = starpu_data_get_sub_data(B_handle, 1, x);
			task->buffers[1].mode = STARPU_R;
			task->buffers[2].handle = starpu_data_get_sub_data(C_handle, 2, x, y);
			task->buffers[2].mode = STARPU_RW;
	
			int ret = starpu_task_submit(task);
			STARPU_ASSERT(!ret);
		}

		starpu_task_wait_for_all();
	}


	gettimeofday(&end, NULL);
	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Time: %2.2f ms\n", timing/1000.0);

	double flops = 2.0*((unsigned long)niter)*((unsigned long)xdim)
				*((unsigned long)ydim)*((unsigned long)zdim);
	fprintf(stderr, "GFlop/s: %.2f\n", flops/timing/1000.0);

	starpu_data_unpartition(C_handle, 0);
	starpu_data_unregister(C_handle);
	
	if (check)
		check_output();
	
	starpu_helper_cublas_shutdown();
	starpu_shutdown();

	return 0;
}
