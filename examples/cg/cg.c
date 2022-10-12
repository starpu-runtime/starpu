/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <math.h>
#include <assert.h>
#include <starpu.h>
#include <common/blas.h>


/*
 *	Conjugate Gradient
 *
 *	Input:
 *		- matrix A
 *		- vector b
 *		- vector x (starting value)
 *		- int i_max, error tolerance eps < 1.
 *	Ouput:
 *		- vector x
 *
 *	Pseudo code:
 *
 *		i <- 0
 *		r <- b - Ax
 *		d <- r
 *		delta_new <- dot(r,r)
 *		delta_0 <- delta_new
 *
 *		while (i < i_max && delta_new > eps^2 delta_0)
 *		{
 *			q <- Ad
 *			alpha <- delta_new/dot(d, q)
 *			x <- x + alpha d
 *
 *			If (i is divisible by 50)
 *				r <- b - Ax
 *			else
 *				r <- r - alpha q
 *
 *			delta_old <- delta_new
 *			delta_new <- dot(r,r)
 *			beta <- delta_new/delta_old
 *			d <- r + beta d
 *			i <- i + 1
 *		}
 *
 *	The dot() operations makes use of reduction to optimize parallelism.
 *
 */

#include "cg.h"

static int copy_handle(starpu_data_handle_t dst, starpu_data_handle_t src, unsigned nblocks);

#define HANDLE_TYPE_VECTOR starpu_data_handle_t
#define HANDLE_TYPE_MATRIX starpu_data_handle_t
#define TASK_INSERT(cl, ...) starpu_task_insert(cl, ##__VA_ARGS__)
#define GET_VECTOR_BLOCK(v, i) starpu_data_get_sub_data(v, 1, i)
#define GET_MATRIX_BLOCK(m, i, j) starpu_data_get_sub_data(m, 2, i, j)
#define BARRIER()
#define GET_DATA_HANDLE(handle)
#define FPRINTF_SERVER FPRINTF

#include "cg_kernels.c"

static TYPE *A, *b, *x;
static TYPE *r, *d, *q;

static int copy_handle(starpu_data_handle_t dst, starpu_data_handle_t src, unsigned nb)
{
	unsigned block;

	for (block = 0; block < nb; block++)
		starpu_data_cpy(starpu_data_get_sub_data(dst, 1, block), starpu_data_get_sub_data(src, 1, block), 1, NULL, NULL);
	return 0;
}

/*
 *	Generate Input data
 */
static void generate_random_problem(void)
{
	int i, j;

	starpu_malloc((void **)&A, n*n*sizeof(TYPE));
	starpu_malloc((void **)&b, n*sizeof(TYPE));
	starpu_malloc((void **)&x, n*sizeof(TYPE));
	assert(A && b && x);

	for (j = 0; j < n; j++)
	{
		b[j] = (TYPE)1.0;
		x[j] = (TYPE)0.0;

		/* We take Hilbert matrix that is not well conditionned but definite positive: H(i,j) = 1/(1+i+j) */
		for (i = 0; i < n; i++)
		{
			A[n*j + i] = (TYPE)(1.0/(1.0+i+j));
		}
	}

	/* Internal vectors */
	starpu_malloc((void **)&r, n*sizeof(TYPE));
	starpu_malloc((void **)&d, n*sizeof(TYPE));
	starpu_malloc((void **)&q, n*sizeof(TYPE));
	assert(r && d && q);

	memset(r, 0, n*sizeof(TYPE));
	memset(d, 0, n*sizeof(TYPE));
	memset(q, 0, n*sizeof(TYPE));
}

static void free_data(void)
{
	starpu_free_noflag(A, n*n*sizeof(TYPE));
	starpu_free_noflag(b, n*sizeof(TYPE));
	starpu_free_noflag(x, n*sizeof(TYPE));
	starpu_free_noflag(r, n*sizeof(TYPE));
	starpu_free_noflag(d, n*sizeof(TYPE));
	starpu_free_noflag(q, n*sizeof(TYPE));
}

static void register_data(void)
{
	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, n, n, n, sizeof(TYPE));
	starpu_vector_data_register(&b_handle, STARPU_MAIN_RAM, (uintptr_t)b, n, sizeof(TYPE));
	starpu_vector_data_register(&x_handle, STARPU_MAIN_RAM, (uintptr_t)x, n, sizeof(TYPE));

	starpu_vector_data_register(&r_handle, STARPU_MAIN_RAM, (uintptr_t)r, n, sizeof(TYPE));
	starpu_vector_data_register(&d_handle, STARPU_MAIN_RAM, (uintptr_t)d, n, sizeof(TYPE));
	starpu_vector_data_register(&q_handle, STARPU_MAIN_RAM, (uintptr_t)q, n, sizeof(TYPE));

	starpu_variable_data_register(&dtq_handle, STARPU_MAIN_RAM, (uintptr_t)&dtq, sizeof(TYPE));
	starpu_variable_data_register(&rtr_handle, STARPU_MAIN_RAM, (uintptr_t)&rtr, sizeof(TYPE));

	if (use_reduction)
	{
		starpu_data_set_reduction_methods(q_handle, &accumulate_vector_cl, &bzero_vector_cl);
		starpu_data_set_reduction_methods(r_handle, &accumulate_vector_cl, &bzero_vector_cl);

		starpu_data_set_reduction_methods(dtq_handle, &accumulate_variable_cl, &bzero_variable_cl);
		starpu_data_set_reduction_methods(rtr_handle, &accumulate_variable_cl, &bzero_variable_cl);
	}
}

static void unregister_data(void)
{
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(b_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(x_handle, STARPU_MAIN_RAM);

	starpu_data_unpartition(r_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(d_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(q_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(b_handle);
	starpu_data_unregister(x_handle);

	starpu_data_unregister(r_handle);
	starpu_data_unregister(d_handle);
	starpu_data_unregister(q_handle);

	starpu_data_unregister(dtq_handle);
	starpu_data_unregister(rtr_handle);
}

/*
 *	Data partitioning filters
 */

struct starpu_data_filter vector_filter;
struct starpu_data_filter matrix_filter_1;
struct starpu_data_filter matrix_filter_2;

static void partition_data(void)
{
	assert(n % nblocks == 0);

	/*
	 *	Partition the A matrix
	 */

	/* Partition into contiguous parts */
	matrix_filter_1.filter_func = starpu_matrix_filter_block;
	matrix_filter_1.nchildren = nblocks;
	/* Partition into non-contiguous parts */
	matrix_filter_2.filter_func = starpu_matrix_filter_vertical_block;
	matrix_filter_2.nchildren = nblocks;

	/* A is in FORTRAN ordering, starpu_data_get_sub_data(A_handle, 2, i,
	 * j) designates the block in column i and row j. */
	starpu_data_map_filters(A_handle, 2, &matrix_filter_1, &matrix_filter_2);

	/*
	 *	Partition the vectors
	 */

	vector_filter.filter_func = starpu_vector_filter_block;
	vector_filter.nchildren = nblocks;

	starpu_data_partition(b_handle, &vector_filter);
	starpu_data_partition(x_handle, &vector_filter);
	starpu_data_partition(r_handle, &vector_filter);
	starpu_data_partition(d_handle, &vector_filter);
	starpu_data_partition(q_handle, &vector_filter);
}

/*
 *	Debug
 */

#if 0
static void display_vector(starpu_data_handle_t handle, TYPE *ptr)
{
	unsigned block_size = n / nblocks;

	unsigned b, ind;
	for (b = 0; b < nblocks; b++)
	{
		starpu_data_acquire(starpu_data_get_sub_data(handle, 1, b), STARPU_R);
		for (ind = 0; ind < block_size; ind++)
		{
			FPRINTF(stderr, "%2.2e ", ptr[b*block_size + ind]);
		}
		FPRINTF(stderr, "| ");
		starpu_data_release(starpu_data_get_sub_data(handle, 1, b));
	}
	FPRINTF(stderr, "\n");
}

static void display_matrix(void)
{
	unsigned i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			FPRINTF(stderr, "%2.2e ", A[j*n + i]);
		}
		FPRINTF(stderr, "\n");
	}
}
#endif

static void display_x_result(void)
{
	unsigned j, i;
	starpu_data_handle_t sub;

	FPRINTF(stderr, "Computed X vector:\n");

	unsigned block_size = n / nblocks;

	for (j = 0; j < nblocks; j++)
	{
		sub = starpu_data_get_sub_data(x_handle, 1, j);
		starpu_data_acquire(sub, STARPU_R);
		for (i = 0; i < block_size; i++)
		{
			FPRINTF(stderr, "% 02.2e\n", x[j*block_size + i]);
		}
		starpu_data_release(sub);
	}
}


static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-help") == 0)
		{
			FPRINTF_SERVER(stderr, "usage: %s [-h] [-nblocks #blocks] [-display-result] [-n problem_size] [-no-reduction] [-maxiter i]\n", argv[0]);
			exit(-1);
		}
	}

	parse_common_args(argc, argv);
}


int main(int argc, char **argv)
{
	int ret;
	double start, end;

	/* Not supported yet */
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return 77;

	parse_args(argc, argv);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() + starpu_opencl_worker_get_count() == 0)
	{
		starpu_shutdown();
		return 77;
	}

	starpu_cublas_init();

	FPRINTF(stderr, "************** PARAMETERS ***************\n");
	FPRINTF(stderr, "Problem size (-n): %lld\n", n);
	FPRINTF(stderr, "Maximum number of iterations (-maxiter): %d\n", i_max);
	FPRINTF(stderr, "Number of blocks (-nblocks): %u\n", nblocks);
	FPRINTF(stderr, "Reduction (-no-reduction): %s\n", use_reduction ? "enabled" : "disabled");

	start = starpu_timing_now();
	generate_random_problem();
	register_data();
	partition_data();
	end = starpu_timing_now();

	FPRINTF(stderr, "Problem intialization timing : %2.2f seconds\n", (end-start)/1e6);

	ret = cg();
	if (ret == -ENODEV)
	{
		ret = 77;
		goto enodev;
	}

	starpu_task_wait_for_all();

	if (display_result)
	{
		display_x_result();
	}

enodev:
	unregister_data();
	free_data();
	starpu_cublas_shutdown();
	starpu_shutdown();
	return ret;
}
