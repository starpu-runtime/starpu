/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2013,2015                           Inria
 * Copyright (C) 2009-2017,2019                           Université de Bordeaux
 * Copyright (C) 2010-2013,2015-2017                      CNRS
 * Copyright (C) 2013                                     Thibaut Lambert
 * Copyright (C) 2010                                     Mehdi Juhoor
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
 * This version of the Cholesky factorization can include an
 * externally-compiler-generated loop nest, which allows to play with
 * compiler-side optimizations.
 */

#include "cholesky.h"
#include "../sched_ctx_utils/sched_ctx_utils.h"

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#endif

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static void callback_turn_spmd_on(void *arg)
{
	(void)arg;
	cl22.type = STARPU_SPMD;
}

static int _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	double start;
	double end;

	unsigned long N = starpu_matrix_get_nx(dataA);
	unsigned long nn = N/nblocks;

	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	if (bound_p || bound_lp_p || bound_mps_p)
		starpu_bound_start(bound_deps_p, 0);
	starpu_fxt_start_profiling();

	start = starpu_timing_now();

#define A(i,j) starpu_data_get_sub_data(dataA, 2, j, i)
#define _POTRF(cl, A, prio) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_MAX_PRIO, \
					 STARPU_RW, A, \
					 STARPU_FLOPS, (double) FLOPS_SPOTRF(nn), \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

#define _TRSM(cl, A, B, prio) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_DEFAULT_PRIO, \
					 STARPU_R, A, \
					 STARPU_RW, B, \
					 STARPU_FLOPS, (double) FLOPS_STRSM(nn,nn), \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

/* TODO: use real SYRK */
#define _SYRK(cl, A, C, prio) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_DEFAULT_PRIO, \
					 STARPU_R, A, \
					 STARPU_R, A, \
					 STARPU_RW, C, \
					 STARPU_FLOPS, (double) FLOPS_SGEMM(nn,nn,nn), \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

#define _GEMM(cl, A, B, C, prio) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_DEFAULT_PRIO, \
					 STARPU_R, A, \
					 STARPU_R, B, \
					 STARPU_RW, C, \
					 STARPU_FLOPS, (double) FLOPS_SGEMM(nn,nn,nn), \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

#define POTRF(A, prio)		_POTRF(&cl11, A, prio)
#define TRSM(A, B, prio)	_TRSM(&cl21, A, B, prio)
#define SYRK(A, B, prio)	_SYRK(&cl22, A, B, prio)
#define GEMM(A, B, C, prio)	_GEMM(&cl22, A, B, C, prio)

#define POTRF_GPU(A, prio)	_POTRF(&cl11_gpu, A, prio)
#define TRSM_GPU(A, B, prio)	_TRSM(&cl21_gpu, A, B, prio)
#define SYRK_GPU(A, B, prio)	_SYRK(&cl22_gpu, A, B, prio)
#define GEMM_GPU(A, B, C, prio)	_GEMM(&cl22_gpu, A, B, C, prio)

#define POTRF_CPU(A, prio)	_POTRF(&cl11_cpu, A, prio)
#define TRSM_CPU(A, B, prio)	_TRSM(&cl21_cpu, A, B, prio)
#define SYRK_CPU(A, B, prio)	_SYRK(&cl22_cpu, A, B, prio)
#define GEMM_CPU(A, B, C, prio)	_GEMM(&cl22_cpu, A, B, C, prio)

#include "cholesky_compiled.c"

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	starpu_fxt_stop_profiling();
	if (bound_p || bound_lp_p || bound_mps_p)
		starpu_bound_stop();

	double timing = end - start;

	double flop = FLOPS_SPOTRF(N);

	if(with_ctxs_p || with_noctxs_p || chole1_p || chole2_p)
		update_sched_ctx_timing_results((flop/timing/1000.0f), (timing/1000000.0f));
	else
	{
		PRINTF("# size\tms\tGFlops");
		if (bound_p)
			PRINTF("\tTms\tTGFlops");
		PRINTF("\n");

		PRINTF("%lu\t%.0f\t%.1f", N, timing/1000, (flop/timing/1000.0f));
		if (bound_lp_p)
		{
			FILE *f = fopen("cholesky.lp", "w");
			starpu_bound_print_lp(f);
			fclose(f);
		}
		if (bound_mps_p)
		{
			FILE *f = fopen("cholesky.mps", "w");
			starpu_bound_print_mps(f);
			fclose(f);
		}
		if (bound_p)
		{
			double res;
			starpu_bound_compute(&res, NULL, 0);
			PRINTF("\t%.0f\t%.1f", res, (flop/res/1000000.0f));
		}
		PRINTF("\n");
	}
	return 0;
}

static int cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks)
{
	starpu_data_handle_t dataA;
	unsigned x, y;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld, size, size, sizeof(float));

	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nblocks
	};

	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nblocks
	};

	starpu_data_map_filters(dataA, 2, &f, &f2);

	for (x = 0; x < nblocks; x++)
		for (y = 0; y < nblocks; y++)
		{
			starpu_data_handle_t data = starpu_data_get_sub_data(dataA, 2, x, y);
			starpu_data_set_coordinates(data, 2, x, y);
		}

	int ret = _cholesky(dataA, nblocks);

	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
	starpu_data_unregister(dataA);

	return ret;
}

static void execute_cholesky(unsigned size, unsigned nblocks)
{
	float *mat = NULL;

#ifndef STARPU_SIMGRID
	unsigned i,j;
	starpu_malloc_flags((void **)&mat, (size_t)size*size*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			/* mat[j +i*size] = ((i == j)?1.0f*size:0.0f); */
		}
	}

/* #define PRINT_OUTPUT */
#ifdef PRINT_OUTPUT
	FPRINTF(stdout, "Input :\n");

	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j)
			{
				FPRINTF(stdout, "%2.2f\t", mat[j +i*size]);
			}
			else
			{
				FPRINTF(stdout, ".\t");
			}
		}
		FPRINTF(stdout, "\n");
	}
#endif
#endif

	cholesky(mat, size, size, nblocks);

#ifndef STARPU_SIMGRID
#ifdef PRINT_OUTPUT
	FPRINTF(stdout, "Results :\n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j)
			{
				FPRINTF(stdout, "%2.2f\t", mat[j +i*size]);
			}
			else
			{
				FPRINTF(stdout, ".\t");
				mat[j+i*size] = 0.0f; /* debug */
			}
		}
		FPRINTF(stdout, "\n");
	}
#endif

	if (check_p)
	{
		FPRINTF(stderr, "compute explicit LLt ...\n");
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i > j)
				{
					mat[j+i*size] = 0.0f; /* debug */
				}
			}
		}
		float *test_mat = malloc(size*size*sizeof(float));
		STARPU_ASSERT(test_mat);

		STARPU_SSYRK("L", "N", size, size, 1.0f,
					mat, size, 0.0f, test_mat, size);

		FPRINTF(stderr, "comparing results ...\n");
#ifdef PRINT_OUTPUT
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i <= j)
				{
					FPRINTF(stdout, "%2.2f\t", test_mat[j +i*size]);
				}
				else
				{
					FPRINTF(stdout, ".\t");
				}
			}
			FPRINTF(stdout, "\n");
		}
#endif

		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i <= j)
				{
	                                float orig = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
	                                float err = fabsf(test_mat[j +i*size] - orig) / orig;
	                                if (err > 0.00001)
					{
	                                        FPRINTF(stderr, "Error[%u, %u] --> %2.6f != %2.6f (err %2.6f)\n", i, j, test_mat[j +i*size], orig, err);
	                                        assert(0);
	                                }
	                        }
			}
	        }
		free(test_mat);
	}
	starpu_free_flags(mat, (size_t)size*size*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
#endif
}

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

#ifdef STARPU_HAVE_MAGMA
	magma_init();
#endif

	int ret;
	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	//starpu_fxt_stop_profiling();

	init_sizes();

	parse_args(argc, argv);

	if(with_ctxs_p || with_noctxs_p || chole1_p || chole2_p)
		parse_args_ctx(argc, argv);

#ifdef STARPU_USE_CUDA
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,cuda_chol_task_11_cost);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,cuda_chol_task_21_cost);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,cuda_chol_task_22_cost);
#else
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,NULL);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,NULL);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,NULL);
#endif

	starpu_cublas_init();

	if(with_ctxs_p)
	{
		construct_contexts();
		start_2benchs(execute_cholesky);
	}
	else if(with_noctxs_p)
		start_2benchs(execute_cholesky);
	else if(chole1_p)
		start_1stbench(execute_cholesky);
	else if(chole2_p)
		start_2ndbench(execute_cholesky);
	else
		execute_cholesky(size_p, nblocks_p);

	starpu_cublas_shutdown();
	starpu_shutdown();

	return 0;
}
