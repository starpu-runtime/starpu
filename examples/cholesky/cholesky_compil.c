/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2013       Thibaut Lambert
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

/* Note: this is using fortran ordering, i.e. column-major ordering, i.e.
 * elements with consecutive row number are consecutive in memory */

#include "cholesky.h"
#include "../sched_ctx_utils/sched_ctx_utils.h"
#include <math.h>

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

	unsigned long nelems = starpu_matrix_get_nx(dataA);
	unsigned long nn = nelems/nblocks;
	int M = nblocks;
	int N = nblocks;

	int lambda_b = starpu_get_env_float_default("CHOLESKY_LAMBDA_B", nblocks);
	int lambda_o_u = starpu_get_env_float_default("CHOLESKY_LAMBDA_O_U", 0);
	int lambda_o_d = starpu_get_env_float_default("CHOLESKY_LAMBDA_O_D", 0);

	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	if (bound_p || bound_lp_p || bound_mps_p)
		starpu_bound_start(bound_deps_p, 0);
	starpu_fxt_start_profiling();

	start = starpu_timing_now();

#define min(x,y)  (x<y?x:y)
#define max(x,y)  (x<y?y:x)
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))

#define A(i,j) starpu_data_get_sub_data(dataA, 2, i, j)

#define _POTRF(cl, A, prio, name) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_MAX_PRIO, \
					 STARPU_RW, A, \
					 STARPU_FLOPS, (double) FLOPS_SPOTRF(nn), \
					 STARPU_NAME, name, \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

#define _TRSM(cl, A, B, prio, name) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_DEFAULT_PRIO, \
					 STARPU_R, A, \
					 STARPU_RW, B, \
					 STARPU_FLOPS, (double) FLOPS_STRSM(nn,nn), \
					 STARPU_NAME, name, \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

/* TODO: use real SYRK */
#define _SYRK(cl, A, C, prio, name) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_DEFAULT_PRIO, \
					 STARPU_R, A, \
					 STARPU_R, A, \
					 STARPU_RW, C, \
					 STARPU_FLOPS, (double) FLOPS_SGEMM(nn,nn,nn), \
					 STARPU_NAME, name, \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

#define _GEMM(cl, A, B, C, prio, name) do { \
		int ret = starpu_task_insert(cl, \
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int) (prio) : (int) STARPU_DEFAULT_PRIO, \
					 STARPU_R, A, \
					 STARPU_R, B, \
					 STARPU_RW, C, \
					 STARPU_FLOPS, (double) FLOPS_SGEMM(nn,nn,nn), \
					 STARPU_NAME, name, \
					 0); \
		if (ret == -ENODEV) return 77; \
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert"); \
} while (0)

#define POTRF(A, prio)		_POTRF(&cl11, A, prio, "potrf")
#define TRSM(A, B, prio)	_TRSM(&cl21, A, B, prio, "trsm")
#define SYRK(A, B, prio)	_SYRK(&cl22, A, B, prio, "syrk")
#define GEMM(A, B, C, prio)	_GEMM(&cl22, A, B, C, prio, "gemm")

#define POTRF_GPU(A, prio)	_POTRF(&cl11_gpu, A, prio, "potrf_gpu")
#define TRSM_GPU(A, B, prio)	_TRSM(&cl21_gpu, A, B, prio, "trsm_gpu")
#define SYRK_GPU(A, B, prio)	_SYRK(&cl22_gpu, A, B, prio, "syrk_gpu")
#define GEMM_GPU(A, B, C, prio)	_GEMM(&cl22_gpu, A, B, C, prio, "gemm_gpu")

#define POTRF_CPU(A, prio)	_POTRF(&cl11_cpu, A, prio, "potrf_cpu")
#define TRSM_CPU(A, B, prio)	_TRSM(&cl21_cpu, A, B, prio, "trsm_cpu")
#define SYRK_CPU(A, B, prio)	_SYRK(&cl22_cpu, A, B, prio, "syrk_cpu")
#define GEMM_CPU(A, B, C, prio)	_GEMM(&cl22_cpu, A, B, C, prio, "gemm_cpu")

#define potrf_oreille_up(k)		{ POTRF_GPU(A(k,k),(2*N - 2*k)); }
#define potrf_oreille_down(k)		{ POTRF_GPU(A(k,k),(2*N - 2*k)); }
#define potrf_cpu(k)			{ POTRF_CPU(A(k,k),(2*N - 2*k)); }
#define potrf_bande(k)			{ POTRF(A(k,k),(2*N - 2*k)); }
#define trsm_oreille_up(k,m)		{ TRSM_GPU(A(k,k),A(m,k), (2*nblocks - 2*k - m)); }
#define trsm_oreille_down(k,m)		{ TRSM_GPU(A(k,k),A(m,k), (2*nblocks - 2*k - m)); }
#define trsm_cpu(k,m)			{ TRSM_CPU(A(k,k),A(m,k), (2*nblocks - 2*k - m)); }
#define trsm_bande(k,m)			{ TRSM(A(k,k),A(m,k), (2*nblocks - 2*k - m)); }
#define herk_oreille_up(k,n)		{ SYRK_GPU(A(n,k),A(n,n), (2*nblocks - 2*k - n)); }
#define herk_oreille_down(k,n)		{ SYRK_GPU(A(n,k),A(n,n), (2*nblocks - 2*k - n)); }
#define herk_cpu(k,n)			{ SYRK(A(n,k),A(n,n), (2*nblocks - 2*k - n)); }
#define herk_bande(k,n)			{ SYRK(A(n,k),A(n,n), (2*nblocks - 2*k - n)); }
#define gemm_oreille_up(k,n,m)		{ GEMM_GPU(A(m,k),A(n,k),A(m,n), (2*nblocks - 2*k - n - m)); }
#define gemm_oreille_down(k,n,m)	{ GEMM_GPU(A(m,k),A(n,k),A(m,n), (2*nblocks - 2*k - n - m)); }
#define gemm_cpu(k,n,m)			{ GEMM(A(m,k),A(n,k),A(m,n), (2*nblocks - 2*k - n - m)); }
#define gemm_bande(k,n,m)		{ GEMM(A(m,k),A(n,k),A(m,n), (2*nblocks - 2*k - n - m)); }

#include "cholesky_compiled.c"

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	starpu_fxt_stop_profiling();
	if (bound_p || bound_lp_p || bound_mps_p)
		starpu_bound_stop();

	double timing = end - start;

	double flop = FLOPS_SPOTRF(nelems);

	if(with_ctxs_p || with_noctxs_p || chole1_p || chole2_p)
		update_sched_ctx_timing_results((flop/timing/1000.0f), (timing/1000000.0f));
	else
	{
		PRINTF("# size\tms\tGFlops");
		if (bound_p)
			PRINTF("\tTms\tTGFlops");
		PRINTF("\n");

		PRINTF("%lu\t%.0f\t%.1f", nelems, timing/1000, (flop/timing/1000.0f));
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
	unsigned m, n;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (m,n) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld, size, size, sizeof(float));

	/* Split into blocks of complete rows first */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nblocks
	};

	/* Then split rows into tiles */
	struct starpu_data_filter f2 =
	{
		/* Note: here "vertical" is for row-major, we are here using column-major. */
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nblocks
	};

	starpu_data_map_filters(dataA, 2, &f, &f2);

	for (m = 0; m < nblocks; m++)
		for (n = 0; n < nblocks; n++)
		{
			starpu_data_handle_t data = starpu_data_get_sub_data(dataA, 2, m, n);
			starpu_data_set_coordinates(data, 2, m, n);
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
	unsigned m,n;
	starpu_malloc_flags((void **)&mat, (size_t)size*size*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	for (n = 0; n < size; n++)
	{
		for (m = 0; m < size; m++)
		{
			mat[m +n*size] = (1.0f/(1.0f+m+n)) + ((m == n)?1.0f*size:0.0f);
			/* mat[m +n*size] = ((m == n)?1.0f*size:0.0f); */
		}
	}

/* #define PRINT_OUTPUT */
#ifdef PRINT_OUTPUT
	FPRINTF(stdout, "Input :\n");

	for (m = 0; m < size; m++)
	{
		for (n = 0; n < size; n++)
		{
			if (n <= m)
			{
				FPRINTF(stdout, "%2.2f\t", mat[m +n*size]);
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
	for (m = 0; m < size; m++)
	{
		for (n = 0; n < size; n++)
		{
			if (n <= m)
			{
				FPRINTF(stdout, "%2.2f\t", mat[m +n*size]);
			}
			else
			{
				FPRINTF(stdout, ".\t");
			}
		}
		FPRINTF(stdout, "\n");
	}
#endif

	if (check_p)
	{
		FPRINTF(stderr, "compute explicit LLt ...\n");
		for (m = 0; m < size; m++)
		{
			for (n = 0; n < size; n++)
			{
				if (n > m)
				{
					mat[m+n*size] = 0.0f; /* debug */
				}
			}
		}
		float *test_mat = malloc(size*size*sizeof(float));
		STARPU_ASSERT(test_mat);

		STARPU_SSYRK("L", "N", size, size, 1.0f,
					mat, size, 0.0f, test_mat, size);

		FPRINTF(stderr, "comparing results ...\n");
#ifdef PRINT_OUTPUT
		for (m = 0; m < size; m++)
		{
			for (n = 0; n < size; n++)
			{
				if (n <= m)
				{
					FPRINTF(stdout, "%2.2f\t", test_mat[m +n*size]);
				}
				else
				{
					FPRINTF(stdout, ".\t");
				}
			}
			FPRINTF(stdout, "\n");
		}
#endif

		for (m = 0; m < size; m++)
		{
			for (n = 0; n < size; n++)
			{
				if (n <= m)
				{
	                                float orig = (1.0f/(1.0f+m+n)) + ((m == n)?1.0f*size:0.0f);
	                                float err = fabsf(test_mat[m +n*size] - orig) / orig;
	                                if (err > 0.0001)
					{
	                                        FPRINTF(stderr, "Error[%u, %u] --> %2.6f != %2.6f (err %2.6f)\n", m, n, test_mat[m +n*size], orig, err);
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
