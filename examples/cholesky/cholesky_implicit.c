/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This version of the Cholesky factorization uses implicit dependency computation.
 * The whole algorithm thus appears clearly in the task submission loop in _cholesky().
 */

/* Note: this is using fortran ordering, i.e. column-major ordering, i.e.
 * elements with consecutive row number are consecutive in memory */

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
	cl_gemm.type = STARPU_SPMD;
}

static int _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	double start;
	double end;

	unsigned k,m,n;
	unsigned long nx = starpu_matrix_get_nx(dataA);
	unsigned long nn = nx/nblocks;

	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	if (bound_p || bound_lp_p || bound_mps_p)
		starpu_bound_start(bound_deps_p, 0);
	starpu_fxt_start_profiling();

	start = starpu_timing_now();

	/* create all the DAG nodes */
	for (k = 0; k < nblocks; k++)
	{
		int ret;
		starpu_iteration_push(k);
                starpu_data_handle_t sdatakk = starpu_data_get_sub_data(dataA, 2, k, k);

                ret = starpu_task_insert(&cl_potrf,
					 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k) : STARPU_MAX_PRIO,
					 STARPU_RW, sdatakk,
					 STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
					 STARPU_FLOPS, (double) FLOPS_SPOTRF(nn),
					 STARPU_NAME, "POTRF",
					 STARPU_TAG_ONLY, TAG_POTRF(k),
					 0);
		if (ret == -ENODEV) return 77;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		for (m = k+1; m<nblocks; m++)
		{
                        starpu_data_handle_t sdatamk = starpu_data_get_sub_data(dataA, 2, m, k);

                        ret = starpu_task_insert(&cl_trsm,
						 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						 STARPU_R, sdatakk,
						 STARPU_RW, sdatamk,
						 STARPU_FLOPS, (double) FLOPS_STRSM(nn, nn),
						 STARPU_NAME, "TRSM",
						 STARPU_TAG_ONLY, TAG_TRSM(m,k),
						 0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		starpu_data_wont_use(sdatakk);

		for (n = k+1; n<nblocks; n++)
		{
                        starpu_data_handle_t sdatank = starpu_data_get_sub_data(dataA, 2, n, k);
			starpu_data_handle_t sdatann = starpu_data_get_sub_data(dataA, 2, n, n);

			ret = starpu_task_insert(&cl_syrk,
						 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - n - n) : (n == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						 STARPU_R, sdatank,
						 cl_syrk.modes[1], sdatann,
						 STARPU_FLOPS, (double) FLOPS_SSYRK(nn, nn),
						 STARPU_NAME, "SYRK",
						 STARPU_TAG_ONLY, TAG_GEMM(k,n,n),
						 0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

			for (m = n+1; m<nblocks; m++)
			{
				starpu_data_handle_t sdatamk = starpu_data_get_sub_data(dataA, 2, m, k);
				starpu_data_handle_t sdatamn = starpu_data_get_sub_data(dataA, 2, m, n);

				ret = starpu_task_insert(&cl_gemm,
							 STARPU_PRIORITY, noprio_p ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
							 STARPU_R, sdatamk,
							 STARPU_R, sdatank,
							 cl_gemm.modes[2], sdatamn,
							 STARPU_FLOPS, (double) FLOPS_SGEMM(nn, nn, nn),
							 STARPU_NAME, "GEMM",
							 STARPU_TAG_ONLY, TAG_GEMM(k,m,n),
							 0);
				if (ret == -ENODEV) return 77;
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
			}
			starpu_data_wont_use(sdatank);
		}
		starpu_iteration_pop();
	}

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	starpu_fxt_stop_profiling();
	if (bound_p || bound_lp_p || bound_mps_p)
		starpu_bound_stop();

	double timing = end - start;

	double flop = FLOPS_SPOTRF(nx);

	if(with_ctxs_p || with_noctxs_p || chole1_p || chole2_p)
		update_sched_ctx_timing_results((flop/timing/1000.0f), (timing/1000000.0f));
	else
	{
		PRINTF("# size\tms\tGFlop/s");
		if (bound_p)
			PRINTF("\tTms\tTGFlop/s");
		PRINTF("\n");

		PRINTF("%lu\t%.0f\t%.1f", nx, timing/1000, (flop/timing/1000.0f));
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
	starpu_data_set_name(dataA, "A");

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
			starpu_data_set_name(data, "subA");
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

	/*
	 * create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 *
	 * and make it better conditioned by adding one on the diagonal.
	 */

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
	initialize_chol_model(&chol_model_potrf,"chol_model_potrf",cpu_chol_task_potrf_cost,cuda_chol_task_potrf_cost);
	initialize_chol_model(&chol_model_trsm,"chol_model_trsm",cpu_chol_task_trsm_cost,cuda_chol_task_trsm_cost);
	initialize_chol_model(&chol_model_syrk,"chol_model_syrk",cpu_chol_task_syrk_cost,cuda_chol_task_syrk_cost);
	initialize_chol_model(&chol_model_gemm,"chol_model_gemm",cpu_chol_task_gemm_cost,cuda_chol_task_gemm_cost);
#else
	initialize_chol_model(&chol_model_potrf,"chol_model_potrf",cpu_chol_task_potrf_cost,NULL);
	initialize_chol_model(&chol_model_trsm,"chol_model_trsm",cpu_chol_task_trsm_cost,NULL);
	initialize_chol_model(&chol_model_syrk,"chol_model_syrk",cpu_chol_task_syrk_cost,NULL);
	initialize_chol_model(&chol_model_gemm,"chol_model_gemm",cpu_chol_task_gemm_cost,NULL);
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
