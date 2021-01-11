/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "cholesky.h"
#include "../sched_ctx_utils/sched_ctx_utils.h"

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#endif

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static void callback_turn_spmd_on(void *arg STARPU_ATTRIBUTE_UNUSED)
{
	cl22.type = STARPU_SPMD;
}

static int _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	double start;
	double end;

	unsigned i,j,k;
	unsigned long n = starpu_matrix_get_nx(dataA);
	unsigned long nn = n/nblocks;

	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	if (bound || bound_lp || bound_mps)
		starpu_bound_start(bound_deps, 0);

	starpu_fxt_start_profiling();

	start = starpu_timing_now();

	/* create all the DAG nodes */
	for (k = 0; k < nblocks; k++)
	{
		int ret;
		starpu_iteration_push(k);
                starpu_data_handle_t sdatakk = starpu_data_get_sub_data(dataA, 2, k, k);

                ret = starpu_task_insert(&cl11,
					 STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k) : STARPU_MAX_PRIO,
					 STARPU_RW, sdatakk,
					 STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
					 STARPU_FLOPS, (double) FLOPS_SPOTRF(nn),
					 STARPU_TAG_ONLY, TAG11(k),
					 0);
		if (ret == -ENODEV) return 77;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		for (j = k+1; j<nblocks; j++)
		{
                        starpu_data_handle_t sdatakj = starpu_data_get_sub_data(dataA, 2, k, j);

                        ret = starpu_task_insert(&cl21,
						 STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - j) : (j == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						 STARPU_R, sdatakk,
						 STARPU_RW, sdatakj,
						 STARPU_FLOPS, (double) FLOPS_STRSM(nn, nn),
						 STARPU_TAG_ONLY, TAG21(k,j),
						 0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		starpu_data_wont_use(sdatakk);

		for (j = k+1; j<nblocks; j++)
		{
                        starpu_data_handle_t sdatakj = starpu_data_get_sub_data(dataA, 2, k, j);
			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
                                {
					starpu_data_handle_t sdataki = starpu_data_get_sub_data(dataA, 2, k, i);
					starpu_data_handle_t sdataij = starpu_data_get_sub_data(dataA, 2, i, j);

					ret = starpu_task_insert(&cl22,
								 STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - j - i) : ((i == k+1) && (j == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
								 STARPU_R, sdataki,
								 STARPU_R, sdatakj,
								 cl22.modes[2], sdataij,
								 STARPU_FLOPS, (double) FLOPS_SGEMM(nn, nn, nn),
								 STARPU_TAG_ONLY, TAG22(k,i,j),
								 0);
					if (ret == -ENODEV) return 77;
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
                                }
			}
			starpu_data_wont_use(sdatakj);
		}
		starpu_iteration_pop();
	}

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	starpu_fxt_stop_profiling();
	if (bound || bound_lp || bound_mps)
		starpu_bound_stop();

	double timing = end - start;

	double flop = FLOPS_SPOTRF(n);

	if(with_ctxs || with_noctxs || chole1 || chole2)
		update_sched_ctx_timing_results((flop/timing/1000.0f), (timing/1000000.0f));
	else
	{
		PRINTF("# size\tms\tGFlops");
		if (bound)
			PRINTF("\tTms\tTGFlops");
		PRINTF("\n");

		PRINTF("%lu\t%.0f\t%.1f", n, timing/1000, (flop/timing/1000.0f));
		if (bound_lp)
		{
			FILE *f = fopen("cholesky.lp", "w");
			starpu_bound_print_lp(f);
		}
		if (bound_mps)
		{
			FILE *f = fopen("cholesky.mps", "w");
			starpu_bound_print_mps(f);
		}
		if (bound)
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
	unsigned i,j;

#ifndef STARPU_SIMGRID
	starpu_malloc_flags((void **)&mat, (size_t)size*size*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			/* mat[j +i*size] = ((i == j)?1.0f*size:0.0f); */
		}
	}
#endif

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

	if (check)
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
	                                float err = abs(test_mat[j +i*size] - orig) / orig;
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

	if(with_ctxs || with_noctxs || chole1 || chole2)
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

	if(with_ctxs)
	{
		construct_contexts(execute_cholesky);
		start_2benchs(execute_cholesky);
	}
	else if(with_noctxs)
		start_2benchs(execute_cholesky);
	else if(chole1)
		start_1stbench(execute_cholesky);
	else if(chole2)
		start_2ndbench(execute_cholesky);
	else
		execute_cholesky(size, nblocks);

	starpu_cublas_shutdown();
	starpu_shutdown();

	return 0;
}
