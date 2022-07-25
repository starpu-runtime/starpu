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

#include "cholesky.h"
#include "../sched_ctx_utils/sched_ctx_utils.h"

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static void callback_turn_spmd_on(void *arg)
{
	(void)arg;
	cl_gemm.type = STARPU_SPMD;
}

int hypervisor_tag = 1;
static void _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;
	struct timeval start;
	struct timeval end;

	unsigned i,j,k;

	int prio_level = g_noprio?STARPU_DEFAULT_PRIO:STARPU_MAX_PRIO;

	gettimeofday(&start, NULL);

	if (g_bound)
		starpu_bound_start(0, 0);
	/* create all the DAG nodes */
	for (k = 0; k < nblocks; k++)
	{
                starpu_data_handle_t sdatakk = starpu_data_get_sub_data(dataA, 2, k, k);
		if(k == 0 && g_with_ctxs)
		{
			 ret = starpu_task_insert(&cl_potrf,
						  STARPU_PRIORITY, prio_level,
						  STARPU_RW, sdatakk,
						  STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
						  STARPU_HYPERVISOR_TAG, hypervisor_tag,
						  0);
			set_hypervisor_conf(START_BENCH, hypervisor_tag++);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		else
			starpu_task_insert(&cl_potrf,
					   STARPU_PRIORITY, prio_level,
					   STARPU_RW, sdatakk,
					   STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
					   0);

		for (j = k+1; j<nblocks; j++)
		{
                        starpu_data_handle_t sdatakj = starpu_data_get_sub_data(dataA, 2, k, j);

                        ret = starpu_task_insert(&cl_trsm,
						 STARPU_PRIORITY, (j == k+1)?prio_level:STARPU_DEFAULT_PRIO,
						 STARPU_R, sdatakk,
						 STARPU_RW, sdatakj,
						 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
                                {
					starpu_data_handle_t sdataki = starpu_data_get_sub_data(dataA, 2, k, i);
					starpu_data_handle_t sdataij = starpu_data_get_sub_data(dataA, 2, i, j);

					if(k == (nblocks-2) && j == (nblocks-1) &&
					   i == (k + 1) && g_with_ctxs)
					{
						ret = starpu_task_insert(&cl_gemm,
									 STARPU_PRIORITY, ((i == k+1) && (j == k+1))?prio_level:STARPU_DEFAULT_PRIO,
									 STARPU_R, sdataki,
									 STARPU_R, sdatakj,
									 STARPU_RW, sdataij,
									 STARPU_HYPERVISOR_TAG, hypervisor_tag,
									 0);
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
						set_hypervisor_conf(END_BENCH, hypervisor_tag++);
					}

					else
					{
						ret = starpu_task_insert(&cl_gemm,
									 STARPU_PRIORITY, ((i == k+1) && (j == k+1))?prio_level:STARPU_DEFAULT_PRIO,
									 STARPU_R, sdataki,
									 STARPU_R, sdatakj,
									 STARPU_RW, sdataij,
									 0);
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					}

                   }
			}
		}
	}

	starpu_task_wait_for_all();
	if (g_bound)
		starpu_bound_stop();

	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	unsigned long n = starpu_matrix_get_nx(dataA);

	double flop = (1.0f*n*n*n)/3.0f;

	if(g_with_ctxs || g_with_noctxs || g_chole1 || g_chole2)
		update_sched_ctx_timing_results((flop/timing/1000.0f), (timing/1000000.0f));
	else
	{
		FPRINTF(stderr, "Computation took (in ms)\n");
		FPRINTF(stdout, "%2.2f\n", timing/1000);

		FPRINTF(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
		if (g_bound)
		{
			double res;
			starpu_bound_compute(&res, NULL, 0);
			FPRINTF(stderr, "Theoretical GFlops: %2.2f\n", (flop/res/1000000.0f));
		}
	}
}

static void cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks)
{
	starpu_data_handle_t dataA;

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

	_cholesky(dataA, nblocks);

	starpu_data_unregister(dataA);
}

static void execute_cholesky(float *pmat, unsigned size, unsigned nblocks)
{
	(void)pmat;
	float *mat;
	starpu_malloc((void **)&mat, (size_t)size*size*sizeof(float));

	unsigned i,j;
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

	cholesky(mat, size, size, nblocks);

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

	if (g_check)
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
	                                        FPRINTF(stderr, "Error[%u, %u] --> %2.2f != %2.2f (err %2.2f)\n", i, j, test_mat[j +i*size], orig, err);
	                                        assert(0);
	                                }
	                        }
			}
	        }
		free(test_mat);
	}
	starpu_free_noflag(mat, (size_t)size*size*sizeof(float));
}

int main(int argc, char **argv)
{
	int ret;

	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	parse_args(argc, argv);

	if(g_with_ctxs || g_with_noctxs || g_chole1 || g_chole2)
		parse_args_ctx(argc, argv);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

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

	if(g_with_ctxs)
	{
		construct_contexts();
		start_2benchs(execute_cholesky);
	}
	else if(g_with_noctxs)
		start_2benchs(execute_cholesky);
	else if(g_chole1)
		start_1stbench(execute_cholesky);
	else if(g_chole2)
		start_2ndbench(execute_cholesky);
	else
		execute_cholesky(NULL, g_size, g_nblocks);

	starpu_cublas_shutdown();
	starpu_shutdown();

	if(g_with_ctxs)
		end_contexts();

	return 0;
}
