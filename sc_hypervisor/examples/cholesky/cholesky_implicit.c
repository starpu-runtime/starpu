/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010	    Mehdi Juhoor
 * Copyright (C) 2013	    Thibaut Lambert
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
static int _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;
	double start;
	double end;

	unsigned k,m,n;
	unsigned long nx = starpu_matrix_get_nx(dataA);
	unsigned long nn = nx/nblocks;

	int prio_level = g_noprio?STARPU_DEFAULT_PRIO:STARPU_MAX_PRIO;

	if (g_bound)
		starpu_bound_start(0, 0);

	start = starpu_timing_now();

	/* create all the DAG nodes */
	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);
		starpu_data_handle_t sdatakk = starpu_data_get_sub_data(dataA, 2, k, k);
		if(k == 0 && g_with_ctxs)
		{
			 ret = starpu_task_insert(&cl_potrf,
						  STARPU_PRIORITY, prio_level,
						  STARPU_RW, sdatakk,
						  STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
						  STARPU_HYPERVISOR_TAG, hypervisor_tag,
						  0);
			if (ret == -ENODEV) return 77;
			set_hypervisor_conf(START_BENCH, hypervisor_tag++);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		else
		{
			ret = starpu_task_insert(&cl_potrf,
					   STARPU_PRIORITY, prio_level,
					   STARPU_RW, sdatakk,
					   STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
					   STARPU_FLOPS, (double) FLOPS_SPOTRF(nn),
					   0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}

		for (m = k+1; m<nblocks; m++)
		{
			starpu_data_handle_t sdatamk = starpu_data_get_sub_data(dataA, 2, m, k);

			ret = starpu_task_insert(&cl_trsm,
						 STARPU_PRIORITY, (m == k+1)?prio_level:STARPU_DEFAULT_PRIO,
						 STARPU_R, sdatakk,
						 STARPU_RW, sdatamk,
						 STARPU_FLOPS, (double) FLOPS_STRSM(nn, nn),
						 0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		starpu_data_wont_use(sdatakk);

		for (n = k+1; n<nblocks; n++)
		{
			starpu_data_handle_t sdatank = starpu_data_get_sub_data(dataA, 2, n, k);

			for (m = n; m<nblocks; m++)
			{
				starpu_data_handle_t sdatamk = starpu_data_get_sub_data(dataA, 2, m, k);
				starpu_data_handle_t sdatamn = starpu_data_get_sub_data(dataA, 2, m, n);

				if(k == (nblocks-2) && m == (nblocks-1) &&
				   n == (k + 1) && g_with_ctxs)
				{
					ret = starpu_task_insert(&cl_gemm,
								 STARPU_PRIORITY, ((n == k+1) && (m == k+1))?prio_level:STARPU_DEFAULT_PRIO,
								 STARPU_R, sdatank,
								 STARPU_R, sdatamk,
								 STARPU_RW, sdatamn,
								 STARPU_HYPERVISOR_TAG, hypervisor_tag,
								 0);
					if (ret == -ENODEV) return 77;
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					set_hypervisor_conf(END_BENCH, hypervisor_tag++);
				}

				else
				{
					ret = starpu_task_insert(&cl_gemm,
								 STARPU_PRIORITY, ((n == k+1) && (m == k+1))?prio_level:STARPU_DEFAULT_PRIO,
								 STARPU_R, sdatamk,
								 STARPU_R, sdatank,
								 STARPU_RW, sdatamn,
								 STARPU_FLOPS, (double) FLOPS_SGEMM(nn, nn, nn),
								 0);
					if (ret == -ENODEV) return 77;
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
				}

			}
			starpu_data_wont_use(sdatank);
		}
		starpu_iteration_pop();
	}

	starpu_task_wait_for_all();
	if (g_bound)
		starpu_bound_stop();

	end = starpu_timing_now();

	double timing = end - start;

	double flop = (1.0f*nx*nx*nx)/3.0f;

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

static void execute_cholesky(float *pmat, unsigned size, unsigned nblocks)
{
	(void)pmat;
	float *mat;
	starpu_malloc((void **)&mat, (size_t)size*size*sizeof(float));

	unsigned m,n;
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

	cholesky(mat, size, size, nblocks);

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
				mat[m+n*size] = 0.0f; /* debug */
			}
		}
		FPRINTF(stdout, "\n");
	}
#endif

	if (g_check)
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
