/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This version of the Cholesky factorization uses explicit dependency
 * declaration through dependency tags.
 * It also uses data partitioning to split the matrix into submatrices
 */

/* Note: this is using fortran ordering, i.e. column-major ordering, i.e.
 * elements with consecutive row number are consecutive in memory */

#include "cholesky.h"
#include <starpu_perfmodel.h>

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#endif

#include <starpu_cusolver.h>

/*
 *	Some useful functions
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;
		task->use_tag = 1;
		task->tag_id = id;

	return task;
}

/*
 *	Create the codelets
 */

static struct starpu_task * create_task_potrf(starpu_data_handle_t dataA, unsigned k)
{
/*	FPRINTF(stdout, "task potrf k = %d TAG = %llx\n", k, (TAG_POTRF(k))); */

	struct starpu_task *task = create_task(TAG_POTRF(k));

	task->cl = &cl_potrf;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_LIBCUSOLVER)
	/* Temporary data to save libcusolver from allocating/deallocating memory */
	task->handles[1] = scratch;
#endif

	/* this is an important task */
	if (!noprio_p)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_POTRF(k), 1, TAG_GEMM(k-1, k, k));
	}

	int n = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_SPOTRF(n);

	return task;
}

static int create_task_trsm(starpu_data_handle_t dataA, unsigned k, unsigned m)
{
	int ret;

	struct starpu_task *task = create_task(TAG_TRSM(k, m));

	task->cl = &cl_trsm;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, m, k);

	if (!noprio_p && (m == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM(k, m), 2, TAG_POTRF(k), TAG_GEMM(k-1, m, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM(k, m), 1, TAG_POTRF(k));
	}

	int nx = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_STRSM(nx, nx);

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_gemm(starpu_data_handle_t dataA, unsigned k, unsigned m, unsigned n)
{
	int ret;

/*	FPRINTF(stdout, "task gemm k,n,m = %d,%d,%d TAG = %llx\n", k,m,n, TAG_GEMM(k,m,n)); */

	struct starpu_task *task = create_task(TAG_GEMM(k, m, n));

	if (m == n)
	{
		task->cl = &cl_syrk;

		/* which sub-data is manipulated ? */
		task->handles[0] = starpu_data_get_sub_data(dataA, 2, n, k);
		task->handles[1] = starpu_data_get_sub_data(dataA, 2, n, n);
		int nx = starpu_matrix_get_nx(task->handles[0]);
		task->flops = FLOPS_SSYRK(nx, nx);
	}
	else
	{
		task->cl = &cl_gemm;

		/* which sub-data is manipulated ? */
		task->handles[0] = starpu_data_get_sub_data(dataA, 2, n, k);
		task->handles[1] = starpu_data_get_sub_data(dataA, 2, m, k);
		task->handles[2] = starpu_data_get_sub_data(dataA, 2, m, n);
		int nx = starpu_matrix_get_nx(task->handles[0]);
		task->flops = FLOPS_SGEMM(nx, nx, nx);
	}

	if (!noprio_p && (n == k + 1) && (m == k +1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GEMM(k, m, n), 3, TAG_GEMM(k-1, m, n), TAG_TRSM(k, n), TAG_TRSM(k, m));
	}
	else
	{
		starpu_tag_declare_deps(TAG_GEMM(k, m, n), 2, TAG_TRSM(k, n), TAG_TRSM(k, m));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static int _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;

	double start;
	double end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned k, m, n;

	start = starpu_timing_now();

	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);
		struct starpu_task *task = create_task_potrf(dataA, k);
		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			ret = starpu_task_submit(task);
			if (ret == -ENODEV)
			{
				starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
				return 77;
			}
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (m = k+1; m<nblocks; m++)
		{
			ret = create_task_trsm(dataA, k, m);
			if (ret == -ENODEV)
			{
				starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
				return 77;
			}

			for (n = k+1; n<nblocks; n++)
			{
				if (n <= m)
				{
					ret = create_task_gemm(dataA, k, m, n);
					if (ret == -ENODEV)
					{
						starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
						return 77;
					}
				}
			}
		}
		starpu_iteration_pop();
	}

	/* schedule the codelet */
	ret = starpu_task_submit(entry_task);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* stall the application until the end of computations */
	starpu_tag_wait(TAG_POTRF(nblocks-1));

	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);

	end = starpu_timing_now();

	double timing = end - start;

	unsigned nx = starpu_matrix_get_nx(dataA);

	double flop = (1.0f*nx*nx*nx)/3.0f;

	PRINTF("# size\tms\tGFlop/s\n");
	PRINTF("%u\t%.0f\t%.1f\n", nx, timing/1000, (flop/timing/1000.0f));

	return 0;
}

static int initialize_system(int argc, char **argv, float **A, unsigned pinned)
{
	int ret;
	int flags = STARPU_MALLOC_SIMULATION_FOLDED|STARPU_MALLOC_SIMULATION_UNIQUE;

#ifdef STARPU_HAVE_MAGMA
	magma_init();
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	init_sizes();

	parse_args(argc, argv);

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
	starpu_cusolver_init();

	if (pinned)
		flags |= STARPU_MALLOC_PINNED;
	starpu_malloc_flags((void **)A, (size_t)size_p*size_p*sizeof(float), flags);

	return 0;
}

static int cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks)
{
	starpu_data_handle_t dataA;
	int ret;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (m,n) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld, size, size, sizeof(float));

	starpu_data_set_sequential_consistency_flag(dataA, 0);

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

	cholesky_kernel_init(size / nblocks);

	ret = _cholesky(dataA, nblocks);

	cholesky_kernel_fini();

	starpu_data_unregister(dataA);
	return ret;
}

static void shutdown_system(float **matA, unsigned dim, unsigned pinned)
{
	int flags = STARPU_MALLOC_SIMULATION_FOLDED|STARPU_MALLOC_SIMULATION_UNIQUE;
	if (pinned)
		flags |= STARPU_MALLOC_PINNED;

	starpu_free_flags(*matA, (size_t)dim*dim*sizeof(float), flags);

	starpu_cusolver_shutdown();
	starpu_cublas_shutdown();
	starpu_shutdown();
}

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	float *mat = NULL;
	int ret = initialize_system(argc, argv, &mat, pinned_p);
	if (ret) return ret;

#ifndef STARPU_SIMGRID
	unsigned long long m,n;

	for (n = 0; n < size_p; n++)
	{
		for (m = 0; m < size_p; m++)
		{
			mat[m +n*size_p] = (1.0f/(1.0f+n+m)) + ((n == m)?1.0f*size_p:0.0f);
			/* mat[m +n*size_p] = ((n == m)?1.0f*size_p:0.0f); */
		}
	}

/* #define PRINT_OUTPUT */
#ifdef PRINT_OUTPUT
	FPRINTF(stdout, "Input :\n");

	for (m = 0; m < size_p; m++)
	{
		for (n = 0; n < size_p; n++)
		{
			if (n <= m)
			{
				FPRINTF(stdout, "%2.2f\t", mat[m +n*size_p]);
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

	ret = cholesky(mat, size_p, size_p, nblocks_p);

#ifndef STARPU_SIMGRID
#ifdef PRINT_OUTPUT
	FPRINTF(stdout, "Results :\n");

	for (m = 0; m < size_p; m++)
	{
		for (n = 0; n < size_p; n++)
		{
			if (n <= m)
			{
				FPRINTF(stdout, "%2.2f\t", mat[m +n*size_p]);
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
		for (m = 0; m < size_p; m++)
		{
			for (n = 0; n < size_p; n++)
			{
				if (n > m)
				{
					mat[m+n*size_p] = 0.0f; /* debug */
				}
			}
		}
		float *test_mat = malloc(size_p*size_p*sizeof(float));
		STARPU_ASSERT(test_mat);

		STARPU_SSYRK("L", "N", size_p, size_p, 1.0f,
			     mat, size_p, 0.0f, test_mat, size_p);

		FPRINTF(stderr, "comparing results ...\n");
#ifdef PRINT_OUTPUT
		for (m = 0; m < size_p; m++)
		{
			for (n = 0; n < size_p; n++)
			{
				if (n <= m)
				{
					FPRINTF(stdout, "%2.2f\t", test_mat[m +n*size_p]);
				}
				else
				{
					FPRINTF(stdout, ".\t");
				}
			}
			FPRINTF(stdout, "\n");
		}
#endif

		for (m = 0; m < size_p; m++)
		{
			for (n = 0; n < size_p; n++)
			{
				if (n <= m)
				{
	                                float orig = (1.0f/(1.0f+m+n)) + ((m == n)?1.0f*size_p:0.0f);
	                                float err = fabsf(test_mat[m +n*size_p] - orig) / orig;
	                                if (err > 0.0001)
					{
						FPRINTF(stderr, "Error[%llu, %llu] --> %2.6f != %2.6f (err %2.6f)\n", m, n, test_mat[m +n*size_p], orig, err);
	                                        assert(0);
	                                }
	                        }
			}
	        }
		free(test_mat);
	}
#endif

	shutdown_system(&mat, size_p, pinned_p);
	return ret;
}
