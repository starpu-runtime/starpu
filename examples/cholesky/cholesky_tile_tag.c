/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * It also directly registers matrix tiles instead of using partitioning.
 */

/* Note: this is using fortran ordering, i.e. column-major ordering, i.e.
 * elements with consecutive row number are consecutive in memory */

#include "cholesky.h"

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#endif

#include "starpu_cusolver.h"

/* A [ m ] [ n ] */
float *A[NMAXBLOCKS][NMAXBLOCKS];
starpu_data_handle_t A_state[NMAXBLOCKS][NMAXBLOCKS];

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

static struct starpu_task * create_task_potrf(unsigned k, unsigned nblocks)
{
	(void)nblocks;
	/*	FPRINTF(stdout, "task potrf k = %d TAG = %llx\n", k, (TAG_POTRF(k))); */

	struct starpu_task *task = create_task(TAG_POTRF(k));

	task->cl = &cl_potrf;

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[k][k];

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_LIBCUSOLVER)
	/* Temporary data to save libcusolver from allocating/deallocating memory */
	task->handles[1] = scratch;
#endif

	/* this is an important task */
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

static int create_task_trsm(unsigned k, unsigned m)
{
	int ret;

	struct starpu_task *task = create_task(TAG_TRSM(m, k));

	task->cl = &cl_trsm;

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[k][k];
	task->handles[1] = A_state[m][k];

	if (m == k+1)
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM(m, k), 2, TAG_POTRF(k), TAG_GEMM(k-1, m, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM(m, k), 1, TAG_POTRF(k));
	}

	int n = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_STRSM(n, n);

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_gemm(unsigned k, unsigned m, unsigned n)
{
	int ret;

/*	FPRINTF(stdout, "task gemm k,n,m = %d,%d,%d TAG = %llx\n", k,m,n, TAG_GEMM(k,m,n)); */

	struct starpu_task *task = create_task(TAG_GEMM(k, m, n));

	if (m == n)
	{
		task->cl = &cl_syrk;

		/* which sub-data is manipulated ? */
		task->handles[0] = A_state[n][k];
		task->handles[1] = A_state[n][n];
		int nx = starpu_matrix_get_nx(task->handles[0]);
		task->flops = FLOPS_SSYRK(nx, nx);
	}
	else
	{
		task->cl = &cl_gemm;

		/* which sub-data is manipulated ? */
		task->handles[0] = A_state[n][k];
		task->handles[1] = A_state[m][k];
		task->handles[2] = A_state[m][n];
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
		starpu_tag_declare_deps(TAG_GEMM(k, m, n), 3, TAG_GEMM(k-1, m, n), TAG_TRSM(n, k), TAG_TRSM(m, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG_GEMM(k, m, n), 2, TAG_TRSM(n, k), TAG_TRSM(m, k));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static int cholesky_no_stride(void)
{
	int ret;

	double start;
	double end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned k, m, n;

	for (k = 0; k < nblocks_p; k++)
	{
		starpu_iteration_push(k);
		struct starpu_task *task = create_task_potrf(k, nblocks_p);
		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			ret = starpu_task_submit(task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (m = k+1; m<nblocks_p; m++)
		{
			ret = create_task_trsm(k, m);
			if (ret == -ENODEV) return 77;

			for (n = k+1; n<nblocks_p; n++)
			{
				if (n <= m)
				{
					ret = create_task_gemm(k, m, n);
					if (ret == -ENODEV) return 77;
				}
			}
		}
		starpu_iteration_pop();
	}

	/* schedule the codelet */
	start = starpu_timing_now();
	ret = starpu_task_submit(entry_task);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* stall the application until the end of computations */
	starpu_tag_wait(TAG_POTRF(nblocks_p-1));

	end = starpu_timing_now();

	double timing = end - start;

	double flop = (1.0f*size_p*size_p*size_p)/3.0f;
	PRINTF("# size\tms\tGFlop/s\n");
	PRINTF("%u\t%.0f\t%.1f\n", size_p, timing/1000, (flop/timing/1000.0f));

	return 0;
}

int main(int argc, char **argv)
{
	unsigned n, m;
	int ret;

#ifdef STARPU_HAVE_MAGMA
	magma_init();
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	init_sizes();

	parse_args(argc, argv);
	assert(nblocks_p <= NMAXBLOCKS);

	FPRINTF(stderr, "BLOCK SIZE = %u\n", size_p / nblocks_p);

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

	/* Disable sequential consistency */
	starpu_data_set_default_sequential_consistency_flag(0);

	starpu_cublas_init();
	starpu_cusolver_init();

	for (m = 0; m < nblocks_p; m++)
	for (n = 0; n < nblocks_p; n++)
	{
		if (n <= m)
		{
			starpu_malloc_flags((void **)&A[m][n], BLOCKSIZE*BLOCKSIZE*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED|STARPU_MALLOC_SIMULATION_UNIQUE);
			assert(A[m][n]);
		}
	}

#ifndef STARPU_SIMGRID
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1) (+ n In to make is stable)
	 * */
	for (m = 0; m < nblocks_p; m++)
	for (n = 0; n < nblocks_p; n++)
	if (n <= m)
	{
		unsigned mm, nn;
		for (mm = 0; mm < BLOCKSIZE; mm++)
		for (nn = 0; nn < BLOCKSIZE; nn++)
		{
			A[m][n][mm*BLOCKSIZE + nn] =
				(float)(1.0f/((float) (1.0+(n*BLOCKSIZE+mm)+(m*BLOCKSIZE+nn))));

			/* make it a little more numerically stable ... ;) */
			if ((n == m) && (mm == nn))
				A[m][n][mm*BLOCKSIZE + nn] += (float)(2*size_p);
		}
	}
#endif

	for (m = 0; m < nblocks_p; m++)
	for (n = 0; n < nblocks_p; n++)
	{
		if (n <= m)
		{
			starpu_matrix_data_register(&A_state[m][n], STARPU_MAIN_RAM, (uintptr_t)A[m][n],
						    BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, sizeof(float));
			starpu_data_set_coordinates(A_state[m][n], 2, n, m);
		}
	}

	cholesky_kernel_init(BLOCKSIZE);

	ret = cholesky_no_stride();

	cholesky_kernel_fini();

	for (m = 0; m < nblocks_p; m++)
	for (n = 0; n < nblocks_p; n++)
	{
		if (n <= m)
		{
			starpu_data_unregister(A_state[m][n]);
			starpu_free_flags(A[m][n], BLOCKSIZE*BLOCKSIZE*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED|STARPU_MALLOC_SIMULATION_UNIQUE);
		}
	}

	starpu_cusolver_shutdown();
	starpu_cublas_shutdown();

	starpu_shutdown();
	return ret;
}
