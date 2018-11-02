/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2017                                Université de Bordeaux
 * Copyright (C) 2012,2013                                Inria
 * Copyright (C) 2010-2013,2015-2017                      CNRS
 * Copyright (C) 2013                                     Thibaut Lambert
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

#include "cholesky.h"

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#endif

/* A [ y ] [ x ] */
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

static struct starpu_task * create_task_11(unsigned k, unsigned nblocks)
{
	(void)nblocks;
	/*	FPRINTF(stdout, "task 11 k = %d TAG = %llx\n", k, (TAG11(k))); */

	struct starpu_task *task = create_task(TAG11(k));

	task->cl = &cl11;

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[k][k];

	/* this is an important task */
	task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	int n = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_SPOTRF(n);

	return task;
}

static int create_task_21(unsigned k, unsigned j)
{
	int ret;

	struct starpu_task *task = create_task(TAG21(k, j));

	task->cl = &cl21;

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[k][k];
	task->handles[1] = A_state[j][k];

	if (j == k+1)
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG21(k, j), 2, TAG11(k), TAG22(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG21(k, j), 1, TAG11(k));
	}

	int n = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_STRSM(n, n);

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_22(unsigned k, unsigned i, unsigned j)
{
	int ret;

/*	FPRINTF(stdout, "task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j)); */

	struct starpu_task *task = create_task(TAG22(k, i, j));

	task->cl = &cl22;

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[i][k];
	task->handles[1] = A_state[j][k];
	task->handles[2] = A_state[j][i];

	if ( (i == k + 1) && (j == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG21(k, i), TAG21(k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG22(k, i, j), 2, TAG21(k, i), TAG21(k, j));
	}

	int n = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_SGEMM(n, n, n);

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
	unsigned i,j,k;

	for (k = 0; k < nblocks_p; k++)
	{
		starpu_iteration_push(k);
		struct starpu_task *task = create_task_11(k, nblocks_p);
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

		for (j = k+1; j<nblocks_p; j++)
		{
			ret = create_task_21(k, j);
			if (ret == -ENODEV) return 77;

			for (i = k+1; i<nblocks_p; i++)
			{
				if (i <= j)
				{
				     ret = create_task_22(k, i, j);
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
	starpu_tag_wait(TAG11(nblocks_p-1));

	end = starpu_timing_now();

	double timing = end - start;

	double flop = (1.0f*size_p*size_p*size_p)/3.0f;
	PRINTF("# size\tms\tGFlops\n");
	PRINTF("%u\t%.0f\t%.1f\n", size_p, timing/1000, (flop/timing/1000.0f));

	return 0;
}

int main(int argc, char **argv)
{
	unsigned x, y;
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
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,cuda_chol_task_11_cost);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,cuda_chol_task_21_cost);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,cuda_chol_task_22_cost);
#else
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,NULL);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,NULL);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,NULL);
#endif

	/* Disable sequential consistency */
	starpu_data_set_default_sequential_consistency_flag(0);

	starpu_cublas_init();

	for (y = 0; y < nblocks_p; y++)
	for (x = 0; x < nblocks_p; x++)
	{
		if (x <= y)
		{
			starpu_malloc_flags((void **)&A[y][x], BLOCKSIZE*BLOCKSIZE*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
			assert(A[y][x]);
		}
	}

#ifndef STARPU_SIMGRID
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1) ( + n In to make is stable )
	 * */
	for (y = 0; y < nblocks_p; y++)
	for (x = 0; x < nblocks_p; x++)
	if (x <= y)
	{
		unsigned i, j;
		for (i = 0; i < BLOCKSIZE; i++)
		for (j = 0; j < BLOCKSIZE; j++)
		{
			A[y][x][i*BLOCKSIZE + j] =
				(float)(1.0f/((float) (1.0+(x*BLOCKSIZE+i)+(y*BLOCKSIZE+j))));

			/* make it a little more numerically stable ... ;) */
			if ((x == y) && (i == j))
				A[y][x][i*BLOCKSIZE + j] += (float)(2*size_p);
		}
	}
#endif

	for (y = 0; y < nblocks_p; y++)
	for (x = 0; x < nblocks_p; x++)
	{
		if (x <= y)
		{
			starpu_matrix_data_register(&A_state[y][x], STARPU_MAIN_RAM, (uintptr_t)A[y][x],
						    BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, sizeof(float));
			starpu_data_set_coordinates(A_state[y][x], 2, x, y);
		}
	}

	ret = cholesky_no_stride();

	for (y = 0; y < nblocks_p; y++)
	for (x = 0; x < nblocks_p; x++)
	{
		if (x <= y)
		{
			starpu_data_unregister(A_state[y][x]);
			starpu_free_flags(A[y][x], BLOCKSIZE*BLOCKSIZE*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
		}
	}

	starpu_cublas_shutdown();

	starpu_shutdown();
	return ret;
}
