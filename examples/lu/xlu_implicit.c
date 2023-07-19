/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

/* LU StarPU implementation using implicit task dependencies. */

#include "xlu.h"
#include "xlu_kernels.h"
#include "starpu_cusolver.h"

static int create_task_getrf(starpu_data_handle_t dataA, unsigned k, unsigned no_prio, int nblocks)
{
	int ret;
	struct starpu_task *task = starpu_task_create();
	task->cl = &cl_getrf;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_LIBCUSOLVER)
	task->handles[1] = scratch;
#endif

	task->tag_id = TAG_GETRF(k);
	task->color = 0xffff00;

	/* this is an important task */
	if (!no_prio)
		task->priority = 3*nblocks - 3*k; /* Bottom-level-based prio */

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_trsm_ll(starpu_data_handle_t dataA, unsigned k, unsigned j, unsigned no_prio, int nblocks)
{
	int ret;
	struct starpu_task *task = starpu_task_create();
	task->cl = &cl_trsm_ll;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, j, k);

	task->tag_id = TAG_TRSM_LL(k,j);
	task->color = 0x8080ff;

	if (!no_prio)
		task->priority = 3*nblocks - (2*k + j); /* Bottom-level-based prio */

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_trsm_ru(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned no_prio, int nblocks)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl_trsm_ru;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, i);

	task->tag_id = TAG_TRSM_RU(k,i);
	task->color = 0x8080c0;

	if (!no_prio)
		task->priority = 3*nblocks - (2*k + i); /* Bottom-level-based prio */

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_gemm(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned j, unsigned no_prio, int nblocks)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl_gemm;
	task->color = 0x00ff00;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, i);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, j, k);
	task->handles[2] = starpu_data_get_sub_data(dataA, 2, j, i);

	task->tag_id = TAG_GEMM(k,i,j);

	if (!no_prio)
		task->priority = 3*nblocks - (k + i + j); /* Bottom-level-based prio */

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

/*
 *	code to bootstrap the factorization
 */

static int dw_codelet_facto_v3(starpu_data_handle_t dataA, unsigned nblocks, unsigned no_prio)
{
	double start;
	double end;

	/* create all the DAG nodes */
	unsigned i,j,k;

	if (bound)
		starpu_bound_start(bounddeps, boundprio);

	start = starpu_timing_now();

	for (k = 0; k < nblocks; k++)
	{
		int ret;

		starpu_iteration_push(k);

		ret = create_task_getrf(dataA, k, no_prio, nblocks);
		if (ret == -ENODEV) return ret;

		for (i = k+1; i<nblocks; i++)
		{
			ret = create_task_trsm_ll(dataA, k, i, no_prio, nblocks);
			if (ret == -ENODEV) return ret;
			ret = create_task_trsm_ru(dataA, k, i, no_prio, nblocks);
			if (ret == -ENODEV) return ret;
		}
		starpu_data_wont_use(starpu_data_get_sub_data(dataA, 2, k, k));

		for (i = k+1; i<nblocks; i++)
		     for (j = k+1; j<nblocks; j++)
		     {
			     ret = create_task_gemm(dataA, k, i, j, no_prio, nblocks);
			     if (ret == -ENODEV) return ret;
		     }
		for (i = k+1; i<nblocks; i++)
		{
			starpu_data_wont_use(starpu_data_get_sub_data(dataA, 2, k, i));
			starpu_data_wont_use(starpu_data_get_sub_data(dataA, 2, i, k));
		}
		starpu_iteration_pop();
	}

	/* stall the application until the end of computations */
	starpu_task_wait_for_all();

	end = starpu_timing_now();

	if (bound)
		starpu_bound_stop();

	double timing = end - start;
	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;

	PRINTF("# size\tms\tGFlop/s");
	if (bound)
		PRINTF("\tTms\tTGFlop/s");
	PRINTF("\n");
	PRINTF("%u\t%.0f\t%.1f", n, timing/1000, flop/timing/1000.0f);
	if (bound)
	{
		double min;
		starpu_bound_compute(&min, NULL, 0);
		PRINTF("\t%.0f\t%.1f", min, flop/min/1000000.0f);
	}
	PRINTF("\n");

	return 0;
}

int STARPU_LU(lu_decomposition)(TYPE *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned no_prio)
{
	starpu_data_handle_t dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld, size, size, sizeof(TYPE));

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

	lu_kernel_init(size / nblocks);

	int ret = dw_codelet_facto_v3(dataA, nblocks, no_prio);

	lu_kernel_fini();

	/* gather all the data */
	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
	starpu_data_unregister(dataA);
	return ret;
}
