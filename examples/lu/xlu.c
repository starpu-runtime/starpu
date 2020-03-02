/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* LU StarPU implementation using explicit tag dependencies */
#include "xlu.h"
#include "xlu_kernels.h"

/*
 *	Construct the DAG
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;

	task->use_tag = 1;
	task->tag_id = id;

	return task;
}

static struct starpu_task *create_task_11(starpu_data_handle_t dataA, unsigned k, unsigned no_prio)
{
/*	printf("task 11 k = %d TAG = %llx\n", k, (TAG11(k))); */

	struct starpu_task *task = create_task(TAG11(k));

	task->cl = &cl11;
	task->color = 0xffff00;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	return task;
}

static int create_task_12(starpu_data_handle_t dataA, unsigned k, unsigned j, unsigned no_prio)
{
	int ret;

/*	printf("task 12 k,i = %d,%d TAG = %llx\n", k,i, TAG12(k,i)); */

	struct starpu_task *task = create_task(TAG12(k, j));

	task->cl = &cl12;
	task->color = 0x8080ff;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, j, k);

	if (!no_prio && (j == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG12(k, j), 2, TAG11(k), TAG22(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG12(k, j), 1, TAG11(k));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_21(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned no_prio)
{
	int ret;
	struct starpu_task *task = create_task(TAG21(k, i));

	task->cl = &cl21;
	task->color = 0x8080c0;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, i);

	if (!no_prio && (i == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG21(k, i), 2, TAG11(k), TAG22(k-1, i, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG21(k, i), 1, TAG11(k));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_22(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned j, unsigned no_prio)
{
	int ret;

/*	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j)); */

	struct starpu_task *task = create_task(TAG22(k, i, j));

	task->cl = &cl22;
	task->color = 0x00ff00;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, i); /* produced by TAG21(k, i) */
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, j, k); /* produced by TAG12(k, j) */
	task->handles[2] = starpu_data_get_sub_data(dataA, 2, j, i); /* produced by TAG22(k-1, i, j) */

	if (!no_prio &&  (i == k + 1) && (j == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG12(k, j), TAG21(k, i));
	}
	else
	{
		starpu_tag_declare_deps(TAG22(k, i, j), 2, TAG12(k, j), TAG21(k, i));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

/*
 *	code to bootstrap the factorization
 */

static int dw_codelet_facto_v3(starpu_data_handle_t dataA, unsigned nblocks, unsigned no_prio)
{
	int ret;
	double start;
	double end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	if (bound)
		starpu_bound_start(bounddeps, boundprio);

	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);
		struct starpu_task *task = create_task_11(dataA, k, no_prio);

		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			ret = starpu_task_submit(task);
			if (ret == -ENODEV) return ret;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (i = k+1; i<nblocks; i++)
		{
			ret = create_task_12(dataA, k, i, no_prio);
			if (ret == -ENODEV) return ret;
			ret = create_task_21(dataA, k, i, no_prio);
			if (ret == -ENODEV) return ret;
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
			     ret = create_task_22(dataA, k, i, j, no_prio);
			     if (ret == -ENODEV) return ret;
			}
		}
		starpu_iteration_pop();
	}

	/* schedule the codelet */
	start = starpu_timing_now();
	ret = starpu_task_submit(entry_task);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* stall the application until the end of computations */
	starpu_tag_wait(TAG11(nblocks-1));

	end = starpu_timing_now();

	if (bound)
		starpu_bound_stop();

	double timing = end - start;
	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;

	PRINTF("# size\tms\tGFlops");
	if (bound)
		PRINTF("\tTms\tTGFlops");
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

	/* We already enforce deps by hand */
	starpu_data_set_sequential_consistency_flag(dataA, 0);

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

	int ret = dw_codelet_facto_v3(dataA, nblocks, no_prio);

	/* gather all the data */
	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
	starpu_data_unregister(dataA);

	return ret;
}
