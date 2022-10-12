/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This implements an LU factorization.
 * The task graph is submitted through dependency tags.
 */

#include "dw_factolu.h"

#define TAG_GETRF(k)	((starpu_tag_t)((1ULL<<60) | (unsigned long long)(k)))
#define TAG_TRSM_LL(k,i)	((starpu_tag_t)(((2ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG_TRSM_RU(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

static unsigned no_prio = 0;

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

static struct starpu_codelet cl_getrf =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_getrf},
	.cpu_funcs_name = {"dw_cpu_codelet_update_getrf"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_getrf},
#endif
	.nbuffers = 1,
	.model = &model_getrf
};

static struct starpu_task *create_task_getrf(starpu_data_handle_t dataA, unsigned k)
{
/*	printf("task 11 k = %d TAG = %llx\n", k, (TAG_GETRF(k))); */

	struct starpu_task *task = create_task(TAG_GETRF(k));

	task->cl = &cl_getrf;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GETRF(k), 1, TAG_GEMM(k-1, k, k));
	}

	return task;
}

static struct starpu_codelet cl_trsm_ll =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_trsm_ll},
	.cpu_funcs_name = {"dw_cpu_codelet_update_trsm_ll"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_trsm_ll},
#endif
	.nbuffers = 2,
	.model = &model_trsm_ll
};

static void create_task_trsm_ll(starpu_data_handle_t dataA, unsigned k, unsigned i)
{
	int ret;

/*	printf("task 12 k,i = %d,%d TAG = %llx\n", k,i, TAG_TRSM_LL(k,i)); */

	struct starpu_task *task = create_task(TAG_TRSM_LL(k, i));

	task->cl = &cl_trsm_ll;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, i, k);

	if (!no_prio && (i == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM_LL(k, i), 2, TAG_GETRF(k), TAG_GEMM(k-1, i, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM_LL(k, i), 1, TAG_GETRF(k));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static struct starpu_codelet cl_trsm_ru =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_trsm_ru},
	.cpu_funcs_name = {"dw_cpu_codelet_update_trsm_ru"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_trsm_ru},
#endif
	.nbuffers = 2,
	.model = &model_trsm_ru
};

static void create_task_trsm_ru(starpu_data_handle_t dataA, unsigned k, unsigned j)
{
	int ret;
	struct starpu_task *task = create_task(TAG_TRSM_RU(k, j));

	task->cl = &cl_trsm_ru;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);

	if (!no_prio && (j == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM_RU(k, j), 2, TAG_GETRF(k), TAG_GEMM(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM_RU(k, j), 1, TAG_GETRF(k));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static struct starpu_codelet cl_gemm =
{
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_gemm},
	.cpu_funcs_name = {"dw_cpu_codelet_update_gemm"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_gemm},
#endif
	.nbuffers = 3,
	.model = &model_gemm
};

static void create_task_gemm(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned j)
{
	int ret;

/*	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG_GEMM(k,i,j)); */

	struct starpu_task *task = create_task(TAG_GEMM(k, i, j));

	task->cl = &cl_gemm;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, i, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);
	task->handles[2] = starpu_data_get_sub_data(dataA, 2, i, j);

	if (!no_prio &&  (i == k + 1) && (j == k +1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 3, TAG_GEMM(k-1, i, j), TAG_TRSM_LL(k, i), TAG_TRSM_RU(k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 2, TAG_TRSM_LL(k, i), TAG_TRSM_RU(k, j));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

/*
 *	code to bootstrap the factorization
 */

static void dw_codelet_facto_v3(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;

	double start;
	double end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		struct starpu_task *task = create_task_getrf(dataA, k);

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

		for (i = k+1; i<nblocks; i++)
		{
			create_task_trsm_ll(dataA, k, i);
			create_task_trsm_ru(dataA, k, i);
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
				create_task_gemm(dataA, k, i, j);
			}
		}
	}

	/* schedule the codelet */
	start = starpu_timing_now();
	ret = starpu_task_submit(entry_task);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		FPRINTF(stderr, "No worker may execute this task\n");
		exit(-1);
	}



	/* stall the application until the end of computations */
	starpu_tag_wait(TAG_GETRF(nblocks-1));

	end = starpu_timing_now();

	double timing = end - start;

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;

	PRINTF("# size\tms\tGFlop/s\n");
	PRINTF("%u\t%.0f\t%.1f\n", n, timing/1000, flop/timing/1000.0f);
}

void dw_factoLU_tag(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned _no_prio)
{

#ifdef CHECK_RESULTS
	FPRINTF(stderr, "Checking results ...\n");
	float *Asaved;
	Asaved = malloc((size_t)ld*ld*sizeof(float));

	memcpy(Asaved, matA, (size_t)ld*ld*sizeof(float));
#endif

	no_prio = _no_prio;

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

	dw_codelet_facto_v3(dataA, nblocks);

	/* gather all the data */
	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);

	starpu_data_unregister(dataA);

#ifdef CHECK_RESULTS
	compare_A_LU(Asaved, matA, size, ld);
#endif
}
