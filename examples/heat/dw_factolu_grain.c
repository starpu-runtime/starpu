/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * It also changes the partitioning during execution: when called first,
 * dw_factoLU_grain_inner splits the matrix with a big granularity (nblocks)
 * and processes nbigblocks blocks, before calling itself again, to process the
 * remainder of the matrix with a smaller granularity.
 */

#include "dw_factolu.h"

#define TAG11(k, prefix)	((starpu_tag_t)( (((unsigned long long)(prefix))<<60)  |  (1ULL<<56) | (unsigned long long)(k)))
#define TAG12(k,i, prefix)	((starpu_tag_t)((((unsigned long long)(prefix))<<60)  | ((2ULL<<56) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21(k,j, prefix)	((starpu_tag_t)( (((unsigned long long)(prefix))<<60)  |  ((3ULL<<56) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j, prefix)	((starpu_tag_t)(  (((unsigned long long)(prefix))<<60)  |  ((4ULL<<56) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

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

static struct starpu_codelet cl11 =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_u11},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u11"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u11},
#endif
	.nbuffers = 1,
	.model = &model_11
};

static struct starpu_task *create_task_11(starpu_data_handle_t dataA, unsigned k, unsigned tag_prefix)
{
/*	FPRINTF(stdout, "task 11 k = %d TAG = %llx\n", k, (TAG11(k))); */

	struct starpu_task *task = create_task(TAG11(k, tag_prefix));

	task->cl = &cl11;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

	/* this is an important task */
	task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG11(k, tag_prefix), 1, TAG22(k-1, k, k, tag_prefix));
	}

	return task;
}

static struct starpu_codelet cl12 =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_u12},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u12"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u12},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.model = &model_12
};

static void create_task_12(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned tag_prefix)
{
	int ret;

/*	FPRINTF(stdout, "task 12 k,i = %d,%d TAG = %llx\n", k,i, TAG12(k,i)); */

	struct starpu_task *task = create_task(TAG12(k, i, tag_prefix));
	
	task->cl = &cl12;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, i, k);

	if (i == k+1)
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG12(k, i, tag_prefix), 2, TAG11(k, tag_prefix), TAG22(k-1, i, k, tag_prefix));
	}
	else
	{
		starpu_tag_declare_deps(TAG12(k, i, tag_prefix), 1, TAG11(k, tag_prefix));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static struct starpu_codelet cl21 =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_u21},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u21"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u21},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.model = &model_21
};

static void create_task_21(starpu_data_handle_t dataA, unsigned k, unsigned j, unsigned tag_prefix)
{
	int ret;
	struct starpu_task *task = create_task(TAG21(k, j, tag_prefix));

	task->cl = &cl21;
	
	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);

	if (j == k+1)
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG21(k, j, tag_prefix), 2, TAG11(k, tag_prefix), TAG22(k-1, k, j, tag_prefix));
	}
	else
	{
		starpu_tag_declare_deps(TAG21(k, j, tag_prefix), 1, TAG11(k, tag_prefix));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static struct starpu_codelet cl22 =
{
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.cpu_funcs = {dw_cpu_codelet_update_u22},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u22"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u22},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.model = &model_22
};

static void create_task_22(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned j, unsigned tag_prefix)
{
	int ret;
/*	FPRINTF(stdout, "task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j)); */

	struct starpu_task *task = create_task(TAG22(k, i, j, tag_prefix));

	task->cl = &cl22;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, i, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);
	task->handles[2] = starpu_data_get_sub_data(dataA, 2, i, j);

	if ( (i == k + 1) && (j == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG22(k, i, j, tag_prefix), 3, TAG22(k-1, i, j, tag_prefix), TAG12(k, i, tag_prefix), TAG21(k, j, tag_prefix));
	}
	else
	{
		starpu_tag_declare_deps(TAG22(k, i, j, tag_prefix), 2, TAG12(k, i, tag_prefix), TAG21(k, j, tag_prefix));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static void dw_factoLU_grain_inner(float *matA, unsigned size, unsigned inner_size,
				unsigned ld, unsigned blocksize, unsigned tag_prefix)
{
	int ret;
	/*
	 * (re)partition data
	 */
	starpu_data_handle_t dataA;
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld, size, size, sizeof(float));

	STARPU_ASSERT((size % blocksize) == 0);
	STARPU_ASSERT((inner_size % blocksize) == 0);

	unsigned nblocks = size / blocksize;
	unsigned maxk = inner_size / blocksize;

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


	/*
	 * submit tasks
	 */

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	/* if maxk < nblocks we'll stop before the LU decomposition is totally done */
	for (k = 0; k < maxk; k++)
	{
		struct starpu_task *task = create_task_11(dataA, k, tag_prefix);

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
			create_task_12(dataA, k, i, tag_prefix);
			create_task_21(dataA, k, i, tag_prefix);
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
				create_task_22(dataA, k, i, j, tag_prefix);
			}
		}
	}

	ret = starpu_task_submit(entry_task);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		FPRINTF(stderr, "No worker may execute this task\n");
		exit(-1);
	}

	/* is this the last call to dw_factoLU_grain_inner ? */
	if (inner_size == size)
	{
		/* we wait for the last task and we are done */
		starpu_tag_wait(TAG11(nblocks-1, tag_prefix));
		starpu_data_unpartition(dataA, STARPU_MAIN_RAM);		
		return;
	}
	else
	{
		/*
		 * call dw_factoLU_grain_inner recursively in the remaining blocks
		 */

		unsigned ndeps_tags = (nblocks - maxk)*(nblocks - maxk);
		starpu_tag_t *tag_array = calloc(ndeps_tags, sizeof(starpu_tag_t));
		STARPU_ASSERT(tag_array);

		unsigned ind = 0;
		for (i = maxk; i < nblocks; i++)
		for (j = maxk; j < nblocks; j++)
		{
			tag_array[ind++] = TAG22(maxk-1, i, j, tag_prefix);
		}

		starpu_tag_wait_array(ind, tag_array);

		free(tag_array);

		starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
		starpu_data_unregister(dataA);

		float *newmatA = &matA[inner_size*(ld+1)];

/*		if (tag_prefix < 2)
		{
			dw_factoLU_grain_inner(newmatA, size-inner_size, (size-inner_size)/2, ld, blocksize/2, tag_prefix+1);
		}
		else
		{ */
			dw_factoLU_grain_inner(newmatA, size-inner_size, size-inner_size, ld, blocksize/2, tag_prefix+1);
/*		} */
	}

}

void dw_factoLU_grain(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned nbigblocks)
{

#ifdef CHECK_RESULTS
	FPRINTF(stderr, "Checking results ...\n");
	float *Asaved;
	Asaved = malloc(ld*ld*sizeof(float));

	memcpy(Asaved, matA, ld*ld*sizeof(float));
#endif

	double start;
	double end;

	/* schedule the codelet */
	start = starpu_timing_now();

	/* that's only ok for powers of 2 yet ! */
	dw_factoLU_grain_inner(matA, size, (size/nblocks) * nbigblocks, ld, size/nblocks, 0);

	end = starpu_timing_now();

	double timing = end - start;

	unsigned n = size;
	double flop = (2.0f*n*n*n)/3.0f;

	PRINTF("# size\tms\tGFlops\n");
	PRINTF("%u\t%.0f\t%.1f\n", n, timing/1000, flop/timing/1000.0f);

#ifdef CHECK_RESULTS
	compare_A_LU(Asaved, matA, size, ld);
	free(Asaved);
#endif
}
