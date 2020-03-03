/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* LU StarPU implementation using implicit task dependencies and partial
 * pivoting */

#include "xlu.h"
#include "xlu_kernels.h"

/*
 *	Construct the DAG
 */

static int create_task_pivot(starpu_data_handle_t *dataAp, unsigned nblocks,
			     struct piv_s *piv_description,
			     unsigned k, unsigned i,
			     starpu_data_handle_t (* get_block)(starpu_data_handle_t *, unsigned, unsigned, unsigned), unsigned no_prio)
{
	int ret;

	struct starpu_task *task = starpu_task_create();

	task->cl = &cl_pivot;
	task->color = 0xc0c000;

	/* which sub-data is manipulated ? */
	task->handles[0] = get_block(dataAp, nblocks, k, i);

	task->tag_id = PIVOT(k, i);

	task->cl_arg = &piv_description[k];

	/* this is an important task */
	if (!no_prio && (i == k+1))
		task->priority = STARPU_MAX_PRIO;

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_11_pivot(starpu_data_handle_t *dataAp, unsigned nblocks,
				unsigned k, struct piv_s *piv_description,
				starpu_data_handle_t (* get_block)(starpu_data_handle_t *, unsigned, unsigned, unsigned), unsigned no_prio)
{
	int ret;

	struct starpu_task *task = starpu_task_create();

	task->cl = &cl11_pivot;
	task->color = 0xffff00;

	task->cl_arg = &piv_description[k];

	/* which sub-data is manipulated ? */
	task->handles[0] = get_block(dataAp, nblocks, k, k);

	task->tag_id = TAG11(k);

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_12(starpu_data_handle_t *dataAp, unsigned nblocks, unsigned k, unsigned j,
			  starpu_data_handle_t (* get_block)(starpu_data_handle_t *, unsigned, unsigned, unsigned), unsigned no_prio)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl12;
	task->color = 0x8080ff;

	/* which sub-data is manipulated ? */
	task->handles[0] = get_block(dataAp, nblocks, k, k);
	task->handles[1] = get_block(dataAp, nblocks, j, k);

	task->tag_id = TAG12(k,j);

	if (!no_prio && (j == k+1))
		task->priority = STARPU_MAX_PRIO;

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_21(starpu_data_handle_t *dataAp, unsigned nblocks, unsigned k, unsigned i,
			  starpu_data_handle_t (* get_block)(starpu_data_handle_t *, unsigned, unsigned, unsigned), unsigned no_prio)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl21;
	task->color = 0x8080c0;

	/* which sub-data is manipulated ? */
	task->handles[0] = get_block(dataAp, nblocks, k, k);
	task->handles[1] = get_block(dataAp, nblocks, k, i);

	task->tag_id = TAG21(k,i);

	if (!no_prio && (i == k+1))
		task->priority = STARPU_MAX_PRIO;

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_22(starpu_data_handle_t *dataAp, unsigned nblocks, unsigned k, unsigned i, unsigned j,
			  starpu_data_handle_t (* get_block)(starpu_data_handle_t *, unsigned, unsigned, unsigned), unsigned no_prio)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl22;
	task->color = 0x00ff00;

	/* which sub-data is manipulated ? */
	task->handles[0] = get_block(dataAp, nblocks, k, i);
	task->handles[1] = get_block(dataAp, nblocks, j, k);
	task->handles[2] = get_block(dataAp, nblocks, j, i);

	task->tag_id = TAG22(k,i,j);

	if (!no_prio &&  (i == k + 1) && (j == k +1) )
		task->priority = STARPU_MAX_PRIO;

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

/*
 *	code to bootstrap the factorization
 */

static int dw_codelet_facto_pivot(starpu_data_handle_t *dataAp,
				  struct piv_s *piv_description,
				  unsigned nblocks,
				  starpu_data_handle_t (* get_block)(starpu_data_handle_t *, unsigned, unsigned, unsigned),
				  double *timing, unsigned no_prio)
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

		ret = create_task_11_pivot(dataAp, nblocks, k, piv_description, get_block, no_prio);
		if (ret == -ENODEV) return ret;

		for (i = 0; i < nblocks; i++)
		{
			if (i != k)
			{
				ret = create_task_pivot(dataAp, nblocks, piv_description, k, i, get_block, no_prio);
				if (ret == -ENODEV) return ret;
			}
		}

		for (i = k+1; i<nblocks; i++)
		{
			ret = create_task_12(dataAp, nblocks, k, i, get_block, no_prio);
			if (ret == -ENODEV) return ret;
			ret = create_task_21(dataAp, nblocks, k, i, get_block, no_prio);
			if (ret == -ENODEV) return ret;
		}
		starpu_data_wont_use(get_block(dataAp, nblocks, k, k));

		for (i = k+1; i<nblocks; i++)
		     for (j = k+1; j<nblocks; j++)
		     {
			     ret = create_task_22(dataAp, nblocks, k, i, j, get_block, no_prio);
			     if (ret == -ENODEV) return ret;
		     }
		for (i = k+1; i<nblocks; i++)
		{
		    starpu_data_wont_use(get_block(dataAp, nblocks, k, i));
		    starpu_data_wont_use(get_block(dataAp, nblocks, i, k));
		}
		starpu_iteration_pop();
	}

	/* stall the application until the end of computations */
	starpu_task_wait_for_all();

	end = starpu_timing_now();

	if (bound)
		starpu_bound_stop();

	*timing = end - start;
	return 0;
}

starpu_data_handle_t get_block_with_striding(starpu_data_handle_t *dataAp, unsigned nblocks, unsigned j, unsigned i)
{
	/* we use filters */
	(void)nblocks;
	return starpu_data_get_sub_data(*dataAp, 2, j, i);
}


int STARPU_LU(lu_decomposition_pivot)(TYPE *matA, unsigned *ipiv, unsigned size, unsigned ld, unsigned nblocks, unsigned no_prio)
{
	if (starpu_mic_worker_get_count() || starpu_mpi_ms_worker_get_count())
		/* These won't work with pivoting: we pass a pointer in cl_args */
		return -ENODEV;

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

	unsigned i;
	for (i = 0; i < size; i++)
		ipiv[i] = i;

	struct piv_s *piv_description = malloc(nblocks*sizeof(struct piv_s));
	unsigned block;
	for (block = 0; block < nblocks; block++)
	{
		piv_description[block].piv = ipiv;
		piv_description[block].first = block * (size / nblocks);
		piv_description[block].last = (block + 1) * (size / nblocks);
	}

	double timing;
	int ret = dw_codelet_facto_pivot(&dataA, piv_description, nblocks, get_block_with_striding, &timing, no_prio);
	if (ret)
		return ret;

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


	/* gather all the data */
	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
	starpu_data_unregister(dataA);

	free(piv_description);
	return 0;
}


starpu_data_handle_t get_block_with_no_striding(starpu_data_handle_t *dataAp, unsigned nblocks, unsigned j, unsigned i)
{
	/* dataAp is an array of data handle */
	return dataAp[i+j*nblocks];
}

int STARPU_LU(lu_decomposition_pivot_no_stride)(TYPE **matA, unsigned *ipiv, unsigned size, unsigned ld, unsigned nblocks, unsigned no_prio)
{
	(void)ld;
	starpu_data_handle_t *dataAp = malloc(nblocks*nblocks*sizeof(starpu_data_handle_t));

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	unsigned bi, bj;
	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		starpu_matrix_data_register(&dataAp[bi+nblocks*bj], STARPU_MAIN_RAM,
			(uintptr_t)matA[bi+nblocks*bj], size/nblocks,
			size/nblocks, size/nblocks, sizeof(TYPE));
	}

	unsigned i;
	for (i = 0; i < size; i++)
		ipiv[i] = i;

	struct piv_s *piv_description = malloc(nblocks*sizeof(struct piv_s));
	unsigned block;
	for (block = 0; block < nblocks; block++)
	{
		piv_description[block].piv = ipiv;
		piv_description[block].first = block * (size / nblocks);
		piv_description[block].last = (block + 1) * (size / nblocks);
	}

	double timing;
	int ret = dw_codelet_facto_pivot(dataAp, piv_description, nblocks, get_block_with_no_striding, &timing, no_prio);
	if (ret)
		return ret;

	unsigned n = starpu_matrix_get_nx(dataAp[0])*nblocks;
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


	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		starpu_data_unregister(dataAp[bi+nblocks*bj]);
	}
	free(dataAp);
	free(piv_description);
	return ret;
}
