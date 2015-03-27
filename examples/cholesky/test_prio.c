/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2015  Université de Bordeaux
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
/* to test with dmda* schedulers */
static int _test_prio(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;
	double start;
	double end;

	unsigned i,j,k;
	unsigned long n = starpu_matrix_get_nx(dataA);
	unsigned long nn = n/nblocks;

	int prio_level = noprio?STARPU_DEFAULT_PRIO:STARPU_MAX_PRIO;

	starpu_fxt_start_profiling();

	start = starpu_timing_now();

	/* create all the DAG nodes */
	for (k = 0; k < nblocks; k++)
	{
                starpu_data_handle_t sdatakk = starpu_data_get_sub_data(dataA, 2, k, k);

                ret = starpu_task_insert(&cl11,
					 STARPU_PRIORITY, k,
					 STARPU_RW, sdatakk,
					 STARPU_FLOPS, (double) FLOPS_SPOTRF(nn),
					 STARPU_TAG_ONLY, TAG11(k),
					 0);
		if (ret == -ENODEV) return 77;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	
		for (j = 0; j<nblocks; j++)
		{
			starpu_data_handle_t sdatakj = starpu_data_get_sub_data(dataA, 2, k, j);
			
			ret = starpu_task_insert(&cl21,
						 STARPU_PRIORITY, k+j,
						 STARPU_R, sdatakj,
						 STARPU_RW, sdatakj,
						 STARPU_FLOPS, (double) FLOPS_STRSM(nn, nn),
						 STARPU_TAG_ONLY, TAG21(k,j),
						 0);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}

	}

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	starpu_fxt_stop_profiling();

	double timing = end - start;

	double flop = FLOPS_SPOTRF(n);

	PRINTF("# size\tms\tGFlops");

	PRINTF("\n");
	
	PRINTF("%lu\t%.0f\t%.1f", n, timing/1000, (flop/timing/1000.0f));

	PRINTF("\n");

	return 0;
}

static int test_prio(float *matA, unsigned size, unsigned ld, unsigned nblocks)
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

	int ret = _test_prio(dataA, nblocks);

	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
	starpu_data_unregister(dataA);

	return ret;
}

static void execute_test_prio(unsigned size, unsigned nblocks)
{
	int ret;
	float *mat = NULL;
	unsigned i,j;

#ifndef STARPU_SIMGRID
	starpu_malloc((void **)&mat, (size_t)size*size*sizeof(float));
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			/* mat[j +i*size] = ((i == j)?1.0f*size:0.0f); */
		}
	}
#endif
        ret = test_prio(mat, size, size, nblocks);

	starpu_free(mat);
}

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	parse_args(argc, argv);


	int ret;
	ret = starpu_init(NULL);
	starpu_fxt_stop_profiling();

	if (ret == -ENODEV)
                return 77;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_CUDA
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,cuda_chol_task_11_cost);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,cuda_chol_task_21_cost);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,cuda_chol_task_22_cost);
#else
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,NULL);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,NULL);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,NULL);
#endif

	starpu_cublas_init();

	execute_test_prio(size, nblocks);

	starpu_cublas_shutdown();
	starpu_shutdown();

	return ret;
}
