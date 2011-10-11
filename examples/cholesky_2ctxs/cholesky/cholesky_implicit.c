/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  INRIA
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

/*
 *	Create the codelets
 */

static starpu_codelet cl11 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.type = STARPU_SEQ,
	.cpu_func = chol_cpu_codelet_update_u11,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u11,
#endif
	.nbuffers = 1,
	.model = &chol_model_11
};

static starpu_codelet cl21 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.type = STARPU_SEQ,
	.cpu_func = chol_cpu_codelet_update_u21,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u21,
#endif
	.nbuffers = 2,
	.model = &chol_model_21
};

static starpu_codelet cl22 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.type = STARPU_SEQ,
	.max_parallelism = INT_MAX,
	.cpu_func = chol_cpu_codelet_update_u22,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u22,
#endif
	.nbuffers = 3,
	.model = &chol_model_22
};

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static void callback_turn_spmd_on(void *arg __attribute__ ((unused)))
{
	cl22.type = STARPU_SPMD;
}

static double _cholesky(starpu_data_handle dataA, unsigned nblocks, double *timing)
{
	struct timeval start;
	struct timeval end;

	unsigned i,j,k;

	int prio_level = noprio?STARPU_DEFAULT_PRIO:STARPU_MAX_PRIO;

	gettimeofday(&start, NULL);

	/* create all the DAG nodes */
	for (k = 0; k < nblocks; k++)
	{
                starpu_data_handle sdatakk = starpu_data_get_sub_data(dataA, 2, k, k);

		starpu_insert_task(&cl11,
				   STARPU_PRIORITY, prio_level,
				   STARPU_RW, sdatakk,
				   STARPU_CALLBACK, (k == 3*nblocks/4)?callback_turn_spmd_on:NULL,
				   0);

		for (j = k+1; j<nblocks; j++)
		{
                        starpu_data_handle sdatakj = starpu_data_get_sub_data(dataA, 2, k, j);
			starpu_insert_task(&cl21,
					   STARPU_PRIORITY, (j == k+1)?prio_level:STARPU_DEFAULT_PRIO,
					   STARPU_R, sdatakk,
					   STARPU_RW, sdatakj,
					   0);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
                                {
					starpu_data_handle sdataki = starpu_data_get_sub_data(dataA, 2, k, i);
					starpu_data_handle sdataij = starpu_data_get_sub_data(dataA, 2, i, j);
					
					starpu_insert_task(&cl22,
							   STARPU_PRIORITY, ((i == k+1) && (j == k+1))?prio_level:STARPU_DEFAULT_PRIO,
							   STARPU_R, sdataki,
							   STARPU_R, sdatakj,
							   STARPU_RW, sdataij,
							   0);
                                }
			}
		}
	}

	starpu_task_wait_for_all();
		
	starpu_data_unpartition(dataA, 0);

	gettimeofday(&end, NULL);

	(*timing) = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	unsigned long n = starpu_matrix_get_nx(dataA);

	double flop = (1.0f*n*n*n)/3.0f;

	double gflops = (flop/(*timing)/1000.0f);
	(*timing) /= 1000000.0f; //sec
	//	(*timing) /= 60.0f; //min
	return gflops;
}

static double cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks, double *timing)
{
	starpu_data_handle dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, 0, (uintptr_t)matA, ld, size, size, sizeof(float));

	struct starpu_data_filter f = {
		.filter_func = starpu_vertical_block_filter_func,
		.nchildren = nblocks
	};

	struct starpu_data_filter f2 = {
		.filter_func = starpu_block_filter_func,
		.nchildren = nblocks
	};

	starpu_data_map_filters(dataA, 2, &f, &f2);
	double gflops = _cholesky(dataA, nblocks, timing);
	starpu_data_unregister(dataA);
	return gflops;
}

double run_cholesky_implicit(int start, int argc, char **argv, double *timing, pthread_barrier_t *barrier)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	unsigned size = 4 * 1024;
	unsigned nblocks = 16;
	parse_args_ctx(start, argc, argv, &size, &nblocks);

	//	starpu_init(NULL);

	//	starpu_helper_cublas_init();

	float *mat;

	starpu_malloc((void **)&mat, (size_t)size*size*sizeof(float));

	unsigned i,j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			//mat[j +i*size] = ((i == j)?1.0f*size:0.0f);
		}
	}

//#define PRINT_OUTPUT
#ifdef PRINT_OUTPUT
	printf("Input :\n");

	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j) {
				printf("%2.2f\t", mat[j +i*size]);
			}
			else {
				printf(".\t");
			}
		}
		printf("\n");
	}
#endif
	double gflops = cholesky(mat, size, size, nblocks, timing);

#ifdef PRINT_OUTPUT
	printf("Results :\n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j) {
				printf("%2.2f\t", mat[j +i*size]);
			}
			else {
				printf(".\t");
				mat[j+i*size] = 0.0f; // debug
			}
		}
		printf("\n");
	}
#endif

	if (check)
	{
		fprintf(stderr, "compute explicit LLt ...\n");
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i > j) {
					mat[j+i*size] = 0.0f; // debug
				}
			}
		}
		float *test_mat = malloc(size*size*sizeof(float));
		STARPU_ASSERT(test_mat);
	
		SSYRK("L", "N", size, size, 1.0f,
					mat, size, 0.0f, test_mat, size);
	
		fprintf(stderr, "comparing results ...\n");
#ifdef PRINT_OUTPUT
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i <= j) {
					printf("%2.2f\t", test_mat[j +i*size]);
				}
				else {
					printf(".\t");
				}
			}
			printf("\n");
		}
#endif
	
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i <= j) {
	                                float orig = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
	                                float err = abs(test_mat[j +i*size] - orig);
	                                if (err > 0.00001) {
	                                        fprintf(stderr, "Error[%d, %d] --> %2.2f != %2.2f (err %2.2f)\n", i, j, test_mat[j +i*size], orig, err);
	                                        assert(0);
	                                }
	                        }
			}
	        }
	}
	starpu_free((void *)mat);
	//	starpu_helper_cublas_shutdown();
	//	starpu_shutdown();

	return gflops;
}
