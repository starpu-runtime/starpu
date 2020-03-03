/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * It also uses data partitioning to split the matrix into submatrices.
 * It also changes the partitioning during execution: when called first,
 * cholesky_grain_rec splits the matrix with a big granularity (nblocks) and
 * processes nbigblocks blocks, before calling itself again, to process the
 * remainder of the matrix with a smaller granularity.
 */

/* Note: this is using fortran ordering, i.e. column-major ordering, i.e.
 * elements with consecutive row number are consecutive in memory */

#include "cholesky.h"

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#endif

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

static struct starpu_task * create_task_11(starpu_data_handle_t dataA, unsigned k, unsigned reclevel)
{
/*	FPRINTF(stdout, "task 11 k = %d TAG = %llx\n", k, (TAG11(k))); */

	struct starpu_task *task = create_task(TAG11_AUX(k, reclevel));

	task->cl = &cl11;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

	/* this is an important task */
	if (!noprio_p)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG11_AUX(k, reclevel), 1, TAG22_AUX(k-1, k, k, reclevel));
	}

	int n = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_SPOTRF(n);

	return task;
}

static int create_task_21(starpu_data_handle_t dataA, unsigned k, unsigned m, unsigned reclevel)
{
	int ret;

	struct starpu_task *task = create_task(TAG21_AUX(k, m, reclevel));

	task->cl = &cl21;

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
		starpu_tag_declare_deps(TAG21_AUX(k, m, reclevel), 2, TAG11_AUX(k, reclevel), TAG22_AUX(k-1, m, k, reclevel));
	}
	else
	{
		starpu_tag_declare_deps(TAG21_AUX(k, m, reclevel), 1, TAG11_AUX(k, reclevel));
	}

	int nx = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_STRSM(nx, nx);

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_22(starpu_data_handle_t dataA, unsigned k, unsigned m, unsigned n, unsigned reclevel)
{
	int ret;

/*	FPRINTF(stdout, "task 22 k,n,m = %d,%d,%d TAG = %llx\nx", k,m,n, TAG22_AUX(k,m,n)); */

	struct starpu_task *task = create_task(TAG22_AUX(k, m, n, reclevel));

	task->cl = &cl22;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, n, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, m, k);
	task->handles[2] = starpu_data_get_sub_data(dataA, 2, m, n);

	if ( (n == k + 1) && (m == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG22_AUX(k, m, n, reclevel), 3, TAG22_AUX(k-1, m, n, reclevel), TAG21_AUX(k, n, reclevel), TAG21_AUX(k, m, reclevel));
	}
	else
	{
		starpu_tag_declare_deps(TAG22_AUX(k, m, n, reclevel), 2, TAG21_AUX(k, n, reclevel), TAG21_AUX(k, m, reclevel));
	}

	int nx = starpu_matrix_get_nx(task->handles[0]);
	task->flops = FLOPS_SGEMM(nx, nx, nx);

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}



/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static int cholesky_grain_rec(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned nbigblocks, unsigned reclevel)
{
	int ret;

	/* create a new codelet */
	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned k, m, n;

	starpu_data_handle_t dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
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

	for (k = 0; k < nbigblocks; k++)
	{
		starpu_iteration_push(k);
		struct starpu_task *task = create_task_11(dataA, k, reclevel);
		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			ret = starpu_task_submit(task);
			if (ret == -ENODEV) return 77;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (m = k+1; m<nblocks; m++)
		{
		     	ret = create_task_21(dataA, k, m, reclevel);
			if (ret == -ENODEV) return 77;

			for (n = k+1; n<nblocks; n++)
			{
				if (n <= m)
				{
				     ret = create_task_22(dataA, k, m, n, reclevel);
				     if (ret == -ENODEV) return 77;
				}
			}
		}
		starpu_iteration_pop();
	}

	/* schedule the codelet */
	ret = starpu_task_submit(entry_task);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	if (nblocks == nbigblocks)
	{
		/* stall the application until the end of computations */
		starpu_tag_wait(TAG11_AUX(nblocks-1, reclevel));
		starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
		starpu_data_unregister(dataA);
		return 0;
	}
	else
	{
		STARPU_ASSERT(reclevel == 0);
		unsigned ndeps_tags = (nblocks - nbigblocks)*(nblocks - nbigblocks);

		starpu_tag_t *tag_array = malloc(ndeps_tags*sizeof(starpu_tag_t));
		STARPU_ASSERT(tag_array);

		unsigned ind = 0;
		for (n = nbigblocks; n < nblocks; n++)
		for (m = nbigblocks; m < nblocks; m++)
		{
			if (n <= m)
				tag_array[ind++] = TAG22_AUX(nbigblocks - 1, m, n, reclevel);
		}

		starpu_tag_wait_array(ind, tag_array);

		free(tag_array);

		starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
		starpu_data_unregister(dataA);

		float *newmatA = &matA[nbigblocks*(size/nblocks)*(ld+1)];

		return cholesky_grain_rec(newmatA, size/nblocks*(nblocks - nbigblocks), ld, (nblocks - nbigblocks)*2, (nblocks - nbigblocks)*2, reclevel+1);
	}
}

static int initialize_system(int argc, char **argv, float **A, unsigned pinned)
{
	int ret;
	int flags = STARPU_MALLOC_SIMULATION_FOLDED;

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
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,cuda_chol_task_11_cost);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,cuda_chol_task_21_cost);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,cuda_chol_task_22_cost);
#else
	initialize_chol_model(&chol_model_11,"chol_model_11",cpu_chol_task_11_cost,NULL);
	initialize_chol_model(&chol_model_21,"chol_model_21",cpu_chol_task_21_cost,NULL);
	initialize_chol_model(&chol_model_22,"chol_model_22",cpu_chol_task_22_cost,NULL);
#endif

	starpu_cublas_init();

	if (pinned)
		flags |= STARPU_MALLOC_PINNED;
	starpu_malloc_flags((void **)A, size_p*size_p*sizeof(float), flags);

	return 0;
}

int cholesky_grain(float *matA, unsigned size, unsigned ld, unsigned nblocks, unsigned nbigblocks)
{
	double start;
	double end;
	int ret;

	start = starpu_timing_now();

	ret = cholesky_grain_rec(matA, size, ld, nblocks, nbigblocks, 0);

	end = starpu_timing_now();

	double timing = end - start;

	double flop = (1.0f*size*size*size)/3.0f;

	PRINTF("# size\tms\tGFlops\n");
	PRINTF("%u\t%.0f\t%.1f\n", size, timing/1000, (flop/timing/1000.0f));

	return ret;
}

static void shutdown_system(float **matA, unsigned dim, unsigned pinned)
{
	int flags = STARPU_MALLOC_SIMULATION_FOLDED;
	if (pinned)
		flags |= STARPU_MALLOC_PINNED;

	starpu_free_flags(*matA, dim*dim*sizeof(float), flags);

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
	unsigned m,n;

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

	ret = cholesky_grain(mat, size_p, size_p, nblocks_p, nbigblocks_p);

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
	                                        FPRINTF(stderr, "Error[%u, %u] --> %2.6f != %2.6f (err %2.6f)\n", m, n, test_mat[m +n*size_p], orig, err);
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
