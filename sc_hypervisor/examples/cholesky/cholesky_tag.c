/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "cholesky.h"

struct starpu_perfmodel chol_model_potrf;
struct starpu_perfmodel chol_model_trsm;
struct starpu_perfmodel chol_model_syrk;
struct starpu_perfmodel chol_model_gemm;

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

static struct starpu_codelet cl_potrf =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_potrf},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_potrf},
#endif
	.nbuffers = 1,
	.model = &chol_model_potrf
};

static struct starpu_task * create_task_potrf(starpu_data_handle_t dataA, unsigned k)
{
/*	FPRINTF(stdout, "task potrf k = %d TAG = %llx\n", k, (TAG_POTRF(k))); */

	struct starpu_task *task = create_task(TAG_POTRF(k));

	task->cl = &cl_potrf;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

	/* this is an important task */
	if (!noprio)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_POTRF(k), 1, TAG_GEMM(k-1, k, k));
	}

	return task;
}

static struct starpu_codelet cl_trsm =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_trsm},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_trsm},
#endif
	.nbuffers = 2,
	.model = &chol_model_trsm
};

static void create_task_trsm(starpu_data_handle_t dataA, unsigned k, unsigned j)
{
	struct starpu_task *task = create_task(TAG_TRSM(k, j));

	task->cl = &cl_trsm;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);

	if (!noprio && (j == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM(k, j), 2, TAG_POTRF(k), TAG_GEMM(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM(k, j), 1, TAG_POTRF(k));
	}

	int ret = starpu_task_submit(task);
        if (STARPU_UNLIKELY(ret == -ENODEV))
	{
                FPRINTF(stderr, "No worker may execute this task\n");
                exit(0);
        }

}

static struct starpu_codelet cl_syrk =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_syrk},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_syrk},
#endif
	.nbuffers = 2,
	.model = &chol_model_syrk
};

static struct starpu_codelet cl_gemm =
{
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_gemm},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_gemm},
#endif
	.nbuffers = 3,
	.model = &chol_model_gemm
};

static void create_task_gemm(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned j)
{
/*	FPRINTF(stdout, "task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG_GEMM(k,i,j)); */

	struct starpu_task *task = create_task(TAG_GEMM(k, i, j));

	if (m ==n)
	{
		task->cl = &cl_syrk;

		/* which sub-data is manipulated ? */
		task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, i);
		task->handles[1] = starpu_data_get_sub_data(dataA, 2, i, j);
	}
	else
	{
		task->cl = &cl_gemm;

		/* which sub-data is manipulated ? */
		task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, i);
		task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);
		task->handles[2] = starpu_data_get_sub_data(dataA, 2, i, j);
	}

	if (!noprio && (i == k + 1) && (j == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 3, TAG_GEMM(k-1, i, j), TAG_TRSM(k, i), TAG_TRSM(k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 2, TAG_TRSM(k, i), TAG_TRSM(k, j));
	}

	int ret = starpu_task_submit(task);
        if (STARPU_UNLIKELY(ret == -ENODEV))
	{
                FPRINTF(stderr, "No worker may execute this task\n");
                exit(0);
        }
}



/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */

static void _cholesky(starpu_data_handle_t dataA, unsigned nblocks)
{
	struct timeval start;
	struct timeval end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	gettimeofday(&start, NULL);

	for (k = 0; k < nblocks; k++)
	{
		struct starpu_task *task = create_task_potrf(dataA, k);
		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			int ret = starpu_task_submit(task);
                        if (STARPU_UNLIKELY(ret == -ENODEV))
			{
                                FPRINTF(stderr, "No worker may execute this task\n");
                                exit(0);
                        }

		}

		for (j = k+1; j<nblocks; j++)
		{
			create_task_trsm(dataA, k, j);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
					create_task_gemm(dataA, k, i, j);
			}
		}
	}

	/* schedule the codelet */
	int ret = starpu_task_submit(entry_task);
        if (STARPU_UNLIKELY(ret == -ENODEV))
	{
                FPRINTF(stderr, "No worker may execute this task\n");
                exit(0);
        }


	/* stall the application until the end of computations */
	starpu_tag_wait(TAG_POTRF(nblocks-1));

	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);

	gettimeofday(&end, NULL);


	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	FPRINTF(stderr, "Computation took (in ms)\n");
	FPRINTF(stdout, "%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataA);

	double flop = (1.0f*n*n*n)/3.0f;
	FPRINTF(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

static int initialize_system(float **A, unsigned dim, unsigned pinned)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

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

	starpu_cublas_init();

	if (pinned)
	{
		starpu_malloc((void **)A, (size_t)dim*dim*sizeof(float));
	}
	else
	{
		*A = malloc(dim*dim*sizeof(float));
	}
	return 0;
}

static void cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks)
{
	starpu_data_handle_t dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld, size, size, sizeof(float));

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

	_cholesky(dataA, nblocks);

	starpu_data_unregister(dataA);
}

static void shutdown_system(float **matA, unsigned dim, unsigned pinned)
{
	if (pinned)
	{
		starpu_free_noflag(*matA, (size_t)dim*dim*sizeof(float));
	}
	else
	{
		free(*matA);
	}

	starpu_cublas_shutdown();
	starpu_shutdown();
}

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	parse_args(argc, argv);

	float *mat;
	int ret = initialize_system(&mat, size, pinned);
	if (ret) return ret;

	unsigned i,j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			/* mat[j +i*size] = ((i == j)?1.0f*size:0.0f); */
		}
	}


#ifdef CHECK_OUTPUT
	FPRINTF(stdout, "Input :\n");

	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j)
			{
				FPRINTF(stdout, "%2.2f\t", mat[j +i*size]);
			}
			else
			{
				FPRINTF(stdout, ".\t");
			}
		}
		FPRINTF(stdout, "\n");
	}
#endif


	cholesky(mat, size, size, nblocks);

#ifdef CHECK_OUTPUT
	FPRINTF(stdout, "Results :\n");

	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j)
			{
				FPRINTF(stdout, "%2.2f\t", mat[j +i*size]);
			}
			else
			{
				FPRINTF(stdout, ".\t");
				mat[j+i*size] = 0.0f; /* debug */
			}
		}
		FPRINTF(stdout, "\n");
	}

	FPRINTF(stderr, "compute explicit LLt ...\n");
	float *test_mat = malloc(size*size*sizeof(float));
	STARPU_ASSERT(test_mat);

	STARPU_SSYRK("L", "N", size, size, 1.0f,
				mat, size, 0.0f, test_mat, size);

	FPRINTF(stderr, "comparing results ...\n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i <= j)
			{
				FPRINTF(stdout, "%2.2f\t", test_mat[j +i*size]);
			}
			else
			{
				FPRINTF(stdout, ".\t");
			}
		}
		FPRINTF(stdout, "\n");
	}
	free(test_mat);
#endif

	shutdown_system(&mat, size, pinned);
	return 0;
}
