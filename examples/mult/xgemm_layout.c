/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2017       Erwan Leria
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
 * Simple parallel GEMM implementation: partition the output matrix in the two
 * dimensions, and the input matrices in the corresponding dimension, and
 * perform the output computations in parallel.
 */
#include "xgemm.h"

static unsigned invalidate_c_tile = 0;
static unsigned random_task_order = 0;
static unsigned recursive_matrix_layout = 0;
static unsigned random_data_access = 0;
static unsigned count_do_schedule = 1;
static unsigned sparse_matrix = 0;
/* % de chance qu'une tâche soit créé avec sparse matrix. */
static int chance_to_be_created = 100;
static TYPE **Cscratch;

static void init_problem_data(void)
{
#ifndef STARPU_SIMGRID
	unsigned i,j;
#endif

	starpu_malloc_flags((void **)&A, zdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&B, xdim*zdim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&C, xdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

#ifndef STARPU_SIMGRID
	/* fill the A and B matrices */
	for (j=0; j < ydim; j++)
	{
		for (i=0; i < zdim; i++)
		{
			A[j+i*ydim] = (TYPE)(starpu_drand48());
		}
	}

	for (j=0; j < zdim; j++)
	{
		for (i=0; i < xdim; i++)
		{
			B[j+i*zdim] = (TYPE)(starpu_drand48());
		}
	}

	for (j=0; j < ydim; j++)
	{
		for (i=0; i < xdim; i++)
		{
			C[j+i*ydim] = (TYPE)(0);
		}
	}
#endif

	if (!tiled)
	{
		unsigned x;
		unsigned ncuda = starpu_cuda_worker_get_count();
		Cscratch = malloc(sizeof(TYPE*) * ncuda);
		for(x = 0; x < ncuda; x++)
		{
			unsigned worker = starpu_worker_get_by_type(STARPU_CUDA_WORKER, x);
			unsigned node = starpu_worker_get_memory_node(worker);
			Cscratch[x] = (TYPE*) starpu_malloc_on_node(node, (xdim / nslicesx) * (ydim / nslicesy) * sizeof(TYPE));
		}
	}
}

void nop(void *descr[], void *arg)
{
	(void) descr;
	(void) arg;
}

static struct starpu_codelet redux_cl =
{
	.where = STARPU_NOWHERE,
	.cpu_funcs = {nop},
	.cpu_funcs_name = {"nop"},
	.cuda_funcs = {nop},
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_RW | STARPU_COMMUTE, STARPU_R},
	.model = &starpu_perfmodel_nop
};

static struct starpu_codelet init_cl =
{
	.where = STARPU_NOWHERE,
	.cpu_funcs = {nop},
	.cpu_funcs_name = {"nop"},
	.cuda_funcs = {nop},
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.model = &starpu_perfmodel_nop
};

static void partition_mult_data(void)
{
	unsigned x, y, z;

	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, ydim, ydim, zdim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B, zdim, zdim, xdim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C, ydim, ydim, xdim, sizeof(TYPE));
	starpu_data_set_reduction_methods(C_handle, &redux_cl, &init_cl);

	struct starpu_data_filter vert;
	memset(&vert, 0, sizeof(vert));
	vert.filter_func = starpu_matrix_filter_vertical_block;
	vert.nchildren = nslicesx;

	struct starpu_data_filter horiz;
	memset(&horiz, 0, sizeof(horiz));
	horiz.filter_func = starpu_matrix_filter_block;
	horiz.nchildren = nslicesy;

	if (tiled)
	{
		struct starpu_data_filter vertA;
		memset(&vertA, 0, sizeof(vertA));
		vertA.filter_func = starpu_matrix_filter_vertical_block;
		vertA.nchildren = nslicesz;

		struct starpu_data_filter horizB;
		memset(&horizB, 0, sizeof(horizB));
		horizB.filter_func = starpu_matrix_filter_block;
		horizB.nchildren = nslicesz;

		starpu_data_map_filters(A_handle, 2, &vertA, &horiz);
		starpu_data_map_filters(B_handle, 2, &vert, &horizB);
		starpu_data_map_filters(C_handle, 2, &vert, &horiz);

		for (y = 0; y < nslicesy; y++)
		for (z = 0; z < nslicesz; z++)
			starpu_data_set_coordinates(starpu_data_get_sub_data(A_handle, 2, z, y), 2, z, y);

		for (x = 0; x < nslicesx; x++)
		for (z = 0; z < nslicesz; z++)
			starpu_data_set_coordinates(starpu_data_get_sub_data(B_handle, 2, x, z), 2, x, z);
	}
	else
	{
		starpu_data_partition(B_handle, &vert);
		starpu_data_partition(A_handle, &horiz);

		starpu_data_map_filters(C_handle, 2, &vert, &horiz);

		for (y = 0; y < nslicesy; y++)
			starpu_data_set_coordinates(starpu_data_get_sub_data(A_handle, 1, y), 2, 0, y);

		for (x = 0; x < nslicesx; x++)
			starpu_data_set_coordinates(starpu_data_get_sub_data(B_handle, 1, x), 2, x, 0);
	}

	for (x = 0; x < nslicesx; x++)
	for (y = 0; y < nslicesy; y++)
		starpu_data_set_coordinates(starpu_data_get_sub_data(C_handle, 2, x, y), 2, x, y);
}

#ifdef STARPU_USE_CUDA
static void cublas_mult2d(void *descr[], void *arg, const TYPE *beta)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	unsigned worker = starpu_worker_get_id_check();
	unsigned devid = starpu_worker_get_devid(worker);
	TYPE *subC = Cscratch[devid];

	unsigned nxC = STARPU_MATRIX_GET_NY(descr[1]);
	unsigned nyC = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = nxC;

	cudaStream_t stream = starpu_cuda_get_local_stream();

	cublasStatus_t status = CUBLAS_GEMM(starpu_cublas_get_local_handle(),
					    CUBLAS_OP_N, CUBLAS_OP_N,
					    nxC, nyC, nyA,
					    &p1_cuda, subA, ldA, subB, ldB,
					    beta, subC, ldC);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

#ifdef STARPU_USE_CUDA
static void cublas_mult(void *descr[], void *arg, const TYPE *beta)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

	cudaStream_t stream = starpu_cuda_get_local_stream();

	if (nxC == ldC)
		cudaMemsetAsync(subC, 0, sizeof(*subC) * nxC * nyC, stream);
	else
	{
		unsigned i;
		for (i = 0; i < nyC; i++)
			cudaMemsetAsync(subC + i*ldC, 0, sizeof(*subC) * nxC, stream);
	}

	cublasStatus_t status = CUBLAS_GEMM(starpu_cublas_get_local_handle(),
					    CUBLAS_OP_N, CUBLAS_OP_N,
					    nxC, nyC, nyA,
					    &p1_cuda, subA, ldA, subB, ldB,
					    beta, subC, ldC);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

#ifdef STARPU_USE_CUDA
static void cublas_gemm2d(void *descr[], void *arg)
{
	cublas_mult2d(descr, arg, &v0_cuda);
}
#endif

#ifdef STARPU_HAVE_BLAS
void cpu_mult2d(void *descr[], void *arg, TYPE beta)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned nxC = STARPU_MATRIX_GET_NY(descr[1]);
	unsigned nyC = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned ldC = nxC;

	TYPE subC[nxC*nyC];

	int worker_size = starpu_combined_worker_get_size();

	if (worker_size == 1)
	{
		/* Sequential CPU task */
		CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB, beta, subC, ldC);
	}
	else
	{
		/* Parallel CPU task */
		unsigned rank = starpu_combined_worker_get_rank();

		unsigned block_size = (nyC + worker_size - 1)/worker_size;
		unsigned new_nyC = STARPU_MIN(nyC, block_size*(rank+1)) - block_size*rank;

		STARPU_ASSERT(nyC == STARPU_MATRIX_GET_NY(descr[1]));

		TYPE *new_subB = &subB[block_size*rank];
		TYPE *new_subC = &subC[block_size*rank];

		CPU_GEMM("N", "N", nxC, new_nyC, nyA, (TYPE)1.0, subA, ldA, new_subB, ldB, beta, new_subC, ldC);
	}
}
#endif

#ifdef STARPU_HAVE_BLAS
void cpu_mult(void *descr[], void *arg, TYPE beta)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

	int worker_size = starpu_combined_worker_get_size();

	if (nxC == ldC)
		memset(subC, 0, sizeof(*subC) * nxC * nyC);
	else
	{
		unsigned i;
		for (i = 0; i < nyC; i++)
			memset(subC + i*ldC, 0, sizeof(*subC) * nxC);
	}

	if (worker_size == 1)
	{
		/* Sequential CPU task */
		CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB, beta, subC, ldC);
	}
	else
	{
		/* Parallel CPU task */
		unsigned rank = starpu_combined_worker_get_rank();

		unsigned block_size = (nyC + worker_size - 1)/worker_size;
		unsigned new_nyC = STARPU_MIN(nyC, block_size*(rank+1)) - block_size*rank;

		STARPU_ASSERT(nyC == STARPU_MATRIX_GET_NY(descr[1]));

		TYPE *new_subB = &subB[block_size*rank];
		TYPE *new_subC = &subC[block_size*rank];

		CPU_GEMM("N", "N", nxC, new_nyC, nyA, (TYPE)1.0, subA, ldA, new_subB, ldB, beta, new_subC, ldC);
	}
}
#endif

#ifdef STARPU_HAVE_BLAS
void cpu_gemm2d(void *descr[], void *arg)
{
	cpu_mult2d(descr, arg, 0.);
}
#endif

/* Codelet for 2D matrix */
static struct starpu_codelet cl_gemm2d =
{
#ifdef STARPU_HAVE_BLAS
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_gemm2d},
	.cpu_funcs_name = {"cpu_gemm2d"},
#endif
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_gemm2d},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_R},
	.model = &starpu_gemm_model
};

/* Codelet for 3D matrix z = 0 */
static struct starpu_codelet cl_gemm0 =
{
#ifdef STARPU_HAVE_BLAS
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_gemm0},
	.cpu_funcs_name = {"cpu_gemm0"},
#endif
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_gemm0},
#elif defined(STARPU_USE_HIP)
	.hip_funcs = {hipblas_gemm0},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.hip_flags = {STARPU_HIP_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_R},
	.model = &starpu_gemm_model
};

/* Codelet for 3D matrix z = 1, 2, 3 */
static struct starpu_codelet cl_gemm =
{
#ifdef STARPU_HAVE_BLAS
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_gemm},
	.cpu_funcs_name = {"cpu_gemm"},
#endif
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_gemm},
#elif defined(STARPU_USE_HIP)
	.hip_funcs = {hipblas_gemm},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.hip_flags = {STARPU_HIP_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_REDUX},
	.model = &starpu_gemm_model
};

/**
   INVALIDATE_C_TILE  Pour choisir de mettre ou non les RW dans les codelets gemm en 3D.
   To randomize tasks or their order RANDOM_TASK_ORDER (only for 2D matrix)
   RECURSIVE_MATRIX_LAYOUT (only for 2D matrix)
   RANDOM_DATA_ACCESS (only for 2D matrix)
   COUNT_DO_SCHEDULE do schedule for HFP pris en compte ou non
   SPARSE_MATRIX 0 by default.  Something else than 0 correspond to the percentage of chance of a task to be created. So SPARSE_MATRIX=10 means you a 10% of the tasks (on average). Fix STARPU_RAND_SEED if you want to have similar results among different schedulers!
*/
static void parse_args(int argc, char **argv)
{
	int i;
	int size_set = 0;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-3d") == 0)
		{
			tiled = 1;
		}

		else if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
			nslicesz = nslicesx;
			if (nslicesx == 0)
			{
				fprintf(stderr, "the number of blocks in X cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
		}

		else if (strcmp(argv[i], "-nblocksx") == 0)
		{
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			if (nslicesx == 0)
			{
				fprintf(stderr, "the number of blocks in X cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
		}

		else if (strcmp(argv[i], "-nblocksy") == 0)
		{
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
			if (nslicesy == 0)
			{
				fprintf(stderr, "the number of blocks in Y cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
		}

		else if (strcmp(argv[i], "-nblocksz") == 0)
		{
			char *argptr;
			nslicesz = strtol(argv[++i], &argptr, 10);
			if (nslicesz == 0)
			{
				fprintf(stderr, "the number of blocks in Z cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
		}

		else if (strcmp(argv[i], "-x") == 0)
		{
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
			if (xdim == 0)
			{
				fprintf(stderr, "the X dimension cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
			size_set = 1;
		}

		else if (strcmp(argv[i], "-xy") == 0)
		{
			char *argptr;
			xdim = ydim = strtol(argv[++i], &argptr, 10);
			if (xdim == 0)
			{
				fprintf(stderr, "the XY dimensions cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
			size_set = 1;
		}

		else if (strcmp(argv[i], "-xyz") == 0)
		{
			char *argptr;
			xdim = ydim = zdim = strtol(argv[++i], &argptr, 10);
			size_set = 1;
		}

		else if (strcmp(argv[i], "-xyz") == 0)
		{
			char *argptr;
			xdim = ydim = zdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-y") == 0)
		{
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
			if (ydim == 0)
			{
				fprintf(stderr, "the Y dimension cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
			size_set = 1;
		}

		else if (strcmp(argv[i], "-z") == 0)
		{
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
			if (zdim == 0)
			{
				fprintf(stderr, "the Z dimension cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
			size_set = 1;
		}

		else if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			xdim = ydim = zdim = strtol(argv[++i], &argptr, 10);
			if (xdim == 0)
			{
				fprintf(stderr, "the size cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
			size_set = 1;
		}

		else if (strcmp(argv[i], "-iter") == 0)
		{
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
			if (niter == 0)
			{
				fprintf(stderr, "the number of iterations cannot be 0!\n");
				exit(EXIT_FAILURE);
			}
		}

		else if (strcmp(argv[i], "-nsleeps") == 0)
		{
			char *argptr;
			nsleeps = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-bound") == 0)
		{
			bound = 1;
		}

		else if (strcmp(argv[i], "-invalidate-c-tile") == 0)
		{
			invalidate_c_tile = 1;
		}

		else if (strcmp(argv[i], "-random-task-order") == 0)
		{
			random_task_order = 1;
		}

		else if (strcmp(argv[i], "-random-data-access") == 0)
		{
			random_data_access = 1;
		}

		else if (strcmp(argv[i], "-recursive-matrix-layout") == 0)
		{
			recursive_matrix_layout = 1;
		}

		else if (strcmp(argv[i], "-no-count-do-schedule") == 0)
		{
			count_do_schedule = 0;
		}

		else if (strcmp(argv[i], "-sparse-matrix") == 0)
		{
			char *argptr;
			sparse_matrix = strtol(argv[++i], &argptr, 10);
			if (sparse_matrix > 100)
			{
				fprintf(stderr, "incorrect value %u for sparse-matrix parameter!\n", sparse_matrix);
				exit(EXIT_FAILURE);
			}
			if (sparse_matrix != 0)
			{
				chance_to_be_created = sparse_matrix;
			}
		}

		else if (strcmp(argv[i], "-hostname") == 0)
		{
			print_hostname = 1;
		}

		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-spmd") == 0)
		{
			cl_gemm0.type = STARPU_SPMD;
		}

		else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			fprintf(stderr,"Usage: %s [-3d] [-nblocks n] [-nblocksx x] [-nblocksy y] [-nblocksz z] [-x x] [-y y] [-xy n] [-z z] [-xyz n] [-size size] [-iter iter] [-bound] [-check] [-spmd] [-hostname] [-nsleeps nsleeps]\n", argv[0]);
			if (tiled)
				fprintf(stderr,"Currently selected: %ux%u * %ux%u and %ux%ux%u blocks (size %ux%u length %u), %u iterations, %u sleeps\n", zdim, ydim, xdim, zdim, nslicesx, nslicesy, nslicesz, xdim / nslicesx, ydim / nslicesy, zdim / nslicesz, niter, nsleeps);
			else
				fprintf(stderr,"Currently selected: %ux%u * %ux%u and %ux%u blocks (size %ux%u length %u), %u iterations, %u sleeps\n", zdim, ydim, xdim, zdim, nslicesx, nslicesy, xdim / nslicesx, ydim / nslicesy, zdim, niter, nsleeps);
			exit(EXIT_SUCCESS);
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			exit(EXIT_FAILURE);
		}
	}

#ifndef STARPU_SIMGRID
	if (check && !size_set)
	{
		/* Check is sequential, reduce its default duration */
		xdim /= 2;
		ydim /= 2;
	}
#endif

#ifdef STARPU_QUICK_CHECK
	niter /= 10;
	if(niter==0)
		niter=1;
#endif

}

#define check_evicted(main_handle, i1, i2) do {	\
	if (index++ < next_evicted) \
		continue; \
	int is_allocated; \
	starpu_data_handle_t sub_handle = starpu_data_get_sub_data(main_handle, 2, i1, i2); \
	starpu_data_query_status(sub_handle, node, &is_allocated, NULL, NULL); \
	if (is_allocated && starpu_data_can_evict(sub_handle, node, is_prefetch)) \
	{								\
		next_evicted = index; \
		FPRINTF(stderr,"evicting %p\n", sub_handle); \
		return sub_handle; \
	} \
} while(0)

/* Don't do this at home, kids, this is really dumb!  */
starpu_data_handle_t dumb_victim_selector(starpu_data_handle_t *toload, unsigned node, enum starpu_is_prefetch is_prefetch)
{
	static unsigned next_evicted; // index of next data to evict, to avoid getting stuck. Yes this is awful.
	unsigned index = 0;

	if (tiled)
	{
		if (next_evicted == nslicesy*nslicesz + nslicesx+nslicesz + nslicesx*nslicesy)
			next_evicted = 0;

		unsigned x, y, z;
		for (y = 0; y < nslicesy; y++)
			for (z = 0; z < nslicesz; z++)
				check_evicted(A_handle, z, y);

		for (x = 0; x < nslicesx; x++)
			for (z = 0; z < nslicesz; z++)
				check_evicted(B_handle, x, z);

		for (x = 0; x < nslicesx; x++)
			for (y = 0; y < nslicesy; y++)
				check_evicted(C_handle, x, y);
	}
	else
	{
		if (next_evicted == 3*nslicesx*nslicesy)
			next_evicted = 0;

		unsigned x, y;
		for (x = 0; x < nslicesx; x++)
			for (y = 0; y < nslicesy; y++)
				check_evicted(A_handle, 1, y);

		for (x = 0; x < nslicesx; x++)
			for (y = 0; y < nslicesy; y++)
				check_evicted(B_handle, 1, x);

		for (x = 0; x < nslicesx; x++)
			for (y = 0; y < nslicesy; y++)
				check_evicted(C_handle, x, y);
	}

	FPRINTF(stderr,"uh, no evictable data\n");
	next_evicted = 0;
	return NULL;
}

int data_evict_from_non_cpus(starpu_data_handle_t handle)
{
	int global_ret=0;
	unsigned nodeid;
	for (nodeid = 0; nodeid < starpu_memory_nodes_get_count(); nodeid++)
	{
		if (starpu_node_get_kind(nodeid) != STARPU_CPU_RAM)
		{
			int ret = starpu_data_evict_from_node(handle, nodeid);
			if (ret != 0)
				global_ret = ret;
		}
	}
	return global_ret;
}

#define SCHEDULE_WAIT() do { 	    \
	if (count_do_schedule == 0) \
	{ \
		starpu_do_schedule(); \
		start = starpu_timing_now(); \
		starpu_resume(); \
		starpu_task_wait_for_all(); \
		end = starpu_timing_now(); \
	} \
	else \
	{ \
		start = starpu_timing_now(); \
		starpu_do_schedule(); \
		starpu_resume(); \
		starpu_task_wait_for_all(); \
		end = starpu_timing_now(); \
	}} while(0)

static int run_data(void)
{
	PRINTF("# ");
	if (print_hostname)
		PRINTF("node\t");
	PRINTF("x\ty\tz\tms\tGFlops\tDeviance");
	if (bound)
		PRINTF("\tTms\tTGFlops\tTims\tTiGFlops\tTDeviance");
	PRINTF("\n");

	starpu_seed(0);

	unsigned sleeps;
	for(sleeps = 0; sleeps < nsleeps; sleeps++)
	{
		if (bound)
			starpu_bound_start(0, 0);

		starpu_fxt_start_profiling();
		double start, end;
		//start = starpu_timing_now(); /* Moved before starpu_resume so we don't start time during scheduling */
		double timing = 0;
		double timing_square = 0;

		/* Matrice 3D */
		if (tiled)
		{
			unsigned iter;
			for (iter = 0; iter < niter; iter++)
			{
				starpu_pause(); /* To get all tasks at once */
				unsigned x,y;
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					starpu_data_handle_t Ctile = starpu_data_get_sub_data(C_handle, 2, x, y);
					if (invalidate_c_tile == 1)
					{
						starpu_data_invalidate(Ctile); /* Modifie les perfs pour DMDAR, à N>35 cela plombe ces performances au niveau de EAGER. La raison est l'allocation. */
					}
					unsigned z;
					for (z = 0; z < nslicesz; z++)
					{
						/* Ajout pour sparse matrix. */
						if (random()%100 < chance_to_be_created)
						{
							struct starpu_codelet *cl;
							cl = (z == 0) ? &cl_gemm0 : &cl_gemm;
							int ret = starpu_task_insert(cl,
										     cl->modes[0], starpu_data_get_sub_data(A_handle, 2, z, y),
										     cl->modes[1], starpu_data_get_sub_data(B_handle, 2, x, z),
										     cl->modes[2], Ctile,
										     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * (zdim/nslicesz)),
										     0);
							if (ret == -ENODEV)
							{
								check = 0;
								starpu_resume();
								return 77;
							}
							STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
						}
					}
					starpu_data_wont_use(Ctile);
				}

				SCHEDULE_WAIT();

				if (niter > 1)
				{
					if (iter != 0)
					{
						timing += end - start;
						timing_square += (end-start) * (end-start);
					}

					for (x = 0; x < nslicesx; x++)
						for (y = 0; y < nslicesy; y++)
						{
							data_evict_from_non_cpus(starpu_data_get_sub_data(C_handle, 2, x, y));

							unsigned z;
							for (z = 0; z < nslicesz; z++)
							{
								data_evict_from_non_cpus(starpu_data_get_sub_data(A_handle, 2, z, y));
								data_evict_from_non_cpus(starpu_data_get_sub_data(B_handle, 2, x, z));
							}
						}
				}
				else
				{
					timing = end - start;
				}
			}
		}
		else if (random_task_order == 1 && recursive_matrix_layout == 0 && random_data_access == 0)
		{
			/* Randomize the order in which task are sent, but the tasks are the same */
			unsigned tab_x[nslicesx][nslicesx];
			unsigned tab_y[nslicesy][nslicesy];
			unsigned iter;
			for (iter = 0; iter < niter; iter++)
			{
				unsigned i, j;
				for (i=0; i < nslicesx; i++)
					for (j = 0; j < nslicesx; j++)
						tab_x[i][j] = i;
				for (i=0; i < nslicesy; i++)
					for (j = 0; j < nslicesy; j++)
						tab_y[i][j] = j;

				//Shuffle
				for(i=0; i<nslicesx*nslicesy; i++)
				{
					unsigned k = i;
					k += random() % ((nslicesx*nslicesy) - i);
					unsigned temp = tab_x[i%nslicesx][i/nslicesx];
					tab_x[i%nslicesx][i/nslicesx] = tab_x[k%nslicesx][k/nslicesx];
					tab_x[k%nslicesx][k/nslicesx] = temp;
					temp = tab_y[i%nslicesy][i/nslicesy];
					tab_y[i%nslicesy][i/nslicesy] = tab_y[k%nslicesy][k/nslicesy];
					tab_y[k%nslicesy][k/nslicesy] = temp;
				}

				starpu_pause();
				for (i = 0; i < nslicesx; i++)
				{
					for (j = 0; j < nslicesy; j++)
					{
						if (random()%100 < chance_to_be_created)
						{
							int ret = starpu_task_insert(&cl_gemm2d,
										     cl_gemm2d.modes[0], starpu_data_get_sub_data(A_handle, 1, tab_y[i][j]),
										     cl_gemm2d.modes[1], starpu_data_get_sub_data(B_handle, 1, tab_x[i][j]),
										     cl_gemm2d.modes[2], starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]),
										     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim),
										     0);
							if (ret == -ENODEV)
							{
								starpu_resume();
								return 77;
							}
							STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
							starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]));
						}
					}
				}

				SCHEDULE_WAIT();

				if (niter > 1)
				{
					if (iter != 0)
					{
						timing += end - start;
						timing_square += (end-start) * (end-start);
					}

					for (i = 0; i < nslicesx; i++)
						for (j = 0; j < nslicesy; j++)
						{
							data_evict_from_non_cpus(starpu_data_get_sub_data(A_handle, 1, j));
							data_evict_from_non_cpus(starpu_data_get_sub_data(B_handle, 1, i));
						}
				}
				else
				{
					timing = end - start;
				}
			}
			//End if RANDOM_TASK_ORDER == 1
		}
		else if (recursive_matrix_layout == 1 && random_data_access == 0)
		{
			/* Tasks arrive in a "Z-order" */
			unsigned tab_x[nslicesx][nslicesx];
			unsigned tab_y[nslicesy][nslicesy];
			unsigned iter;
			for (iter = 0; iter < niter; iter++)
			{
				unsigned i, j;
				for (i= 0; i < nslicesx; i++)
					for (j = 0; j < nslicesx; j++)
						tab_x[i][j] = i;
				for (i= 0; i < nslicesy; i++)
					for (j = 0; j < nslicesy; j++)
						tab_y[i][j] = j;

				for (i= 0; i < nslicesx; i++)
				{
					int x_z_layout, x_z_layout_i;
					int i_bis = 0;
					for (j = 0; j < nslicesx; j++)
					{
						if (i_bis%2 == 1)
						{
							x_z_layout_i = nslicesx/2;
						}
						if (j >= 4)
						{
							x_z_layout = (j/4)*2;
						}
						tab_x[i][j] = j%2 + x_z_layout + x_z_layout_i;
					}
					x_z_layout = 0;
					x_z_layout_i = 0;
					if (i%2 == 1)
					{
						i_bis++;
					}
				}

				for (i= 0; i < nslicesy; i++)
				{
					int y_z_layout_i = 0; int i_bis = 0; int y_z_layout = 0;
					for (j = 0; j < nslicesy; j++)
					{
						int j_bis = 0;
						if (i >= 4)
						{
							y_z_layout_i = 4*(i/4);
						}
						if (j_bis%2 == 1)
						{
							y_z_layout = 1;
						}
						if (i%2 == 1)
						{
							y_z_layout += 2;
						}
						tab_y[i][j] = y_z_layout + y_z_layout_i;
						if (j%2 == 1)
						{
							j_bis++;
						}
						y_z_layout = 0;
						y_z_layout_i = 0;
					}
					y_z_layout = 0;
					if (i%2 == 1)
					{
						i_bis++;
					}
				}

				starpu_pause();
				for (i = 0; i < nslicesx; i++)
				{
					for (j = 0; j < nslicesy; j++)
					{
						if (random()%100 < chance_to_be_created)
						{
							int ret = starpu_task_insert(&cl_gemm2d,
										     cl_gemm2d.modes[0], starpu_data_get_sub_data(A_handle, 1, tab_y[i][j]),
										     cl_gemm2d.modes[1], starpu_data_get_sub_data(B_handle, 1, tab_x[i][j]),
										     cl_gemm2d.modes[2], starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]),
										     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim),
										     0);
							if (ret == -ENODEV)
							{
								starpu_resume();
								return 77;
							}
							STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
							starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]));
						}
					}
				}

				SCHEDULE_WAIT();

				if (iter != 0)
				{
					timing += end - start;
					timing_square += (end-start) * (end-start);
				}
			}
			//End If RECURSIVE_MATRIX_LAYOUT == 1
		}
		/* This is the random 2D matrix operation we use */
		else if (random_data_access == 1)
		{
			/* Each task takes as data a random line and a random column from A and B */
			unsigned iter;
			for (iter = 0; iter < niter; iter++)
			{
				starpu_pause();
				unsigned x, y;
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					if (random()%100 < chance_to_be_created)
					{
						int ret = starpu_task_insert(&cl_gemm2d,
									     cl_gemm2d.modes[0], starpu_data_get_sub_data(A_handle, 1, random()%nslicesy),
									     cl_gemm2d.modes[1], starpu_data_get_sub_data(B_handle, 1, random()%nslicesx),
									     cl_gemm2d.modes[2], starpu_data_get_sub_data(C_handle, 2, x, y),
									     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim),
									     0);
						if (ret == -ENODEV)
						{
							starpu_resume();
							return 77;
						}
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
						starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, x, y));
					}
				}

				SCHEDULE_WAIT();

				/* If I have more than 1 iteration I want the mean timing, else I don't */
				if (niter > 1)
				{
					if (iter != 0)
					{
						timing += end - start;
						timing_square += (end-start) * (end-start);
					}

					for (x = 0; x < nslicesx; x++)
					for (y = 0; y < nslicesy; y++)
					{
						data_evict_from_non_cpus(starpu_data_get_sub_data(A_handle, 1, y));
						data_evict_from_non_cpus(starpu_data_get_sub_data(B_handle, 1, x));
					}
				}
				else
				{
					timing = end - start;
				}
			}
		}
		else
		{
			/* Normal execution of xgemm */
			unsigned iter;
			for (iter = 0; iter < niter; iter++)
			{
				starpu_pause();
				unsigned x,y;
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					if (random()%100 < chance_to_be_created)
					{
						int ret = starpu_task_insert(&cl_gemm2d,
									     cl_gemm2d.modes[0], starpu_data_get_sub_data(A_handle, 1, y),
									     cl_gemm2d.modes[1], starpu_data_get_sub_data(B_handle, 1, x),
									     cl_gemm2d.modes[2], starpu_data_get_sub_data(C_handle, 2, x, y),
									     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim),
									     0);
						if (ret == -ENODEV)
						{
							starpu_resume();
							return 77;
						}
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
						starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, x, y));
					}
				}

				SCHEDULE_WAIT();

				if (niter > 1)
				{
					if (iter != 0)
					{
						timing += end - start;
						timing_square += (end-start) * (end-start);
					}

					for (x = 0; x < nslicesx; x++)
					for (y = 0; y < nslicesy; y++)
					{
						data_evict_from_non_cpus(starpu_data_get_sub_data(A_handle, 1, y));
						data_evict_from_non_cpus(starpu_data_get_sub_data(B_handle, 1, x));
					}
				}
				else
				{
					timing = end - start;
				}

				starpu_reset_scheduler();
			}
			/* End of normal execution of 2D matrix. */
		}

		starpu_fxt_stop_profiling();

		if (bound)
			starpu_bound_stop();

		double min, min_int;
		if (bound)
			starpu_bound_compute(&min, &min_int, 1);

		if (print_hostname)
		{
			char hostname[255];
			gethostname(hostname, 255);
			PRINTF("%s\t", hostname);
		}

		/* Don't count first iteration */
		niter--;
		if (niter+1 > 1) /* We also print the deviance */
		{
			double flops = 2.0 * ((unsigned long long)(niter)) * ((unsigned long long)xdim) * ((unsigned long long)ydim) * ((unsigned long long)zdim);
			/* Cas sparse je divise les flops */
			if (sparse_matrix != 0)
			{
				flops = (flops*sparse_matrix)/100;
			}
			double average = timing/niter;
			double deviation = sqrt(fabs(timing_square / niter - average*average));
			PRINTF("%u\t%u\t%u\t%.0f\t%.1f\t%f", xdim, ydim, zdim, timing/niter/1000.0, flops/timing/1000.0, flops/niter/(average*average)*deviation/1000.0);
			if (bound)
				PRINTF("\t%.0f\t%.1f\t%.0f\t%.1f\t%f", min, flops/min/1000000.0, min_int, flops/min_int/1000000.0, flops/niter/(average*average)*deviation/1000.0);
			PRINTF("\n");
		}
		else /* We don't */
		{
			double flops = 2.0 * ((unsigned long long)(niter+1)) * ((unsigned long long)xdim) * ((unsigned long long)ydim) * ((unsigned long long)zdim);
			PRINTF("%u\t%u\t%u\t%.0f\t%.1f\t%f", xdim, ydim, zdim, timing/(niter+1)/1000.0, flops/timing/1000.0, 0.0);
			if (bound)
				PRINTF("\t%.0f\t%.1f\t%.0f\t%.1f\t%f", min, flops/min/1000000.0, min_int, flops/min_int/1000000.0, 0.0);
			PRINTF("\n");
		}

		if (sleeps < nsleeps-1)
		{
			sleep(10);
		}
	}
	return 0;
}
