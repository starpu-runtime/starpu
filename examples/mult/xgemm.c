/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010-2010  Mehdi Juhoor
 * Copyright (C) 2017-2017  Erwan Leria
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
}

static void partition_mult_data(void)
{
	unsigned x, y, z;

	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, ydim, ydim, zdim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B, zdim, zdim, xdim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C, ydim, ydim, xdim, sizeof(TYPE));

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

	cublasStatus_t status = CUBLAS_GEMM(starpu_cublas_get_local_handle(),
					    CUBLAS_OP_N, CUBLAS_OP_N,
					    nxC, nyC, nyA,
					    &p1_cuda, subA, ldA, subB, ldB,
					    beta, subC, ldC);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
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
#elif defined(STARPU_USE_HIP) && defined(STARPU_USE_HIPBLAS)
	.hip_funcs = {hipblas_gemm0},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.hip_flags = {STARPU_HIP_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.model = &starpu_gemm_model
};

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
#elif defined(STARPU_USE_HIP) && defined(STARPU_USE_HIPBLAS)
	.hip_funcs = {hipblas_gemm},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.hip_flags = {STARPU_HIP_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &starpu_gemm_model
};

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

static int run_data(void)
{
	PRINTF("# ");
	if (print_hostname)
		PRINTF("node\t");
	PRINTF("x\ty\tz\tms\tGFlop/s");
	if (bound)
		PRINTF("\tTms\tTGFlop/s\tTims\tTiGFlop/s");
	PRINTF("\n");

	unsigned sleeps;
	for (sleeps = 0; sleeps < nsleeps; sleeps++)
	{
		if (bound)
			starpu_bound_start(0, 0);

	        starpu_fxt_start_profiling();
                double start = starpu_timing_now();

		unsigned x, y, z, iter;
		for (iter = 0; iter < niter; iter++)
		{
			if (tiled)
			{
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					starpu_data_handle_t Ctile = starpu_data_get_sub_data(C_handle, 2, x, y);
					for (z = 0; z < nslicesz; z++)
					{
						struct starpu_codelet *cl = z == 0 ? &cl_gemm0 : &cl_gemm;
						int ret = starpu_task_insert(cl,
									     cl->modes[0], starpu_data_get_sub_data(A_handle, 2, z, y),
									     cl->modes[1], starpu_data_get_sub_data(B_handle, 2, x, z),
									     cl->modes[2], Ctile,
									     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * (zdim/nslicesz)),
									     0);
						if (ret == -ENODEV)
						{
							check = 0;
							return 77;
						}
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					}
					starpu_data_wont_use(Ctile);
				}
			}
			else
			{
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					int ret = starpu_task_insert(&cl_gemm0,
								     cl_gemm0.modes[0], starpu_data_get_sub_data(A_handle, 1, y),
								     cl_gemm0.modes[1], starpu_data_get_sub_data(B_handle, 1, x),
								     cl_gemm0.modes[2], starpu_data_get_sub_data(C_handle, 2, x, y),
								     STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim),
								     0);
					if (ret == -ENODEV)
					{
						check = 0;
						return 77;
					}
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					starpu_data_wont_use(starpu_data_get_sub_data(C_handle, 2, x, y));
				}
			}

			starpu_task_wait_for_all();
		}

		double end = starpu_timing_now();
		starpu_fxt_stop_profiling();

		if (bound)
			starpu_bound_stop();

		double timing = end - start;
		double min, min_int;
		double flops = 2.0*((unsigned long long)(niter))*((unsigned long long)xdim)
				   *((unsigned long long)ydim)*((unsigned long long)zdim);

		if (bound)
			starpu_bound_compute(&min, &min_int, 1);

		if (print_hostname)
		{
			char hostname[255];
			gethostname(hostname, 255);
			PRINTF("%s\t", hostname);
		}
		PRINTF("%u\t%u\t%u\t%.0f\t%.1f", xdim, ydim, zdim, timing/(niter)/1000.0, flops/timing/1000.0);
		if (bound)
			PRINTF("\t%.0f\t%.1f\t%.0f\t%.1f", min, flops/min/1000000.0, min_int, flops/min_int/1000000.0);
		PRINTF("\n");

		if (sleeps < nsleeps-1)
		{
			sleep(10);
		}
	}
	return 0;
}

