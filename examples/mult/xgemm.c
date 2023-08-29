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
#ifndef TYPE
#error "Do not compile xgemm.c directly, compile sgemm.c or dgemm.c"
#endif

#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <starpu.h>
#include <starpu_fxt.h>

#ifdef STARPU_HAVE_BLAS
#include <common/blas.h>
#endif

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <starpu_cublas_v2.h>
static const TYPE p1 = 1.0;
static const TYPE m1 = -1.0;
static const TYPE v0 = 0.0;
#endif

#ifdef STARPU_USE_HIP
#include <hip/hip_runtime.h>
#include <starpu_hipblas.h>
static const TYPE p1 = 1.0;
static const TYPE m1 = -1.0;
static const TYPE v0 = 0.0;
#endif

#ifdef STARPU_QUICK_CHECK
static unsigned niter = 2;
#else
static unsigned niter = 10;
#endif
static unsigned nsleeps = 1;
static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
static unsigned nslicesz = 4;
#if defined(STARPU_QUICK_CHECK) && !defined(STARPU_SIMGRID)
static unsigned xdim = 256;
static unsigned ydim = 256;
static unsigned zdim = 64;
#else
static unsigned xdim = 960*4;
static unsigned ydim = 960*4;
static unsigned zdim = 960*4;
#endif
static unsigned size_set = 0;
static unsigned check = 0;
static unsigned bound = 0;
static unsigned print_hostname = 0;
static unsigned tiled = 0;

static TYPE *A, *B, *C;
static starpu_data_handle_t A_handle, B_handle, C_handle;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)

#ifdef STARPU_HAVE_BLAS
static int check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", ydim, xdim, zdim, (TYPE)-1.0f, A, ydim, B, zdim, (TYPE)1.0f, C, ydim);

	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(xdim*ydim, C, 1);

	if (err < EPSILON*xdim*ydim*zdim)
	{
		FPRINTF(stderr, "Results are OK\n");
		return 0;
	}
	else
	{
		int max;
		max = CPU_IAMAX(xdim*ydim, C, 1);

		FPRINTF(stderr, "There were errors ... err = %f\n", err);
		FPRINTF(stderr, "Max error : %e\n", C[max]);
		return 1;
	}
}
#endif

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

	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A,
		ydim, ydim, zdim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B,
		zdim, zdim, xdim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C,
		ydim, ydim, xdim, sizeof(TYPE));

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
			&p1, subA, ldA, subB, ldB,
			beta, subC, ldC);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}

static void cublas_gemm0(void *descr[], void *arg)
{
	cublas_mult(descr, arg, &v0);
}

static void cublas_gemm(void *descr[], void *arg)
{
	cublas_mult(descr, arg, &p1);
}
#endif

#ifdef STARPU_USE_HIP
static void hipblas_mult(void *descr[], void *arg, const TYPE *beta)
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

        hipblasStatus_t status = HIPBLAS_GEMM(starpu_hipblas_get_local_handle(),
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        nxC, nyC, nyA,
                        &p1, subA, ldA, subB, ldB,
                        beta, subC, ldC);
        if (status != HIPBLAS_STATUS_SUCCESS)
                STARPU_HIPBLAS_REPORT_ERROR(status);
}

static void hipblas_gemm0(void *descr[], void *arg)
{
        hipblas_mult(descr, arg, &v0);
}

static void hipblas_gemm(void *descr[], void *arg)
{
        hipblas_mult(descr, arg, &p1);
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

void cpu_gemm0(void *descr[], void *arg)
{
	cpu_mult(descr, arg, 0.);
}

void cpu_gemm(void *descr[], void *arg)
{
	cpu_mult(descr, arg, 1.);
}
#endif

static struct starpu_perfmodel starpu_gemm_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = STARPU_GEMM_STR(gemm)
};

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
#elif defined(STARPU_USE_HIP)
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
}

int main(int argc, char **argv)
{
	double start, end;
	int ret;

	parse_args(argc, argv);

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

	starpu_fxt_autostart_profiling(0);
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_cublas_init();
	starpu_hipblas_init();

	init_problem_data();
	partition_mult_data();

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
                start = starpu_timing_now();

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
						ret = starpu_task_insert(cl,
									 cl->modes[0], starpu_data_get_sub_data(A_handle, 2, z, y),
									 cl->modes[1], starpu_data_get_sub_data(B_handle, 2, x, z),
									 cl->modes[2], Ctile,
									 STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * (zdim/nslicesz)),
									 0);
						if (ret == -ENODEV)
						{
							check = 0;
							ret = 77;
							goto enodev;
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
					ret = starpu_task_insert(&cl_gemm0,
								 cl_gemm0.modes[0], starpu_data_get_sub_data(A_handle, 1, y),
								 cl_gemm0.modes[1], starpu_data_get_sub_data(B_handle, 1, x),
								 cl_gemm0.modes[2], starpu_data_get_sub_data(C_handle, 2, x, y),
								 STARPU_FLOPS, (double) (2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim),
								 0);
					if (ret == -ENODEV)
					{
						check = 0;
						ret = 77;
						goto enodev;
					}
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
					starpu_data_wont_use(starpu_data_get_sub_data(C_handle, 2, x, y));
				}
			}

			starpu_task_wait_for_all();
		}

		end = starpu_timing_now();
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

enodev:
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

#ifdef STARPU_HAVE_BLAS
#ifndef STARPU_SIMGRID
	if (check)
		ret = check_output();
#endif
#endif

	starpu_free_flags(A, zdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, xdim*zdim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, xdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

	starpu_cublas_shutdown();
	starpu_hipblas_shutdown();
	starpu_shutdown();

	return ret;
}
