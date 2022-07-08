/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020, 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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


#define TYPE	float

#define CUBLAS_GEMM cublasSgemm
#define CPU_GEMM	STARPU_SGEMM
#define CPU_ASUM	STARPU_SASUM
#define CPU_IAMAX	STARPU_ISAMAX
#define STARPU_GEMM(name)	starpu_sgemm_##name

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_GEMM_STR(name)  xstr(STARPU_GEMM(name))


#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <starpu.h>
#include <starpu_fxt.h>

#include <common/blas.h>

static int do_start_trs_mod=2;

static int _do_start_transaction(int val)
{
	if (do_start_trs_mod == 0)
	{
		return 0;
	}
	return ((val+1) % do_start_trs_mod == 0);
}

int do_start_transaction(void *descr, void *arg)
{
	int val = (int)(intptr_t)arg;
	int ret = _do_start_transaction(val);
	return ret;
}

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <starpu_cublas_v2.h>
static const TYPE p1 = 1.0;
static const TYPE m1 = -1.0;
static const TYPE v0 = 0.0;
#endif

static unsigned niter = 10;
static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
#if defined(STARPU_QUICK_CHECK) && !defined(STARPU_SIMGRID)
static unsigned xdim = 256;
static unsigned ydim = 256;
static unsigned zdim = 64;
#else
static unsigned xdim = 960*4;
static unsigned ydim = 960*4;
static unsigned zdim = 960*4;
#endif
static unsigned check = 0;

static TYPE *A, *B, *C;
static starpu_data_handle_t A_handle, B_handle, C_handle;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); }} while(0)

static void check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", ydim, xdim, zdim, (TYPE)-1.0f, A, ydim, B, zdim, (TYPE)1.0f, C, ydim);

	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(xdim*ydim, C, 1);

	if (err < xdim*ydim*0.001)
	{
		FPRINTF(stderr, "Results are OK\n");
	}
	else
	{
		int max;
		max = CPU_IAMAX(xdim*ydim, C, 1);

		FPRINTF(stderr, "There were errors ... err = %f\n", err);
		FPRINTF(stderr, "Max error : %e\n", C[max]);
	}
}

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

	starpu_data_partition(B_handle, &vert);
	starpu_data_partition(A_handle, &horiz);

	starpu_data_map_filters(C_handle, 2, &vert, &horiz);
}

#ifdef STARPU_USE_CUDA
static void cublas_mult(void *descr[], void *arg)
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
					    &v0, subC, ldC);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void cpu_mult(void *descr[], void *arg)
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
		CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB, (TYPE)0.0, subC, ldC);
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

		CPU_GEMM("N", "N", nxC, new_nyC, nyA, (TYPE)1.0, subA, ldA, new_subB, ldB, (TYPE)0.0, new_subC, ldC);
	}
}

static struct starpu_perfmodel starpu_gemm_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = STARPU_GEMM_STR(gemm)
};

static struct starpu_codelet cl =
{
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_mult},
	.cpu_funcs_name = {"cpu_mult"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_mult},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = STARPU_VARIABLE_NBUFFERS, /* required for transactions */
	/* .modes = {STARPU_R, STARPU_R, STARPU_RW}, */
	.model = &starpu_gemm_model
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
		}

		else if (strcmp(argv[i], "-nblocksx") == 0)
		{
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nblocksy") == 0)
		{
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-x") == 0)
		{
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-y") == 0)
		{
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-z") == 0)
		{
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			xdim = ydim = zdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-iter") == 0)
		{
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
		}

		/* Modulo operand to decide which iterations to confirm or cancel.
		 *
		 * An iteration will be confirmed if ((iter+1) % MOD) == 0, and
		 * cancelled otherwise. As a special value, if MOD == 0, all
		 * iterations will be cancelled. */
		else if (strcmp(argv[i], "-mod") == 0)
		{
			char *argptr;
			do_start_trs_mod = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-spmd") == 0)
		{
			cl.type = STARPU_SPMD;
		}

		else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			fprintf(stderr,"Usage: %s [-nblocks n] [-nblocksx x] [-nblocksy y] [-x x] [-y y] [-z z] [-size size] [-iter iter] [-check] [-spmd] [-mod start_trs_mod]\n", argv[0]);
			fprintf(stderr,"Currently selected: %ux%u * %ux%u and %ux%u blocks, %u iterations, transaction confirmation modulo = %d\n", zdim, ydim, xdim, zdim, nslicesx, nslicesy, niter, do_start_trs_mod);
			exit(EXIT_SUCCESS);
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s", argv[i]);
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv)
{
	double start, end;
	int ret;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.nmpi_ms = 0;
	conf.ntcpip_ms = 0;

	parse_args(argc, argv);

	starpu_fxt_autostart_profiling(0);
	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_cublas_init();

	init_problem_data();
	partition_mult_data();

	starpu_fxt_start_profiling();
	start = starpu_timing_now();

	unsigned x, y, iter;
	struct starpu_transaction *transaction = starpu_transaction_open(do_start_transaction, (void*)(intptr_t)0);
	if (transaction == NULL)
	{
		starpu_cublas_shutdown();
		starpu_shutdown();
		return 77; /* transaction begin task submit failed with ENODEV */
	}

	int simulated_transaction_status = _do_start_transaction(0);
	int n_cancelled_iter=0;

	for (iter = 0; iter < niter; iter++)
	{
		if (iter > 0)
		{
			starpu_transaction_next_epoch(transaction, (void*)(intptr_t)iter);
			simulated_transaction_status = _do_start_transaction(iter);
		}
		if (simulated_transaction_status == 0)
		{
			n_cancelled_iter++;
		}
		for (x = 0; x < nslicesx; x++)
		{
			for (y = 0; y < nslicesy; y++)
			{
				struct starpu_task *task = starpu_task_create();

				task->cl = &cl;

				task->handles[0] = starpu_data_get_sub_data(A_handle, 1, y);
				task->handles[1] = starpu_data_get_sub_data(B_handle, 1, x);
				task->handles[2] = starpu_data_get_sub_data(C_handle, 2, x, y);
				task->modes[0] = STARPU_R;
				task->modes[1] = STARPU_R;
				task->modes[2] = STARPU_RW;
				task->nbuffers = 3;
				task->transaction = transaction;

				task->flops = 2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim;

				ret = starpu_task_submit(task);
				if (ret == -ENODEV)
				{
					ret = 77;
					goto enodev;
				}
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
				starpu_data_wont_use(starpu_data_get_sub_data(C_handle, 2, x, y));
			}
		}

	}
	starpu_transaction_close(transaction);
	starpu_task_wait_for_all();

	end = starpu_timing_now();
	starpu_fxt_stop_profiling();

	double timing = end - start;

	PRINTF("# x\ty\tz\ts\tniter\tncancelled");
	PRINTF("\n");
	PRINTF("%u\t%u\t%u\t%.3f\t%d\t%d", xdim, ydim, zdim, timing/1.0e6, niter, n_cancelled_iter);
	PRINTF("\n");

enodev:
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

	if (check)
		check_output();

	starpu_free_flags(A, zdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, xdim*zdim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, xdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

	starpu_cublas_shutdown();
	starpu_shutdown();

	return ret;
}
