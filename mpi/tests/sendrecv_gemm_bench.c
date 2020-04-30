/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Simple *not distributed* parallel GEMM implementation and sendrecv bench at the same time.
 *
 * This bench is a merge of mpi/tests/sendrecv_bench and examples/mult/sgemm
 *
 * A *non-distributed* GEMM is computed on each node, while a sendrecv bench is running,
 * completely independently. The goal is to measure the impact of worker computations on
 * communications.
 *
 * Use the -nblocks parameter to define the matrix size (matrix size = nblocks * 320), such as
 * the GEMM finishes after the sendrecv bench.
 */
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <starpu_mpi.h>
#include <starpu_fxt.h>

#include <common/blas.h>

#include "helper.h"
#include "abstract_sendrecv_bench.h"
#include "../../examples/mult/simple.h"

#define CHECK_TASK_SUBMIT(ret) do {				\
	if (ret == -ENODEV)					\
	{							\
		ret = 77;					\
		goto enodev;					\
	}							\
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");	\
} while(0)

static int mpi_rank;
static int comm_thread_cpuid = -1;
static unsigned nslices = 4;
#if defined(STARPU_QUICK_CHECK) && !defined(STARPU_SIMGRID)
static unsigned matrix_dim = 256;
#else
static unsigned matrix_dim = 320 * 4;
#endif
static unsigned check = 0;

static TYPE *A, *B, *C;
static starpu_data_handle_t A_handle, B_handle, C_handle;

static starpu_pthread_barrier_t thread_barrier;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)

static void check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", matrix_dim, matrix_dim, matrix_dim, (TYPE)-1.0f, A, matrix_dim, B, matrix_dim, (TYPE)1.0f, C, matrix_dim);

	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(matrix_dim*matrix_dim, C, 1);

	if (err < matrix_dim*matrix_dim*0.001)
	{
		FPRINTF(stderr, "Results are OK\n");
	}
	else
	{
		int max;
		max = CPU_IAMAX(matrix_dim*matrix_dim, C, 1);

		FPRINTF(stderr, "There were errors ... err = %f\n", err);
		FPRINTF(stderr, "Max error : %e\n", C[max]);
	}
}

static void init_problem_data(void)
{
#ifndef STARPU_SIMGRID
	unsigned i,j;
#endif

	starpu_malloc_flags((void **)&A, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&B, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&C, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

#ifndef STARPU_SIMGRID
	/* fill the matrices */
	for (j=0; j < matrix_dim; j++)
	{
		for (i=0; i < matrix_dim; i++)
		{
			A[j+i*matrix_dim] = (TYPE)(starpu_drand48());
			B[j+i*matrix_dim] = (TYPE)(starpu_drand48());
			C[j+i*matrix_dim] = (TYPE)(0);
		}
	}
#endif
}

static void partition_mult_data(void)
{
	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A,
		matrix_dim, matrix_dim, matrix_dim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B,
		matrix_dim, matrix_dim, matrix_dim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C,
		matrix_dim, matrix_dim, matrix_dim, sizeof(TYPE));

	struct starpu_data_filter vert;
	memset(&vert, 0, sizeof(vert));
	vert.filter_func = starpu_matrix_filter_vertical_block;
	vert.nchildren = nslices;

	struct starpu_data_filter horiz;
	memset(&horiz, 0, sizeof(horiz));
	horiz.filter_func = starpu_matrix_filter_block;
	horiz.nchildren = nslices;

	starpu_data_partition(B_handle, &vert);
	starpu_data_partition(A_handle, &horiz);

	starpu_data_map_filters(C_handle, 2, &vert, &horiz);
}


void cpu_init_matrix_random(void *descr[], void *arg)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[0]);

	for (unsigned i = 0; i < nx *ny; i++)
	{
		subA[i] = (TYPE) (starpu_drand48());
		subB[i] = (TYPE) (starpu_drand48());
	}
}


void cpu_init_matrix_zero(void *descr[], void *arg)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[0]);

	for (unsigned i = 0; i < nx *ny; i++)
	{
		subA[i] = (TYPE) (0);
	}
}


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
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &starpu_gemm_model
};

static struct starpu_codelet cl_init_matrix_random =
{
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_init_matrix_random},
	.cpu_funcs_name = {"cpu_init_matrix_random"},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_W}
};

static struct starpu_codelet cl_init_matrix_zero =
{
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_init_matrix_zero},
	.cpu_funcs_name = {"cpu_init_matrix_zero"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nslices = strtol(argv[++i], &argptr, 10);
			matrix_dim = 320 * nslices;
		}

		else if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			unsigned matrix_dim_tmp = strtol(argv[++i], &argptr, 10);
			if (matrix_dim_tmp % 320 != 0)
			{
				fprintf(stderr, "Matrix size has to be a multiple of 320\n");
			}
			else
			{
				matrix_dim = matrix_dim_tmp;
				nslices = matrix_dim / 320;
			}
		}

		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-spmd") == 0)
		{
			cl.type = STARPU_SPMD;
		}

		else if (strcmp(argv[i], "-comm-thread-cpuid") == 0)
		{
			comm_thread_cpuid = atoi(argv[++i]);
		}

		else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			fprintf(stderr,"Usage: %s [-nblocks n] [-size size] [-check] [-spmd] [-comm-thread-cpuid cpuid]\n", argv[0]);
			fprintf(stderr,"Currently selected: matrix size: %u - %u blocks\n", matrix_dim, nslices);
			fprintf(stderr, "Use -comm-thread-cpuid to specifiy where to bind the comm benchmarking thread\n");
			exit(EXIT_SUCCESS);
		}

		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			exit(EXIT_FAILURE);
		}
	}
}


static void* comm_thread_func(void* arg)
{
	if (comm_thread_cpuid < 0)
	{
		comm_thread_cpuid = starpu_get_next_bindid(STARPU_THREAD_ACTIVE, NULL, 0);
	}

	if (starpu_bind_thread_on(comm_thread_cpuid, 0, "Comm") < 0)
	{
		char hostname[65];
		gethostname(hostname, sizeof(hostname));
		_STARPU_DISP("[%s] No core was available for the comm thread. You should increase STARPU_RESERVE_NCPU or decrease STARPU_NCPU\n", hostname);
	}

	sendrecv_bench(mpi_rank, &thread_barrier);

	return NULL;
}

int main(int argc, char **argv)
{
	double start, end;
	int ret, mpi_init, worldsize;
	starpu_pthread_t comm_thread;

	char hostname[255];
	gethostname(hostname, 255);

	parse_args(argc, argv);

	starpu_fxt_autostart_profiling(0);

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &mpi_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	if (worldsize < 2)
	{
		if (mpi_rank == 0)
			FPRINTF(stderr, "We need 2 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}


	STARPU_PTHREAD_BARRIER_INIT(&thread_barrier, NULL, 2);


	// Start comm thread, benchmarking sendrecv:
	STARPU_PTHREAD_CREATE(&comm_thread, NULL, comm_thread_func, NULL);


	// Main thread will submit GEMM tasks:
	starpu_malloc_flags((void **)&A, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&B, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&C, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	partition_mult_data();


	if (mpi_rank == 0)
	{
		PRINTF("# node\tx\ty\tz\tms\tGFlops\n");
	}

	starpu_pause();

	unsigned x, y;
#ifndef STARPU_SIMGRID
	// Initialize matrices:
	for (x = 0; x < nslices; x++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl_init_matrix_random;
		task->handles[0] = starpu_data_get_sub_data(A_handle, 1, x);
		task->handles[1] = starpu_data_get_sub_data(B_handle, 1, x);
		ret = starpu_task_submit(task);
		CHECK_TASK_SUBMIT(ret);

		for (y = 0; y < nslices; y++)
		{
			task = starpu_task_create();
			task->cl = &cl_init_matrix_zero;
			task->handles[0] = starpu_data_get_sub_data(C_handle, 2, x, y);
			ret = starpu_task_submit(task);
			CHECK_TASK_SUBMIT(ret);
		}
	}
#endif

	for (x = 0; x < nslices; x++)
	for (y = 0; y < nslices; y++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		task->handles[0] = starpu_data_get_sub_data(A_handle, 1, y);
		task->handles[1] = starpu_data_get_sub_data(B_handle, 1, x);
		task->handles[2] = starpu_data_get_sub_data(C_handle, 2, x, y);
		task->flops = 2ULL * (matrix_dim/nslices) * (matrix_dim/nslices) * matrix_dim;

		ret = starpu_task_submit(task);
		CHECK_TASK_SUBMIT(ret);
		starpu_data_wont_use(starpu_data_get_sub_data(C_handle, 2, x, y));
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);
	starpu_fxt_start_profiling();

	STARPU_PTHREAD_BARRIER_WAIT(&thread_barrier);

	start = starpu_timing_now();
	starpu_resume();
	starpu_task_wait_for_all();
	end = starpu_timing_now();
	starpu_pause(); // Pause not to disturb comm thread if it isn't done

	double timing = end - start;
	double flops = 2.0*((unsigned long long)matrix_dim) * ((unsigned long long)matrix_dim)*((unsigned long long)matrix_dim);

	PRINTF("%s\t%u\t%u\t%u\t%.0f\t%.1f\n", hostname, matrix_dim, matrix_dim, matrix_dim, timing/1000.0, flops/timing/1000.0);


enodev:
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

	if (check)
		check_output();

	starpu_free_flags(A, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);


	// Wait comm thread:
	STARPU_PTHREAD_JOIN(comm_thread, NULL);
	STARPU_PTHREAD_BARRIER_DESTROY(&thread_barrier);

	starpu_fxt_stop_profiling();

	starpu_resume();
	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return ret;
}
