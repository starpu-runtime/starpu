/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "helper.h"
#include "abstract_sendrecv_bench.h"
#include "gemm_helper.h"

static int mpi_rank;
static starpu_pthread_barrier_t thread_barrier;

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
		fprintf(stderr, "[%s] No core was available for the comm thread. You should increase STARPU_RESERVE_NCPU or decrease STARPU_NCPU\n", hostname);
	}

	int ret = sendrecv_bench(mpi_rank, &thread_barrier, /* half-duplex communications */ 0, /* allocate MPI buffers on CPU */ STARPU_MAIN_RAM);
	if (ret == -ENODEV)
	{
		fprintf(stderr, "No device available\n");
	}

	return NULL;
}

void parse_args(int argc, char **argv)
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

		else if (strcmp(argv[i], "-comm-thread-cpuid") == 0)
		{
			comm_thread_cpuid = atoi(argv[++i]);
		}

		else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			fprintf(stderr,"Usage: %s [-nblocks n] [-size size] [-check] [-comm-thread-cpuid cpuid]\n", argv[0]);
			fprintf(stderr,"Currently selected: matrix size: %u - %u blocks\n", matrix_dim, nslices);
			fprintf(stderr, "Use -comm-thread-cpuid to specify where to bind the comm benchmarking thread\n");
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
	int ret, worldsize;
	starpu_pthread_t comm_thread;

	char hostname[255];
	gethostname(hostname, 255);

	parse_args(argc, argv);

	starpu_fxt_autostart_profiling(0);

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
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

		return STARPU_TEST_SKIPPED;
	}

	STARPU_PTHREAD_BARRIER_INIT(&thread_barrier, NULL, 2);

	// Start comm thread, benchmarking sendrecv:
	STARPU_PTHREAD_CREATE(&comm_thread, NULL, comm_thread_func, NULL);

	// Main thread will submit GEMM tasks:
	gemm_alloc_data();

	if (mpi_rank == 0)
	{
		printf("# node\tx\ty\tz\tms\tGFlops\n");
	}

	starpu_pause();

	if(gemm_init_data() == -ENODEV || gemm_submit_tasks() == -ENODEV)
	{
		starpu_mpi_barrier(MPI_COMM_WORLD);
		STARPU_PTHREAD_BARRIER_WAIT(&thread_barrier);
		ret = 77;
		goto enodev;
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);
	starpu_fxt_start_profiling();

	STARPU_PTHREAD_BARRIER_WAIT(&thread_barrier);

	start = starpu_timing_now();
	starpu_resume();
	starpu_task_wait_for_all();
	end = starpu_timing_now();

	double timing = end - start;
	double flops = 2.0*((unsigned long long)matrix_dim) * ((unsigned long long)matrix_dim)*((unsigned long long)matrix_dim);

	printf("%s\t%u\t%u\t%u\t%.0f\t%.1f\n", hostname, matrix_dim, matrix_dim, matrix_dim, timing/1000.0, flops/timing/1000.0);


enodev:
	gemm_release();

	// Wait comm thread:
	STARPU_PTHREAD_JOIN(comm_thread, NULL);
	STARPU_PTHREAD_BARRIER_DESTROY(&thread_barrier);

	starpu_fxt_stop_profiling();

	if (ret)
		starpu_resume();
	starpu_mpi_shutdown();

	return ret;
}
