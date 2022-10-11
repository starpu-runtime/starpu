/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * sendrecv benchmark from different tasks, executed simultaneously on serveral
 * workers.
 * Inspired a lot from NewMadeleine examples/piom/nm_piom_pingpong.c
 *
 * The goal is to measure impact of calls to starpu_mpi_* from different threads.
 *
 * Use STARPU_NCPU to set the number of parallel ping pongs
 *
 *
 * Note: This currently can not work with the MPI backend with more than 1 CPU,
 * since with big sizes, the MPI_Wait call in the MPI thread may block waiting
 * for the peer to call MPI_Recv+Wait, and there is no guarantee that the peer
 * will call MPI_Recv+Wait for the same data since tasks can proceed in any
 * order.
 */

#include <starpu_mpi.h>
#include "helper.h"
#include "bench_helper.h"

#define NB_WARMUP_PINGPONGS 10

/* We reduce NX_MAX, since some NICs don't support exchanging simultaneously such amount of memory */
#undef NX_MAX
#ifdef STARPU_QUICK_CHECK
#define NX_MAX (1024)
#else
#define NX_MAX (64 * 1024 * 1024)
#endif


void cpu_task(void* descr[], void* args)
{
	int mpi_rank;
	uint64_t iterations =
#ifdef STARPU_QUICK_CHECK
		10;
#else
	LOOPS_DEFAULT / 100;
#endif
	uint64_t s;
	starpu_data_handle_t handle_send, handle_recv;
	double t1, t2;
	int asked_worker;
	int current_worker = starpu_worker_get_id();
	uint64_t j;
	uint64_t k;
	int ret;

	starpu_codelet_unpack_args(args, &mpi_rank, &asked_worker, &s, &handle_send, &handle_recv);

	STARPU_ASSERT(asked_worker == current_worker);

	iterations = bench_nb_iterations(iterations, s);
	double* lats = malloc(sizeof(double) * iterations);

	for (j = 0; j < NB_WARMUP_PINGPONGS; j++)
	{
		if (mpi_rank == 0)
		{
			ret = starpu_mpi_send(handle_send, 1, 0, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
			ret = starpu_mpi_recv(handle_recv, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		}
		else
		{
			ret = starpu_mpi_recv(handle_recv, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
			ret = starpu_mpi_send(handle_send, 0, 1, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		}
	}

	for (j = 0; j < iterations; j++)
	{
		if (mpi_rank == 0)
		{
			t1 = starpu_timing_now();
			ret = starpu_mpi_send(handle_send, 1, 0, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
			ret = starpu_mpi_recv(handle_recv, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
			t2 = starpu_timing_now();

			lats[j] =  (t2 - t1) / 2;
		}
		else
		{
			ret = starpu_mpi_recv(handle_recv, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
			ret = starpu_mpi_send(handle_send, 0, 1, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		}
	}

	if (mpi_rank == 0)
	{
		qsort(lats, iterations, sizeof(double), &comp_double);

		const double min_lat = lats[0];
		const double max_lat = lats[iterations - 1];
		const double med_lat = lats[(iterations - 1) / 2];
		const double d1_lat = lats[(iterations - 1) / 10];
		const double d9_lat = lats[9 * (iterations - 1) / 10];
		double avg_lat = 0.0;

		for(k = 0; k < iterations; k++)
		{
			avg_lat += lats[k];
		}

		avg_lat /= iterations;
		const double bw_million_byte = s / min_lat;
		const double bw_mbyte        = bw_million_byte / 1.048576;

		printf("%2d\t\t%9lld\t%9.3lf\t%9.3f\t%9.3f\t%9.3lf\t%9.3lf\t%9.3lf\t%9.3lf\t%9.3lf\n",
			current_worker, (long long) s, min_lat, bw_million_byte, bw_mbyte, d1_lat, med_lat, avg_lat, d9_lat, max_lat);
		fflush(stdout);
	}

	free(lats);
}

static struct starpu_codelet cl =
{
	.cpu_funcs = { cpu_task },
	.cpu_funcs_name = { "cpu_task" },
	.nbuffers = 0
};

int main(int argc, char **argv)
{
	int ret, rank, worldsize;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	if (worldsize < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need 2 processes.\n");

		starpu_mpi_shutdown();

		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
	{
		printf("Times in us\n");
		printf("# worker | size  (Bytes)\t|  latency \t| 10^6 B/s \t| MB/s   \t| d1    \t|median  \t| avg    \t| d9    \t| max\n");
	}
	else if (rank >= 2)
	{
		starpu_mpi_shutdown();

		return 0;
	}


	unsigned cpu_count = starpu_cpu_worker_get_count();
	uint64_t s;
	unsigned i;

	int* workers = malloc(cpu_count * sizeof(int));
	float** vectors_send = malloc(cpu_count * sizeof(float*));
	float** vectors_recv = malloc(cpu_count * sizeof(float*));
	starpu_data_handle_t* handles_send = malloc(cpu_count * sizeof(starpu_data_handle_t));
	starpu_data_handle_t* handles_recv = malloc(cpu_count * sizeof(starpu_data_handle_t));

	for (s = NX_MIN; s <= NX_MAX; s = bench_next_size(s))
	{
		starpu_pause();

		for (i = 0; i < cpu_count; i++)
		{
			workers[i] = i;
			vectors_send[i] = malloc(s);
			vectors_recv[i] = malloc(s);
			memset(vectors_send[i], 0, s);
			memset(vectors_recv[i], 0, s);

			starpu_vector_data_register(&handles_send[i], STARPU_MAIN_RAM, (uintptr_t) vectors_send[i], s, 1);
			starpu_vector_data_register(&handles_recv[i], STARPU_MAIN_RAM, (uintptr_t) vectors_recv[i], s, 1);

			ret = starpu_task_insert(&cl,
						 STARPU_EXECUTE_ON_WORKER, workers[i],
						 STARPU_VALUE, &rank, sizeof(int),
						 STARPU_VALUE, workers + i, sizeof(int),
						 STARPU_VALUE, &s, sizeof(uint64_t),
						 STARPU_VALUE, &handles_send[i], sizeof(starpu_data_handle_t),
						 STARPU_VALUE, &handles_recv[i], sizeof(starpu_data_handle_t), 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}

		starpu_resume();
		starpu_task_wait_for_all();

		for (i = 0; i < cpu_count; i++)
		{
			starpu_data_unregister(handles_send[i]);
			starpu_data_unregister(handles_recv[i]);
			free(vectors_send[i]);
			free(vectors_recv[i]);
		}
	}

	free(workers);
	free(vectors_send);
	free(vectors_recv);
	free(handles_send);
	free(handles_recv);

	starpu_mpi_shutdown();

	return 0;
}
