/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019                                     Inria
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
 *
 * Basic send receive benchmark.
 * Inspired a lot from NewMadeleine examples/benchmarks/nm_bench_sendrecv.c
 */

#include <math.h>
#include <starpu_mpi.h>
#include "helper.h"

#define NX_MAX (512 * 1024 * 1024) // kB
#define NX_MIN 0
#define MULT_DEFAULT 2
#define INCR_DEFAULT 0
#define NX_STEP 1.4 // multiplication
#define LOOPS_DEFAULT 10000

int times_nb_nodes;
int times_size;
int worldsize;

static int comp_double(const void*_a, const void*_b)
{
	const double* a = _a;
	const double* b = _b;

	if(*a < *b)
		return -1;
	else if(*a > *b)
		return 1;
	else
		return 0;
}

static inline uint64_t _next(uint64_t len, double multiplier, uint64_t increment)
{
	uint64_t next = len * multiplier + increment;

	if(next <= len)
		next++;

	return next;
}


static inline uint64_t _iterations(int iterations, uint64_t len)
{
	const uint64_t max_data = 512 * 1024 * 1024;

	if(len <= 0)
		len = 1;

	uint64_t data_size = ((uint64_t)iterations * (uint64_t)len);

	if(data_size  > max_data)
	{
		iterations = (max_data / (uint64_t)len);
		if(iterations < 2)
			iterations = 2;
	}

	return iterations;
}

int main(int argc, char **argv)
{
	int ret, rank;
	starpu_data_handle_t handle_send, handle_recv;
	int mpi_init;
	float* vector_send = NULL;
	float* vector_recv = NULL;
	double t1, t2;
	double* lats = malloc(sizeof(double) * LOOPS_DEFAULT);
	uint64_t iterations = LOOPS_DEFAULT;
	double multiplier = MULT_DEFAULT;
	uint64_t increment = INCR_DEFAULT;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	STARPU_ASSERT_MSG(worldsize == 2, "We need two prcesses.");


	if (rank == 0)
	{
		printf("Times in us\n");
		printf("# size  (Bytes)\t|  latency \t| 10^6 B/s \t| MB/s   \t| median  \t| avg    \t| max\n");
	}

	int array_size = 0;

	for (uint64_t s = NX_MIN; s <= NX_MAX; s = _next(s, multiplier, increment))
	{
		vector_send = malloc(s);
		vector_recv = malloc(s);
		memset(vector_send, 0, s);
		memset(vector_recv, 0, s);

		starpu_vector_data_register(&handle_send, STARPU_MAIN_RAM, (uintptr_t) vector_send, s, 1);
		starpu_vector_data_register(&handle_recv, STARPU_MAIN_RAM, (uintptr_t) vector_recv, s, 1);

		iterations = _iterations(iterations, s);

		starpu_mpi_barrier(MPI_COMM_WORLD);

		for (int j = 0; j < iterations; j++)
		{
			if (rank == 0)
			{
				t1 = starpu_timing_now();
				starpu_mpi_send(handle_send, 1, 0, MPI_COMM_WORLD);
				starpu_mpi_recv(handle_recv, 1, 1, MPI_COMM_WORLD, NULL);
				t2 = starpu_timing_now();

				const double delay = t2 - t1;
				const double t = delay / 2;

				lats[j] = t;
			}
			else
			{
				starpu_mpi_recv(handle_recv, 0, 0, MPI_COMM_WORLD, NULL);
				starpu_mpi_send(handle_send, 0, 1, MPI_COMM_WORLD);
			}

			starpu_mpi_barrier(MPI_COMM_WORLD);
		}

		if (rank == 0)
		{
			qsort(lats, iterations, sizeof(double), &comp_double);

			const double min_lat = lats[0];
			const double max_lat = lats[iterations - 1];
			const double med_lat = lats[(iterations - 1) / 2];
			double avg_lat = 0.0;

			for(int k = 0; k < iterations; k++)
			{
				avg_lat += lats[k];
			}

			avg_lat /= iterations;
			const double bw_million_byte = s / min_lat;
			const double bw_mbyte        = bw_million_byte / 1.048576;

			printf("%9lld\t%9.3lf\t%9.3f\t%9.3f\t%9.3lf\t%9.3lf\t%9.3lf\n",
				(long long)s, min_lat, bw_million_byte, bw_mbyte, med_lat, avg_lat, max_lat);
			fflush(stdout);
		}
		starpu_data_unregister(handle_recv);
		starpu_data_unregister(handle_send);

		free(vector_send);
		free(vector_recv);
	}

	starpu_mpi_shutdown();

	return 0;
}
