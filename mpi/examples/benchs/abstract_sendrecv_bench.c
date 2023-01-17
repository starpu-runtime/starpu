/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "bench_helper.h"
#include "abstract_sendrecv_bench.h"

void sendrecv_bench(int mpi_rank, starpu_pthread_barrier_t* thread_barrier, int bidir)
{
	uint64_t iterations = LOOPS_DEFAULT;
	uint64_t s;
	uint64_t j;
	uint64_t k;

	if (mpi_rank >= 2)
	{
		starpu_pause();
		if (thread_barrier != NULL)
		{
			STARPU_PTHREAD_BARRIER_WAIT(thread_barrier);
		}

		for (s = NX_MIN; s <= NX_MAX; s = bench_next_size(s))
		{
			iterations = bench_nb_iterations(iterations, s);

			starpu_mpi_barrier(MPI_COMM_WORLD);

			for (j = 0; j < iterations; j++)
			{
				starpu_mpi_barrier(MPI_COMM_WORLD);
			}
		}
		starpu_resume();

		return;
	}

	if (mpi_rank == 0)
	{
		printf("Times in us\n");
		printf("# size  (Bytes)\t|  latency \t| 10^6 B/s \t| MB/s   \t| d1    \t|median  \t| avg    \t| d9    \t| max\n");
	}

	starpu_data_handle_t handle_send, handle_recv;
	float* vector_send = NULL;
	float* vector_recv = NULL;
	double t1, t2, global_tstart, global_tend;
	double* lats = malloc(sizeof(double) * LOOPS_DEFAULT);
	starpu_mpi_req send_req, recv_req;
	int ret;

	if (thread_barrier != NULL)
	{
		STARPU_PTHREAD_BARRIER_WAIT(thread_barrier);
	}

	global_tstart = starpu_timing_now();
	for (s = NX_MIN; s <= NX_MAX; s = bench_next_size(s))
	{
		enum starpu_worker_archtype worker_type;

		if(starpu_cuda_worker_get_count()>0)
		{
			worker_type = STARPU_CUDA_WORKER;
		}
		else if (starpu_hip_worker_get_count()>0)
		{
			worker_type = STARPU_HIP_WORKER;
		}
		else
		{
			worker_type = STARPU_CPU_WORKER;
		}

		int worker_id = starpu_worker_get_by_type(worker_type, 0);
		int mem_node  = starpu_worker_get_memory_node(worker_id);

		vector_send = starpu_malloc_on_node_flags(mem_node, s, STARPU_MALLOC_PINNED);
		vector_recv = starpu_malloc_on_node_flags(mem_node, s, STARPU_MALLOC_PINNED);

		starpu_vector_data_register(&handle_send, mem_node, (uintptr_t) vector_send, s, 1);
		starpu_vector_data_register(&handle_recv, mem_node, (uintptr_t) vector_recv, s, 1);

		iterations = bench_nb_iterations(iterations, s);

		starpu_mpi_barrier(MPI_COMM_WORLD);

		for (j = 0; j < iterations; j++)
		{
			if (mpi_rank == 0)
			{
				t1 = starpu_timing_now();
				if (bidir)
				{
					ret = starpu_mpi_isend(handle_send, &send_req, 1, 0, MPI_COMM_WORLD);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
					ret = starpu_mpi_irecv(handle_recv, &recv_req, 1, 1, MPI_COMM_WORLD);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
					ret = starpu_mpi_wait(&send_req, MPI_STATUS_IGNORE);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
					ret = starpu_mpi_wait(&recv_req, MPI_STATUS_IGNORE);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
				}
				else
				{
					ret = starpu_mpi_send(handle_send, 1, 0, MPI_COMM_WORLD);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
					ret = starpu_mpi_recv(handle_recv, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
				}
				t2 = starpu_timing_now();

				const double t = (t2 - t1) / 2;

				lats[j] = t;
			}
			else
			{
				if (bidir)
				{
					ret = starpu_mpi_irecv(handle_recv, &recv_req, 0, 0, MPI_COMM_WORLD);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
					ret = starpu_mpi_isend(handle_send, &send_req, 0, 1, MPI_COMM_WORLD);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
					ret = starpu_mpi_wait(&recv_req, MPI_STATUS_IGNORE);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
					ret = starpu_mpi_wait(&send_req, MPI_STATUS_IGNORE);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
				}
				else
				{
					ret = starpu_mpi_recv(handle_recv, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
					ret = starpu_mpi_send(handle_send, 0, 1, MPI_COMM_WORLD);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
				}
			}

			starpu_mpi_barrier(MPI_COMM_WORLD);
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

			printf("%9lld\t%9.3lf\t%9.3f\t%9.3f\t%9.3lf\t%9.3lf\t%9.3lf\t%9.3lf\t%9.3lf\n",
				(long long)s, min_lat, bw_million_byte, bw_mbyte, d1_lat, med_lat, avg_lat, d9_lat, max_lat);
			fflush(stdout);
		}
		starpu_data_unregister(handle_recv);
		starpu_data_unregister(handle_send);

		starpu_free_on_node_flags(mem_node, (void*)vector_send, s, STARPU_MALLOC_PINNED);
		starpu_free_on_node_flags(mem_node, (void*)vector_recv, s, STARPU_MALLOC_PINNED);
	}
	global_tend = starpu_timing_now();

	if (mpi_rank == 0)
	{
		printf("Comm bench took %9.3lf ms\n", (global_tend - global_tstart) / 1000);
	}

	free(lats);
}
