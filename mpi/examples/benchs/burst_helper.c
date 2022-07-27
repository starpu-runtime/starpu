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
#include <starpu_mpi.h>

#include "helper.h"
#include "burst_helper.h"

#if defined(STARPU_SIMGRID) || defined(STARPU_QUICK_CHECK)
#define NB_REQUESTS 10
#else
#define NB_REQUESTS 50
#endif
#define NX_ARRAY (320 * 320)

static starpu_data_handle_t* recv_handles;
static starpu_data_handle_t* send_handles;
static float** recv_buffers;
static float** send_buffers;
static starpu_mpi_req* recv_reqs;
static starpu_mpi_req* send_reqs;

int burst_nb_requests = NB_REQUESTS;

void burst_init_data(int rank)
{
	unsigned nx = NX_ARRAY;
	if (RUNNING_ON_VALGRIND)
		nx = 4*4;

	if (rank == 0 || rank == 1)
	{
		recv_handles = malloc(burst_nb_requests * sizeof(starpu_data_handle_t));
		send_handles = malloc(burst_nb_requests * sizeof(starpu_data_handle_t));
		recv_buffers = malloc(burst_nb_requests * sizeof(float*));
		send_buffers = malloc(burst_nb_requests * sizeof(float*));
		recv_reqs = malloc(burst_nb_requests * sizeof(starpu_mpi_req));
		send_reqs = malloc(burst_nb_requests * sizeof(starpu_mpi_req));

		int i = 0;
		for (i = 0; i < burst_nb_requests; i++)
		{
			send_buffers[i] = malloc(nx * sizeof(float));
			memset(send_buffers[i], 0, nx * sizeof(float));
			starpu_vector_data_register(&send_handles[i], STARPU_MAIN_RAM, (uintptr_t) send_buffers[i], nx, sizeof(float));

			recv_buffers[i] = malloc(nx * sizeof(float));
			memset(recv_buffers[i], 0, nx * sizeof(float));
			starpu_vector_data_register(&recv_handles[i], STARPU_MAIN_RAM, (uintptr_t) recv_buffers[i], nx, sizeof(float));
		}
	}
}

void burst_free_data(int rank)
{
	if (rank == 0 || rank == 1)
	{
		int i = 0;
		for (i = 0; i < burst_nb_requests; i++)
		{
			starpu_data_unregister(send_handles[i]);
			free(send_buffers[i]);

			starpu_data_unregister(recv_handles[i]);
			free(recv_buffers[i]);
		}

		free(recv_handles);
		free(send_handles);
		free(recv_buffers);
		free(send_buffers);
		free(recv_reqs);
		free(send_reqs);
	}
}

/* Burst simultaneous from both nodes: 0 and 1 post all the recvs, synchronise, and then post all the sends */
void burst_bidir(int rank)
{
	int other_rank = (rank == 0) ? 1 : 0;
	int i, ret;

	FPRINTF(stderr, "Simultaneous....start (rank %d)\n", rank);

	if (rank == 0 || rank == 1)
	{
		for (i = 0; i < burst_nb_requests; i++)
		{
			recv_reqs[i] = NULL;
			ret = starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], other_rank, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	if (rank == 0 || rank == 1)
	{
		for (i = 0; i < burst_nb_requests; i++)
		{
			send_reqs[i] = NULL;
			ret = starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_prio");
		}

		for (i = 0; i < burst_nb_requests; i++)
		{
			if (recv_reqs[i]) ret = starpu_mpi_wait(&recv_reqs[i], MPI_STATUS_IGNORE);
			if (send_reqs[i]) ret = starpu_mpi_wait(&send_reqs[i], MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		}
	}

	FPRINTF(stderr, "Simultaneous....end (rank %d)\n", rank);
	starpu_mpi_barrier(MPI_COMM_WORLD);
}

void burst_unidir(int sender, int receiver, int rank)
{
	FPRINTF(stderr, "%d -> %d... start (rank %d)\n", sender, receiver, rank);
	int i, ret;

	if (rank == receiver)
	{
		for (i = 0; i < burst_nb_requests; i++)
		{
			recv_reqs[i] = NULL;
			ret = starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], sender, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	if (rank == sender)
	{
		for (i = 0; i < burst_nb_requests; i++)
		{
			send_reqs[i] = NULL;
			ret = starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], receiver, i, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_prio");
		}
	}

	if (rank == sender || rank == receiver)
	{
		for (i = 0; i < burst_nb_requests; i++)
		{
			if (rank != sender && recv_reqs[i]) ret = starpu_mpi_wait(&recv_reqs[i], MPI_STATUS_IGNORE);
			if (rank == sender && send_reqs[i]) ret = starpu_mpi_wait(&send_reqs[i], MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		}
	}

	FPRINTF(stderr, "%d -> %d... end (rank %d)\n", sender, receiver, rank);

	starpu_mpi_barrier(MPI_COMM_WORLD);
}

/* Half burst from both nodes, second half burst is triggered after some requests finished. */
void burst_bidir_half_postponed(int rank)
{
	int other_rank = (rank == 0) ? 1 : 0;
	int i, ret;

	FPRINTF(stderr, "Half/half burst...start (rank %d)\n", rank);

	if (rank == 0 || rank == 1)
	{
		for (i = 0; i < burst_nb_requests; i++)
		{
			recv_reqs[i] = NULL;
			ret = starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], other_rank, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	if (rank == 0 || rank == 1)
	{
		for (i = 0; i < (burst_nb_requests / 2); i++)
		{
			send_reqs[i] = NULL;
			ret = starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_prio");
		}

		if (recv_reqs[burst_nb_requests / 4])
		{
			ret = starpu_mpi_wait(&recv_reqs[burst_nb_requests / 4], MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		}

		for (i = (burst_nb_requests / 2); i < burst_nb_requests; i++)
		{
			send_reqs[i] = NULL;
			ret = starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_prio");
		}

		for (i = 0; i < burst_nb_requests; i++)
		{
			if (recv_reqs[i]) ret = starpu_mpi_wait(&recv_reqs[i], MPI_STATUS_IGNORE);
			if (send_reqs[i]) ret = starpu_mpi_wait(&send_reqs[i], MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		}
	}

	FPRINTF(stderr, "Half/half burst...done (rank %d)\n", rank);
	starpu_mpi_barrier(MPI_COMM_WORLD);
}

void burst_all(int rank)
{
	double start, end;
	start = starpu_timing_now();

	/* Burst simultaneous from both nodes: 0 and 1 post all the recvs, synchronise, and then post all the sends */
	burst_bidir(rank);

	/* Burst from 0 to 1 : rank 1 posts all the recvs, barrier, then rank 0 posts all the sends */
	burst_unidir(0, 1, rank);

	/* Burst from 1 to 0 : rank 0 posts all the recvs, barrier, then rank 1 posts all the sends */
	burst_unidir(1, 0, rank);

	/* Half burst from both nodes, second half burst is triggered after some requests finished. */
	burst_bidir_half_postponed(rank);

	end = starpu_timing_now();
	FPRINTF(stderr, "All bursts took %.0f ms\n", (end - start) / 1000.0);
}
