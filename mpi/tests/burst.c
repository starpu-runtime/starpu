/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This test sends simultaneously many communications, with various configurations.
 *
 * Global purpose is to watch the behaviour with traces.
 */

#include <starpu_mpi.h>
#include "helper.h"

#ifdef STARPU_SIMGRID
#define NB_REQUESTS 10
#else
#define NB_REQUESTS 500
#endif
#define NX_ARRAY (320 * 320)

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

void recv_callback(void* arg)
{
	int* received = arg;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	*received = 1;
	STARPU_PTHREAD_COND_SIGNAL(&cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

int main(int argc, char **argv)
{
	int ret, rank, size, mpi_init, other_rank;
	starpu_data_handle_t recv_handles[NB_REQUESTS];
	starpu_data_handle_t send_handles[NB_REQUESTS];
	float* recv_buffers[NB_REQUESTS];
	float* send_buffers[NB_REQUESTS];
	starpu_mpi_req recv_reqs[NB_REQUESTS];
	starpu_mpi_req send_reqs[NB_REQUESTS];
	MPI_Status status;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (rank > 1)
	{
		starpu_mpi_barrier(MPI_COMM_WORLD);
		starpu_mpi_wait_for_all(MPI_COMM_WORLD);

		starpu_mpi_barrier(MPI_COMM_WORLD);
		starpu_mpi_wait_for_all(MPI_COMM_WORLD);

		starpu_mpi_barrier(MPI_COMM_WORLD);
		starpu_mpi_wait_for_all(MPI_COMM_WORLD);

		starpu_mpi_barrier(MPI_COMM_WORLD);
		starpu_mpi_wait_for_all(MPI_COMM_WORLD);

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();

		return 0;
	}

	other_rank = (rank == 0) ? 1 : 0;


	/* Burst simultaneous from both nodes */
	if (rank == 0)
	{
		printf("Simultaneous....\n");
	}

	for (int i = 0; i < NB_REQUESTS; i++)
	{
		send_buffers[i] = malloc(NX_ARRAY * sizeof(float));
		memset(send_buffers[i], 0, NX_ARRAY * sizeof(float));
		starpu_vector_data_register(&send_handles[i], STARPU_MAIN_RAM, (uintptr_t) send_buffers[i], NX_ARRAY, sizeof(float));

		recv_buffers[i] = malloc(NX_ARRAY * sizeof(float));
		memset(recv_buffers[i], 0, NX_ARRAY * sizeof(float));
		starpu_vector_data_register(&recv_handles[i], STARPU_MAIN_RAM, (uintptr_t) recv_buffers[i], NX_ARRAY, sizeof(float));

		starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], other_rank, i, MPI_COMM_WORLD);
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	for (int i = 0; i < NB_REQUESTS; i++)
	{
		starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);


	/* Burst from 0 to 1 */
	if (rank == 0)
	{
		printf("Done.\n");
		printf("0 -> 1...\n");
	}
	else
	{
		for (int i = 0; i < NB_REQUESTS; i++)
		{
			starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], other_rank, i, MPI_COMM_WORLD);
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	if (rank == 0)
	{
		for (int i = 0; i < NB_REQUESTS; i++)
		{
			starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
		}
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);


	/* Burst from 1 to 0 */
	if (rank == 0)
	{
		printf("Done.\n");
		printf("1 -> 0...\n");

		for (int i = 0; i < NB_REQUESTS; i++)
		{
			starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], other_rank, i, MPI_COMM_WORLD);
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	if (rank == 1)
	{
		for (int i = 0; i < NB_REQUESTS; i++)
		{
			starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
		}
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);


	/* Half burst from both nodes, second half burst is triggered after some requests finished. */
	if (rank == 0)
	{
		printf("Done.\n");
		printf("Half/half burst...\n");
	}

	int received = 0;

	for (int i = 0; i < NB_REQUESTS; i++)
	{
		if (i == (NB_REQUESTS / 4))
		{
			starpu_mpi_irecv_detached(recv_handles[i], other_rank, i, MPI_COMM_WORLD, recv_callback, &received);
		}
		else
		{
			starpu_mpi_irecv(recv_handles[i], &recv_reqs[i], other_rank, i, MPI_COMM_WORLD);
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	for (int i = 0; i < (NB_REQUESTS / 2); i++)
	{
		starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	while (!received)
		STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	for (int i = (NB_REQUESTS / 2); i < NB_REQUESTS; i++)
	{
		starpu_mpi_isend_prio(send_handles[i], &send_reqs[i], other_rank, i, i, MPI_COMM_WORLD);
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("Done.\n");
	}


	for (int i = 0; i < NB_REQUESTS; i++)
	{
		starpu_data_unregister(send_handles[i]);
		free(send_buffers[i]);

		starpu_data_unregister(recv_handles[i]);
		free(recv_buffers[i]);
	}

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
