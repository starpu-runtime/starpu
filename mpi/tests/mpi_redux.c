/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

void callback(void *arg)
{
	unsigned *received = arg;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	*received = *received + 1;
	FPRINTF_MPI(stderr, "received = %u\n", *received);
	STARPU_PTHREAD_COND_SIGNAL(&cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

int main(int argc, char **argv)
{
	int ret, rank, size, sum;
	int value=0;
	starpu_data_handle_t *handles;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	sum = ((size-1) * (size) / 2);

	if (rank == 0)
	{
		int src;
		int received = 1;

		handles = malloc(size * sizeof(starpu_data_handle_t));

		for(src=1 ; src<size ; src++)
		{
			starpu_variable_data_register(&handles[src], -1, (uintptr_t)NULL, sizeof(int));
			starpu_mpi_irecv_detached(handles[src], src, 12+src, MPI_COMM_WORLD, callback, &received);
		}

		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		while (received != size)
			STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

		for(src=1 ; src<size ; src++)
		{
			void *ptr = starpu_data_get_local_ptr(handles[src]);
			value += *((int *)ptr);
			starpu_data_unregister(handles[src]);
		}

		for(src=1 ; src<size ; src++)
		{
			starpu_variable_data_register(&handles[src], STARPU_MAIN_RAM, (uintptr_t)&sum, sizeof(int));
			starpu_mpi_send(handles[src], src, 12+src, MPI_COMM_WORLD);
			starpu_data_unregister(handles[src]);
		}
	}
	else
	{
		value = rank;
		handles = malloc(sizeof(starpu_data_handle_t));
		starpu_variable_data_register(&handles[0], STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(int));
		starpu_mpi_send(handles[0], 0, 12+rank, MPI_COMM_WORLD);
		starpu_data_unregister_submit(handles[0]);

		starpu_variable_data_register(&handles[0], STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(int));
		starpu_mpi_recv(handles[0], 0, 12+rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		starpu_data_unregister(handles[0]);
	}

	starpu_task_wait_for_all();
	free(handles);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	STARPU_ASSERT_MSG(sum == value, "Sum of first %d integers is %d, not %d\n", size-1, sum, value);

	return 0;
}
