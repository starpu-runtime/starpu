/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define VAL0 12
#define VAL1 24

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

void callback(void *arg)
{
	unsigned *received = arg;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	*received = *received + 1;
	FPRINTF_MPI(stderr, "Request %u received\n", *received);
	STARPU_PTHREAD_COND_SIGNAL(&cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

int do_test(int rank, int sdetached, int rdetached)
{
	int ret, i;
	int val[2];
	starpu_data_handle_t data[2];

        ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	if (rank == 1)
	{
		val[0] = VAL0;
		val[1] = VAL1;
	}
	else
	{
		val[0] = -1;
		val[1] = -1;
	}
	starpu_variable_data_register(&data[0], STARPU_MAIN_RAM, (uintptr_t)&val[0], sizeof(val[0]));
	starpu_variable_data_register(&data[1], STARPU_MAIN_RAM, (uintptr_t)&val[1], sizeof(val[1]));
	starpu_mpi_data_register(data[0], 77, 1);
	starpu_mpi_data_register(data[1], 88, 1);

	if (rank == 1)
	{
		for(i=1 ; i>=0 ; i--)
		{
			if (sdetached)
				starpu_mpi_isend_detached(data[i], 0, starpu_data_get_tag(data[i]), MPI_COMM_WORLD, NULL, NULL);
			else
				starpu_mpi_send(data[i], 0, starpu_data_get_tag(data[i]), MPI_COMM_WORLD);
		}
	}
	else if (rank == 0)
	{
		int received = 0;

		for(i=0 ; i<2 ; i++)
			FPRINTF_MPI(stderr, "Value[%d] = %d\n", i, val[i]);
		for(i=0 ; i<2 ; i++)
		{
			if (rdetached)
				starpu_mpi_irecv_detached(data[i], 1, starpu_data_get_tag(data[i]), MPI_COMM_WORLD, callback, &received);
			else
				starpu_mpi_recv(data[i], 1, starpu_data_get_tag(data[i]), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if (rdetached)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&mutex);
			while (received != 2)
			{
				FPRINTF_MPI(stderr, "Received %d messages\n", received);
				STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		}

		for(i=0 ; i<2 ; i++)
			starpu_data_acquire(data[i], STARPU_R);
		for(i=0 ; i<2 ; i++)
			FPRINTF_MPI(stderr, "Value[%d] = %d\n", i, val[i]);
		for(i=0 ; i<2 ; i++)
			starpu_data_release(data[i]);
	}
	FPRINTF_MPI(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	starpu_data_unregister(data[0]);
	starpu_data_unregister(data[1]);

	if (rank == 0)
	{
		ret = (val[0] == VAL0 && val[1] == VAL1) ? 0 : 1;
	}
	starpu_mpi_shutdown();
	return ret;
}

int main(int argc, char **argv)
{
	int size;
	int rank;
	int ret=0;
	int sdetached, rdetached;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
        starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
        starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

        if (size < 2)
        {
		FPRINTF_MPI(stderr, "We need at least 2 processes.\n");
                MPI_Finalize();
                return STARPU_TEST_SKIPPED;
        }

	for(sdetached=0 ; sdetached<=1 ; sdetached++)
	{
		for(rdetached=0 ; rdetached<=1 ; rdetached++)
		{
			ret += do_test(rank, sdetached, rdetached);
		}
	}

        MPI_Finalize();
	return ret;
}
