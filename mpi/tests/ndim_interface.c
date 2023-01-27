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

#include <starpu_mpi.h>
#include <stdlib.h>
#include "helper.h"

#ifdef STARPU_QUICK_CHECK
#  define NITER	16
#else
#  define NITER	2048
#endif

#define BIGSIZE	32
#define SIZE	16

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	/* Node 0 will allocate a big 4-dim array and only register an inner part of
	 * it as the 4-dim array, Node 1 will allocate a 4-dim array of small size and
	 * register it directly. Node 0 and 1 will then exchange the content of
	 * their arrays. */

	int *arr4d = NULL;
	starpu_data_handle_t arr4d_handle = NULL;

	if (rank == 0)
	{
		arr4d = calloc(BIGSIZE*BIGSIZE*BIGSIZE*BIGSIZE, sizeof(int));
		assert(arr4d);

		/* fill the inner 4-dim array */
		unsigned i, j, k, l;
		int n = 0;
		for (l = 0; l < SIZE; l++)
		{
			for (k = 0; k < SIZE; k++)
			{
				for (j = 0; j < SIZE; j++)
				{
					for (i = 0; i < SIZE; i++)
					{
						arr4d[i + j*BIGSIZE + k*BIGSIZE*BIGSIZE + l*BIGSIZE*BIGSIZE*BIGSIZE] = n++;
					}
				}
			}
		}

		unsigned nn[4] = {SIZE, SIZE, SIZE, SIZE};
		unsigned ldn[4] = {1, BIGSIZE, BIGSIZE*BIGSIZE, BIGSIZE*BIGSIZE*BIGSIZE};

		starpu_ndim_data_register(&arr4d_handle, STARPU_MAIN_RAM, (uintptr_t)arr4d, ldn, nn, 4, sizeof(int));
	}
	else if (rank == 1)
	{
		arr4d = calloc(SIZE*SIZE*SIZE*SIZE, sizeof(int));
		assert(arr4d);

		unsigned nn[4] = {SIZE, SIZE, SIZE, SIZE};
		unsigned ldn[4] = {1, SIZE, SIZE*SIZE, SIZE*SIZE*SIZE};

		starpu_ndim_data_register(&arr4d_handle, STARPU_MAIN_RAM, (uintptr_t)arr4d, ldn, nn, 4, sizeof(int));
	}

	if (rank == 0)
	{
		ret = starpu_mpi_send(arr4d_handle, 1, 0x42, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");

		MPI_Status status;
		ret = starpu_mpi_recv(arr4d_handle, 1, 0x1337, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");

		/* check the content of the 4-dim array */
		ret = starpu_data_acquire(arr4d_handle, STARPU_R);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

		int m = 10;
		unsigned i, j, k, l;
		for (l = 0; l < SIZE; l++)
		{
			for (k = 0; k < SIZE; k++)
			{
				for (j = 0; j < SIZE; j++)
				{
					for (i = 0; i < SIZE; i++)
					{
						assert(arr4d[i + j*BIGSIZE + k*BIGSIZE*BIGSIZE + l*BIGSIZE*BIGSIZE*BIGSIZE] == m);
						m++;
					}
				}
			}
		}

		starpu_data_release(arr4d_handle);
	}
	else if (rank == 1)
	{
		MPI_Status status;
		ret = starpu_mpi_recv(arr4d_handle, 0, 0x42, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");

		/* check the content of the 4-dim array and modify it */
		ret = starpu_data_acquire(arr4d_handle, STARPU_RW);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

		int n = 0, m = 10;
		unsigned i, j, k, l;
		for (l = 0; l < SIZE; l++)
		{
			for (k = 0; k < SIZE; k++)
			{
				for (j = 0; j < SIZE; j++)
				{
					for (i = 0; i < SIZE; i++)
					{
						assert(arr4d[i + j*SIZE + k*SIZE*SIZE + l*SIZE*SIZE*SIZE] == n);
						n++;
						arr4d[i + j*SIZE + k*SIZE*SIZE + l*SIZE*SIZE*SIZE] = m++;
					}
				}
			}
		}

		starpu_data_release(arr4d_handle);

		ret = starpu_mpi_send(arr4d_handle, 0, 0x1337, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}

	FPRINTF(stdout, "Rank %d is done\n", rank);
	fflush(stdout);

	if (rank == 0 || rank == 1)
	{
		starpu_data_unregister(arr4d_handle);
		free(arr4d);
	}
	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
