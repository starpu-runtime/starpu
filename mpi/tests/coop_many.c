/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Every rank asynchronously sends coop and receives from coop several times */

#include <starpu_mpi.h>
#include "helper.h"

#define NX (256*256)
#define NB_MCASTS 10

int main(int argc, char **argv)
{
	int ret, rank, worldsize;
	int mpi_init;
	int i = 0, j = 0;
	MPI_Status status;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	float **vectors = malloc(NB_MCASTS*worldsize*sizeof(float*));
	starpu_data_handle_t *handles = malloc(NB_MCASTS*worldsize*sizeof(starpu_data_handle_t));
	starpu_mpi_req *reqs = malloc(NB_MCASTS*worldsize*sizeof(starpu_mpi_req));
	for (i = 0; i < NB_MCASTS*worldsize; i++)
	{
		vectors[i] = malloc(NX*sizeof(float));
		for (j = 0; j < NX; j++)
		{
			vectors[i][j] = i;
		}
		starpu_vector_data_register(&handles[i], STARPU_MAIN_RAM, (uintptr_t) vectors[i], NX, sizeof(float));
	}

	int sender_rank = 0;
	// Submit all communications:
	for (sender_rank = 0; sender_rank < worldsize; sender_rank++)
	{
		for (i = 0; i < NB_MCASTS; i++)
		{
			int tag = sender_rank*NB_MCASTS+i;
			assert(tag < worldsize*NB_MCASTS);

			if (rank == sender_rank)
			{
				starpu_mpi_coop_sends_data_handle_nb_sends(handles[tag], worldsize-1);
				for (j = 0; j < worldsize; j++)
				{
					if (j != sender_rank)
					{
						ret = starpu_mpi_isend_detached(handles[tag], j, tag, MPI_COMM_WORLD, NULL, NULL);
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
					}
				}
			}
			else
			{
				ret = starpu_mpi_irecv(handles[tag], &reqs[tag], sender_rank, tag, MPI_COMM_WORLD);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
			}
		}
	}

	// Wait for all receives:
	for (sender_rank = 0; sender_rank < worldsize; sender_rank++)
	{
		for (i = 0; i < NB_MCASTS; i++)
		{
			int tag = sender_rank*NB_MCASTS+i;
			assert(tag < worldsize*NB_MCASTS);

			if (rank != sender_rank)
			{
				ret = starpu_mpi_wait(&reqs[tag], MPI_STATUS_IGNORE);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");

				starpu_data_acquire(handles[i], STARPU_R);
				STARPU_ASSERT_MSG(vectors[i][0] == i, "vectors[%d][0] = %f, expected %d\n", i, vectors[i][0], i);
				STARPU_ASSERT_MSG(vectors[i][NX-1] == i, "vector[%d][%d] = %f, expected %d\n", i, NX-1, vectors[i][NX-1], i);
				starpu_data_release(handles[i]);
			}
		}
	}

	// This barrier is unblocked after all receives are done, that means all isends are also done, so we can after that unregister handles (there is no implicit wait on the isends)
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	for (i = 0; i < NB_MCASTS*worldsize; i++)
	{
		starpu_data_unregister(handles[i]);
		free(vectors[i]);
	}
	free(vectors);
	free(handles);
	free(reqs);

	printf("[%d] end\n", rank);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
