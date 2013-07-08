/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#include <unistd.h>

//#define NB 1000
#define NB 10

int main(int argc, char **argv)
{
	int ret, rank, size, i;
	starpu_data_handle_t tab_handle[NB];
	int value[NB];

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need a even number of processes.\n");

		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	for(i=0 ; i<NB ; i++)
	{
		value[i]=i*rank;
		starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&value[i], sizeof(int));
		starpu_data_set_tag(tab_handle[i], i);
	}

	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	if (rank%2)
	{
		starpu_mpi_send(tab_handle[0], other_rank, 0, MPI_COMM_WORLD);
		starpu_mpi_send(tab_handle[NB-1], other_rank, NB-1, MPI_COMM_WORLD);
		for(i=1 ; i<NB-1 ; i++)
		{
			starpu_mpi_send(tab_handle[i], other_rank, i, MPI_COMM_WORLD);
		}
	}
	else
	{
		starpu_mpi_req req[NB];
		memset(req, 0, NB*sizeof(starpu_mpi_req));

		starpu_mpi_irecv(tab_handle[0], &req[0], other_rank, 0, MPI_COMM_WORLD);
		STARPU_ASSERT(req[0] != NULL);
		// We sleep to make sure that the data for the tag 9 will be received before the recv is posted
		usleep(2000000);
		for(i=1 ; i<NB ; i++)
		{
			starpu_mpi_irecv(tab_handle[i], &req[i], other_rank, i, MPI_COMM_WORLD);
			STARPU_ASSERT(req[i] != NULL);
		}
		for(i=0 ; i<NB ; i++)
		{
			starpu_mpi_wait(&req[i], NULL);
			int *rvalue = (int *)starpu_data_get_local_ptr(tab_handle[i]);
			STARPU_ASSERT_MSG(*rvalue==i*other_rank, "Incorrect received value: %d != %d\n", *rvalue, i*other_rank);
		}
	}

	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(tab_handle[i]);

	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return 0;
}
