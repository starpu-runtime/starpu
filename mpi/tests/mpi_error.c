/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2014-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2016  CNRS
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

#define SIZE 100

int main(int argc, char **argv)
{
	int ret, rank, size;
	float *tab;
	starpu_data_handle_t tab_handle;

	MPI_Init(&argc, &argv);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	if (size<2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		starpu_shutdown();
		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
	{
		tab = calloc(SIZE, sizeof(float));
		starpu_vector_data_register(&tab_handle, STARPU_MAIN_RAM, (uintptr_t)tab, SIZE, sizeof(float));
		starpu_mpi_send(tab_handle, 1, 12, MPI_COMM_WORLD);
		starpu_data_unregister(tab_handle);
		free(tab);
	}
	else if (rank == 1)
	{
		MPI_Status status;

		tab = calloc(SIZE-2, sizeof(float));
		starpu_vector_data_register(&tab_handle, STARPU_MAIN_RAM, (uintptr_t)tab, SIZE-2, sizeof(float));
		starpu_mpi_recv(tab_handle, 0, 12, MPI_COMM_WORLD, &status);
		starpu_data_unregister(tab_handle);
		free(tab);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return 0;
}
