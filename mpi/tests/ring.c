/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_mpi.h>

#define NITER	2048

unsigned token = 42;
starpu_data_handle token_handle;

int main(int argc, char **argv)
{
	MPI_Init(NULL, NULL);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			fprintf(stderr, "We need at least 2 processes.\n");

		MPI_Finalize();
		return 0;
	}

	starpu_init(NULL);
	starpu_mpi_initialize();

	starpu_register_vector_data(&token_handle, 0, (uintptr_t)&token, 1, sizeof(unsigned));

	unsigned nloops = NITER;
	unsigned loop;

	unsigned last_loop = nloops - 1;
	unsigned last_rank = size - 1;

	for (loop = 0; loop < nloops; loop++)
	{
		int tag = loop*size + rank;

		if (!((loop == 0) && (rank == 0)))
		{
			token = 0;
			starpu_mpi_recv(token_handle, (rank+size-1)%size, tag, MPI_COMM_WORLD);
		}
		else {
			token = 0;
			fprintf(stdout, "Start with token value %d\n", token);
		}

		token += 1;
		
		if (!((loop == last_loop) && (rank == last_rank)))
		{
			starpu_mpi_send(token_handle, (rank+1)%size, tag+1, MPI_COMM_WORLD);
		}
		else {
			fprintf(stdout, "Finished : token value %d\n", token);
		}
	}
	
	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	if (rank == last_rank)
	{
		STARPU_ASSERT(token == nloops*size);
	}

	return 0;
}
