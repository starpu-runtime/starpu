/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int nb_nodes)
{
	return x % nb_nodes;
}

void cpu_codelet(void *descr[], void *_args)
{
	int *vector = (int *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
	unsigned i;
	int rank;

	starpu_codelet_unpack_args(_args, &rank);
	for (i = 0; i < nx; i++)
	{
		//fprintf(stderr,"rank %d v[%d] = %d\n", rank, i, vector[i]);
		vector[i] *= rank+2;
	}
}

static struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_codelet},
	.nbuffers = 1,
	.modes = {STARPU_RW},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
};

void scallback(void *arg)
{
	char *msg = arg;
	FPRINTF_MPI(stderr, "Sending completed for <%s>\n", msg);
}

void rcallback(void *arg)
{
	char *msg = arg;
	FPRINTF_MPI(stderr, "Reception completed for <%s>\n", msg);
}

int main(int argc, char **argv)
{
	int rank, nodes, ret, x;
	int *vector = NULL;
	starpu_data_handle_t *data_handles;
	int size=10;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);

	if (starpu_cpu_worker_get_count() == 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
	{
		/* Allocate the vector */
		vector = malloc(size * sizeof(int));
		for(x=0 ; x<size ; x++)
			vector[x] = x+10;

		// Print vector
		FPRINTF_MPI(stderr, " Input vector: ");
		for(x=0 ; x<size ; x++)
		{
			FPRINTF(stderr, "%d\t", vector[x]);
		}
		FPRINTF(stderr,"\n");
	}

	/* Allocate data handles and register data to StarPU */
	data_handles = (starpu_data_handle_t *) calloc(size, sizeof(starpu_data_handle_t));
	for(x = 0; x < size ; x++)
	{
		int mpi_rank = my_distrib(x, nodes);
		if (rank == 0)
		{
			starpu_vector_data_register(&data_handles[x], 0, (uintptr_t)&vector[x], 1, sizeof(int));
		}
		else if (mpi_rank == rank)
		{
			/* I do not own this index but i will need it for my computations */
			starpu_vector_data_register(&data_handles[x], -1, (uintptr_t)NULL, 1, sizeof(int));
		}
		else
		{
			/* I know it's useless to allocate anything for this */
			data_handles[x] = NULL;
		}
		if (data_handles[x])
		{
			starpu_mpi_data_register(data_handles[x], x, 0);
		}
	}

	/* Scatter the matrix among the nodes */
	for(x = 0; x < size ; x++)
	{
		if (data_handles[x])
		{
			int mpi_rank = my_distrib(x, nodes);
			starpu_mpi_data_set_rank(data_handles[x], mpi_rank);
		}
	}
	starpu_mpi_scatter_detached(data_handles, size, 0, MPI_COMM_WORLD, scallback, "scatter", NULL, NULL);

	/* Calculation */
	for(x = 0; x < size ; x++)
	{
		if (data_handles[x])
		{
			int owner = starpu_mpi_data_get_rank(data_handles[x]);
			if (owner == rank)
			{
				FPRINTF_MPI(stderr,"Computing on data[%d]\n", x);
				starpu_task_insert(&cl,
						   STARPU_VALUE, &rank, sizeof(rank),
						   STARPU_RW, data_handles[x],
						   0);
			}
		}
	}

	/* Gather the matrix on main node */
	starpu_mpi_gather_detached(data_handles, size, 0, MPI_COMM_WORLD, scallback, "gather", rcallback, "gather");
	for(x = 0; x < size ; x++)
	{
		if (data_handles[x])
		{
			starpu_mpi_data_set_rank(data_handles[x], 0);
		}
	}

	/* Unregister matrix from StarPU */
	for(x=0 ; x<size ; x++)
	{
		if (data_handles[x])
		{
			starpu_data_unregister(data_handles[x]);
		}
	}

	// Print vector
	if (rank == 0)
	{
		FPRINTF_MPI(stderr, "Output vector: ");
		for(x=0 ; x<size ; x++)
		{
			FPRINTF(stderr, "%d\t", vector[x]);
		}
		FPRINTF(stderr,"\n");
		for(x=0 ; x<size ; x++)
		{
			int mpi_rank = my_distrib(x, nodes);
			if (vector[x] != (x+10) * (mpi_rank+2))
			{
				FPRINTF_MPI(stderr, "Incorrect value for vector[%d]. computed %d != expected %d\n", x, vector[x], (x+10) * (mpi_rank+2));
				ret = 1;
			}
		}
		free(vector);
	}

	// Free memory
	free(data_handles);

	starpu_mpi_shutdown();
	return (rank == 0) ? ret : 0;
}
