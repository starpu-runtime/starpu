/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi_lb.h>
#include "helper.h"

#if !defined(STARPU_HAVE_UNSETENV) || !defined(STARPU_USE_MPI_MPI)

#warning unsetenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

void get_neighbors(int **neighbor_ids, int *nneighbors)
{
	int rank, size;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
	*nneighbors = 1;
	*neighbor_ids = malloc(sizeof(int));
	*neighbor_ids[0] = rank==size-1?0:rank+1;
}

void get_data_unit_to_migrate(starpu_data_handle_t **handle_unit, int *nhandles, int dst_node)
{
	(void)handle_unit;
	(void)dst_node;
	*nhandles = 0;
}

int main(int argc, char **argv)
{
	int ret;
	struct starpu_mpi_lb_conf itf;
	int mpi_init;

	itf.get_neighbors = get_neighbors;
	itf.get_data_unit_to_migrate = get_data_unit_to_migrate;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	unsetenv("STARPU_MPI_LB");
	starpu_mpi_lb_init(NULL, NULL);
	starpu_mpi_lb_shutdown();

	starpu_mpi_lb_init("heat", &itf);
	starpu_mpi_lb_shutdown();

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}

#endif
