/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This example illustrates the combination how to set up an alternative source
 * for a data handle. Here, a data A is owned by 0. All the odd nodes set up one
 * as an alternative source for A. Then, both even and odd nodes get A for reading
 * purposes.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu_mpi.h>
#include "helper.h"

int main(int argc, char *argv[])
{
	float A  = 1.1;
	static starpu_data_handle_t A_h;

	static int comm_rank; /* mpi rank of the process */
	static int comm_size; /* size of the mpi session */

	starpu_mpi_tag_t tag = 0;
	/* Initializes StarPU and the StarPU-MPI layer */
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conf");

	/* Get the process rank and session size */
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_rank == 0)
	{
		starpu_variable_data_register(&A_h, STARPU_MAIN_RAM,
					      (uintptr_t)&A, sizeof(A));
		starpu_data_acquire(A_h, STARPU_W);
		A = 1.1;
		starpu_data_release(A_h);
		FPRINTF(stderr, "[%d] I own A=%f (%p)\n", comm_rank,A,A_h);
	}
	else
	{
		starpu_variable_data_register(&A_h, -1,
					      (uintptr_t) NULL, sizeof(A));
		FPRINTF(stderr, "[%d] I do not own A (%p)\n", comm_rank,A_h);
	}
	starpu_mpi_data_register(A_h, tag++, 0);

	int rk;
	for (rk = 0; rk < comm_size; rk++)
	{
		if (comm_rank == rk)
		{
			FPRINTF(stderr, "[%d] I will receive A from %d\n", rk, rk % 2);
		}
		starpu_mpi_data_set_source(A_h, rk, rk % 2);
		starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, A_h, rk, NULL, NULL);
	}
	starpu_data_unregister_submit(A_h);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);

	starpu_mpi_shutdown();
	return 0;
}
