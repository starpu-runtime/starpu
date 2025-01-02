/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(int argc, char **argv)
{
	int ret, rank, size;
	unsigned send[2] = {42, 11};
	unsigned recv[2] = {33, 33};
	starpu_mpi_req req[2];
	starpu_data_handle_t send_handle[2];
	starpu_data_handle_t recv_handle[2];
	struct starpu_conf conf;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
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

	starpu_variable_data_register(&send_handle[0], STARPU_MAIN_RAM, (uintptr_t)&send[0], sizeof(unsigned));
	starpu_variable_data_register(&send_handle[1], STARPU_MAIN_RAM, (uintptr_t)&send[1], sizeof(unsigned));
	starpu_variable_data_register(&recv_handle[0], STARPU_MAIN_RAM, (uintptr_t)&recv[0], sizeof(unsigned));
	starpu_variable_data_register(&recv_handle[1], STARPU_MAIN_RAM, (uintptr_t)&recv[1], sizeof(unsigned));

	if (rank == 0)
	{
		ret = starpu_mpi_isend(send_handle[0], &(req[0]), 1, 12, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
		ret = starpu_mpi_isend(send_handle[1], &(req[1]), 1, 13, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
	}
	else if (rank == 1)
	{
		ret = starpu_mpi_irecv(recv_handle[0], &(req[0]), 0, 12, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
		ret = starpu_mpi_irecv(recv_handle[1], &(req[1]), 0, 13, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
	}

	if (rank == 0 || rank == 1)
	{
		int nb_req=2;
		while (nb_req)
		{
			int r=0;
			for(r=0 ; r<2 ; r++)
			{
				if (req[r])
				{
					int finished = 0;
					MPI_Status status;
					ret = starpu_mpi_test(&req[r], &finished, &status);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_test");
					STARPU_ASSERT(finished != -1);
					if (finished)
					{
						FPRINTF(stderr, "[%d] Request %d finished\n", rank, r);
						req[r] = NULL;
						nb_req--;
					}
				}
			}
#ifdef STARPU_SIMGRID
			starpu_sleep(0.001);
#endif
		}
	}
	FPRINTF(stderr, "[%d] All requests finished\n", rank);

	starpu_data_unregister(send_handle[0]);
	starpu_data_unregister(send_handle[1]);
	starpu_data_unregister(recv_handle[0]);
	starpu_data_unregister(recv_handle[1]);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
