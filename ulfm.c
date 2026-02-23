/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2023  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <signal.h>

int starpu_mpi_ulfm_init(int* argc, char** argv[], MPI_Comm* comm, int* type);
int starpu_mpi_ulfm_comm_replace(MPI_Comm old_comm, MPI_Comm* new_comm_p, int* failure_number, int failed_ranks[], MPI_Comm* survivors_comm);

int main(int argc, char* argv[])
{
	int ret;
	int size, rank;
	struct starpu_conf conf;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
#if 0
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc!! = -1;
#endif

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	FPRINTF_MPI(stderr, "Init ok - my rnk %d - size %d\n", rank, size);

	MPI_Comm comm=MPI_COMM_WORLD;
	int type;
	starpu_mpi_ulfm_init(&argc, &argv, &comm, &type);

	if (0 && rank == size-1)
	{
		kill(getpid(), SIGKILL);
	}
	sleep(2);

	MPI_Comm ncomm, scomm;
	int failure_number, failed_ranks[size];
	starpu_mpi_ulfm_comm_replace(comm, &ncomm, &failure_number, failed_ranks, &scomm);

	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	FPRINTF_MPI(stderr, "After failure - my rnk %d - size %d\n", rank, size);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return EXIT_SUCCESS;
}
