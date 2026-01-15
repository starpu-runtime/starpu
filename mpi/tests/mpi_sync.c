/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi_private.h>
#include "helper.h"

// This test checks that when starpu_mpi_issend functions are used,
// then data should not be sent too early but only when a matching
// receive has been posted

#define NB 6

typedef int (*send_func)(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);

int exchange_variable(int rank, starpu_mpi_tag_t initial_tag, send_func func)
{
	int ret=0, i;
	starpu_data_handle_t tab_handle[NB];
	int value[NB];

	for(i=0 ; i<NB ; i++)
	{
		if (rank%2)
			value[i]=(i+1)*(rank+1) + (initial_tag);
		else
			value[i] = 42;
		starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&value[i], sizeof(int));
		starpu_mpi_data_register(tab_handle[i], i, rank);
	}

	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	if (rank%2)
	{
		for(i=NB-1 ; i>=0 ; i--)
		{
			ret = func(tab_handle[i], other_rank, initial_tag+i, MPI_COMM_WORLD, NULL, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_issend");
		}
		for(i=0 ; i<NB ; i++)
			starpu_data_unregister(tab_handle[i]);
	}
	else
	{
		for(i=0 ; i<NB ; i++)
		{
			ret = starpu_mpi_recv(tab_handle[i], other_rank, initial_tag+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
			starpu_data_unregister(tab_handle[i]);
			if (value[i] != (i+1)*(other_rank+1) + (initial_tag))
			{
				FPRINTF_MPI(stderr, "Incorrect received value: %d != %d\n", value[i], (i+1)*(other_rank+1)+(int)(initial_tag));
				ret = 1;
			}
			else
			{
				FPRINTF_MPI(stderr, "Correct received value: %d == %d\n", value[i], (i+1)*(other_rank+1)+(int)(initial_tag));
			}
		}
	}

	return ret;
}

#ifdef STARPU_USE_MPI_NMAD
int main(int argc, char **argv)
{
	return 0;
}
#else
int main(int argc, char **argv)
{
	int ret=0;
	int rank=0, size;
	int mpi_init;
	struct starpu_conf conf;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) { ret = STARPU_TEST_SKIPPED; goto enodev; }
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		FPRINTF(stderr, "We need a even number of processes.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return 0;
	}

	int nb;
	ret = exchange_variable(rank, 100, starpu_mpi_isend_detached);
	if (rank%2 == 0)
	{
		nb = _starpu_mpi_get_early_data_nb();
		FPRINTF(stderr, "[%d] Number of received early data %d\n", rank, nb);
		if (nb == 0)
		{
			FPRINTF(stderr, "Error. The number of received early data should be more than 0");
			ret ++;
		}
	}
	ret += exchange_variable(rank, 1000, starpu_mpi_issend_detached);
	if (rank%2 == 0)
	{
		int nb2 = _starpu_mpi_get_early_data_nb();
		FPRINTF(stderr, "[%d] Number of received early data %d\n", rank, nb2-nb);
		if (nb2 != nb)
		{
			FPRINTF(stderr, "Error. The number of received early data should be 0 and not %d\n", (nb2-nb));
			ret ++;
		}
	}

	starpu_mpi_shutdown();

enodev:
	if (!mpi_init)
		MPI_Finalize();

	return rank == 0 ? ret : 0;
}
#endif
