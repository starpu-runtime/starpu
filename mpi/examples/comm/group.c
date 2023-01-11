/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

int main(int argc, char **argv)
{
#ifdef STARPU_HAVE_MPI_COMM_CREATE_GROUP
	int thread_support;

	if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_support) != MPI_SUCCESS)
	{
		fprintf(stderr,"MPI_Init_thread failed\n");
		exit(1);
	}
	if (thread_support == MPI_THREAD_FUNNELED)
		fprintf(stderr,"Warning: MPI only has funneled thread support, not serialized, hoping this will work\n");
	if (thread_support < MPI_THREAD_FUNNELED)
		fprintf(stderr,"Warning: MPI does not have thread support!\n");

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size < 4)
	{
		FPRINTF(stderr, "We need at least 4 processes.\n");
		MPI_Finalize();
		return (world_rank==0) ? STARPU_TEST_SKIPPED : 0;
	}

	// create a new communicator with the even ranks processes
	int ranks[world_size/2];
	int pos,n;
	for(pos=0,n=0 ; pos<world_size ; pos+=2,n+=1)
	{
		ranks[pos/2] = pos;
	}

	MPI_Group world_group, even_group;
	MPI_Comm even_comm;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	MPI_Group_incl(world_group, n, ranks, &even_group);
	MPI_Comm_create_group(MPI_COMM_WORLD, even_group, 0, &even_comm);

	int even_rank=-1, even_size=-1;
	if (even_comm != MPI_COMM_NULL)
	{
		MPI_Comm_rank(even_comm, &even_rank);
		MPI_Comm_size(even_comm, &even_size);
	}

	FPRINTF(stderr, "WORLD RANK/SIZE: %d/%d \t EVEN RANK/SIZE: %d/%d\n", world_rank, world_size, even_rank, even_size);

	int ret;
	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	if (even_rank == 0)
	{
		starpu_data_handle_t handle;
		int data=42;
		starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&data, sizeof(int));
		ret = starpu_mpi_send(handle, 1, 0, even_comm);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		starpu_data_unregister(handle);
	}
	else if (even_rank == 1)
	{
		starpu_data_handle_t handle;
		starpu_variable_data_register(&handle, -1, (uintptr_t)NULL, sizeof(int));
		ret = starpu_mpi_recv(handle, 0, 0, even_comm, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(handle, STARPU_R);
		int *xx = (int *)starpu_variable_get_local_ptr(handle);
		FPRINTF(stderr, "[%d][%d] received %d\n", world_rank, even_rank, *xx);
		STARPU_ASSERT_MSG(*xx==42, "Received value %d is incorrect (should be %d)\n", *xx, 42);
		starpu_data_release(handle);
		starpu_data_unregister(handle);
	}

	if (world_rank == 0)
	{
		starpu_data_handle_t handle;
		int data=144;
		starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&data, sizeof(int));
		ret = starpu_mpi_send(handle, 1, 0, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		starpu_data_unregister(handle);
	}
	else if (world_rank == 1)
	{
		starpu_data_handle_t handle;
		starpu_variable_data_register(&handle, -1, (uintptr_t)NULL, sizeof(int));
		ret = starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(handle, STARPU_R);
		int *xx = (int *)starpu_variable_get_local_ptr(handle);
		FPRINTF(stderr, "[%d][%d] received %d\n", world_rank, even_rank, *xx);
		STARPU_ASSERT_MSG(*xx==144, "Received value %d is incorrect (should be %d)\n", *xx, 42);
		starpu_data_release(handle);
		starpu_data_unregister(handle);
	}

	starpu_mpi_shutdown();

	MPI_Group_free(&world_group);
	if (even_comm != MPI_COMM_NULL)
	{
		MPI_Group_free(&even_group);
		MPI_Comm_free(&even_comm);
	}

	MPI_Finalize();
	return 0;
#else
	return STARPU_TEST_SKIPPED;
#endif
}
