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
#include "helper.h"

void callback(void *arg)
{
	FPRINTF_MPI(stderr, "value in callback: %d\n", *((int *)arg));
}

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;
	starpu_data_handle_t src_handle, dst_handle;
	int value;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
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

	{
		value = rank;
		starpu_variable_data_register(&src_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(src_handle, 12, 0);
		starpu_variable_data_register(&dst_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(dst_handle, 42, 1);

		FPRINTF_MPI(stderr, "value before copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == rank, "before copy value %d should be %d\n", value, rank);
		starpu_mpi_data_cpy(dst_handle, src_handle, MPI_COMM_WORLD, 0, callback, &value);
		starpu_data_unregister(src_handle);
		starpu_data_unregister(dst_handle);

		FPRINTF_MPI(stderr, "value after copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == 0, "after copy value %d should be %d\n", value, 0);
	}

	{
		value = rank+12;
		starpu_variable_data_register(&src_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(src_handle, 12, 0);
		starpu_variable_data_register(&dst_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(dst_handle, 42, 1);

		FPRINTF_MPI(stderr, "value before copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == rank+12, "before copy value %d should be %d\n", value, rank+12);
		starpu_mpi_data_cpy(dst_handle, src_handle, MPI_COMM_WORLD, 1, callback, &value);
		starpu_data_unregister(src_handle);
		starpu_data_unregister(dst_handle);

		FPRINTF_MPI(stderr, "value after copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == 12, "after copy value %d should be %d\n", value, 12);
	}

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
