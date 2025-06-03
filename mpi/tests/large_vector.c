/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2024-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <sys/mman.h>
#include <starpu_mpi.h>
#include "helper.h"

void func_cpu(void *_descr[], void *_args)
{
	(void)_descr;
	(void)_args;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.modes = {STARPU_RW}
};

void print_buffer(int rank, char *buffer, size_t nx)
{
	size_t x;
	FPRINTF(stdout, "[%d] ", rank);
	if (nx > 25)
	{
		for(x = 0; x < 10; x++)
		{
			FPRINTF(stdout, "'%c' ", buffer[x]);
		}
		FPRINTF(stdout, " ... ");
		for(x = nx-10; x < nx; x++)
		{
			FPRINTF(stdout, "'%c' ", buffer[x]);
		}
	}
	else
	{
		for(x = 0; x < nx; x++)
		{
			FPRINTF(stdout, "'%c' ", buffer[x]);
		}
	}
	FPRINTF(stdout, "\n");
}

void init_buffer(int rank, char *buffer, size_t nx)
{
	if (rank == 0)
	{
		size_t x;
		int value='a';
		for(x = 0; x < nx; x++)
		{
			buffer[x] = value;
			value ++;
			if (value > 'z')
				value = 'a';
		}
		print_buffer(rank, buffer, nx);
	}

}

void check_buffer(int rank, char *buffer, size_t nx)
{
	if (rank != 0)
	{
		print_buffer(rank, buffer, nx);

		size_t x;
		int value='a';
		for(x = 0; x < nx; x++)
		{
			if (rank == 1)
			{
				STARPU_ASSERT_MSG(buffer[x]==value, "[rank %d] Expected value %c is not received value %c\n", rank, value, buffer[x]);
				value ++;
				if (value > 'z')
					value = 'a';
			}
			else
			{
				STARPU_ASSERT_MSG(buffer[x]==0, "[rank %d] Value %c has been modified in %c\n", rank, buffer[x], 0);
			}
		}
	}
}

void check_dataset(int rank, int vector_or_matrix, size_t nx, size_t elemsize)
{
	char *buffer;
	starpu_data_handle_t data_handle;
	int ret;

	FPRINTF_MPI(stderr, "Checking with %zu elements of size %zu\n", nx, elemsize);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	buffer = mmap(NULL, nx*elemsize*sizeof(buffer[0]), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
	if (buffer == MAP_FAILED)
	{
		perror("mmap");
		exit(-1);
	}
	init_buffer(rank, buffer, nx*elemsize);

	if (vector_or_matrix)
		starpu_vector_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, nx, elemsize);
	else
		starpu_matrix_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, nx, nx, 1, elemsize);
	starpu_mpi_data_register(data_handle, 42, 0);

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_RW, data_handle,
				     STARPU_EXECUTE_ON_NODE, 1,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");

	starpu_task_wait_for_all();
	starpu_data_unregister(data_handle);
	check_buffer(rank, buffer, nx*elemsize);
}

int main(int argc, char **argv)
{
	int rank, size;
	int ret;
	struct starpu_conf conf;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

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

//	check_dataset(rank, 1, 10, 1);
	check_dataset(rank, 0, 10, 1);
//
//	check_dataset(rank, 1, INT_MAX, 1);
//	check_dataset(rank, 1, INT_MAX, 2);
	check_dataset(rank, 0, INT_MAX, 1);
//
//	check_dataset(rank, 1, (size_t)2*INT_MAX+12, 1);
	check_dataset(rank, 0, (size_t)2*INT_MAX+12, 1);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();


	return 0;
}
