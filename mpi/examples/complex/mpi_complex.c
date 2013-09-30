/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012, 2013  Centre National de la Recherche Scientifique
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
#include <interface/complex_interface.h>
#include <interface/complex_codelet.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void display_foo_codelet(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	int *foo = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	FPRINTF(stderr, "foo = %d\n", *foo);
}

struct starpu_codelet foo_display =
{
	.cpu_funcs = {display_foo_codelet, NULL},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

int main(int argc, char **argv)
{
	int rank, nodes;
	int ret;
	int compare;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(&argc, &argv, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2)
	{
		fprintf(stderr, "This program needs at least 2 nodes (%d available)\n", nodes);
		ret = 77;
	}
	else
	{
		starpu_data_handle_t handle;
		starpu_data_handle_t handle2;

		double real[2] = {4.0, 2.0};
		double imaginary[2] = {7.0, 9.0};

		double real2[2] = {14.0, 12.0};
		double imaginary2[2] = {17.0, 19.0};

		if (rank == 1)
		{
			real[0] = 0.0;
			real[1] = 0.0;
			imaginary[0] = 0.0;
			imaginary[1] = 0.0;
		}

		starpu_complex_data_register(&handle, STARPU_MAIN_RAM, real, imaginary, 2);
		starpu_complex_data_register(&handle2, -1, real2, imaginary2, 2);

		if (rank == 0)
		{
			int *compare_ptr = &compare;

			starpu_task_insert(&cl_display, STARPU_VALUE, "node0 initial value", strlen("node0 initial value"), STARPU_R, handle, 0);
			starpu_mpi_isend_detached(handle, 1, 10, MPI_COMM_WORLD, NULL, NULL);
			starpu_mpi_irecv_detached(handle2, 1, 20, MPI_COMM_WORLD, NULL, NULL);

			starpu_task_insert(&cl_display, STARPU_VALUE, "node0 received value", strlen("node0 received value"), STARPU_R, handle2, 0);
			starpu_task_insert(&cl_compare, STARPU_R, handle, STARPU_R, handle2, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), 0);
		}
		else if (rank == 1)
		{
			starpu_mpi_irecv_detached(handle, 0, 10, MPI_COMM_WORLD, NULL, NULL);
			starpu_task_insert(&cl_display, STARPU_VALUE, "node1 received value", strlen("node1 received value"), STARPU_R, handle, 0);
			starpu_mpi_isend_detached(handle, 0, 20, MPI_COMM_WORLD, NULL, NULL);
		}

		starpu_task_wait_for_all();

		starpu_data_unregister(handle);
		starpu_data_unregister(handle2);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();

	if (rank == 0) return !compare; else return ret;
}
