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

void display_foo_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	int *foo = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	fprintf(stderr, "foo = %d\n", *foo);
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

	starpu_data_handle_t handle;
	starpu_data_handle_t handle2;
	starpu_data_handle_t foo_handle;

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
		if (rank == 0)
		{
			double real[2] = {4.0, 2.0};
			double imaginary[2] = {7.0, 9.0};

			double real2[2] = {14.0, 12.0};
			double imaginary2[2] = {17.0, 19.0};

			int *compare_ptr = &compare;

			starpu_complex_data_register(&handle, 0, real, imaginary, 2);
			starpu_complex_data_register(&handle2, -1, real2, imaginary2, 2);

			starpu_insert_task(&cl_display, STARPU_R, handle, 0);
			starpu_mpi_isend_detached(handle, 1, 10, MPI_COMM_WORLD, NULL, NULL);
			starpu_mpi_irecv_detached(handle2, 1, 20, MPI_COMM_WORLD, NULL, NULL);

			starpu_insert_task(&cl_display, STARPU_R, handle2, 0);
			starpu_insert_task(&cl_compare, STARPU_R, handle, STARPU_R, handle2, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), 0);

			{
				// We send a dummy variable only to check communication with predefined datatypes
				int foo=12;
				starpu_variable_data_register(&foo_handle, 0, (uintptr_t)&foo, sizeof(foo));
				starpu_mpi_isend_detached(foo_handle, 1, 40, MPI_COMM_WORLD, NULL, NULL);
				starpu_insert_task(&foo_display, STARPU_R, foo_handle, 0);
			}
		}
		else if (rank == 1)
		{
			double real[2] = {0.0, 0.0};
			double imaginary[2] = {0.0, 0.0};

			starpu_complex_data_register(&handle, 0, real, imaginary, 2);
			starpu_mpi_irecv_detached(handle, 0, 10, MPI_COMM_WORLD, NULL, NULL);
			starpu_insert_task(&cl_display, STARPU_R, handle, 0);
			starpu_mpi_isend_detached(handle, 0, 20, MPI_COMM_WORLD, NULL, NULL);

			{
				// We send a dummy variable only to check communication with predefined datatypes
				int foo=12;
				starpu_variable_data_register(&foo_handle, -1, (uintptr_t)NULL, sizeof(foo));
				starpu_mpi_irecv_detached(foo_handle, 0, 40, MPI_COMM_WORLD, NULL, NULL);
				starpu_insert_task(&foo_display, STARPU_R, foo_handle, 0);
			}

		}
	}

	starpu_task_wait_for_all();

	if (rank == 0)
	{
		starpu_data_unregister(handle2);
	}
	if (rank == 0 || rank == 1)
	{
		starpu_data_unregister(handle2);
		starpu_data_unregister(foo_handle);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();

	if (rank == 0) return !compare; else return ret;
}
