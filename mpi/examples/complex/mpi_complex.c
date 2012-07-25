/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

int main(int argc, char **argv)
{
	int rank, nodes;
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	starpu_mpi_initialize_extended(&rank, &nodes);

	if (nodes < 2)
	{
		fprintf(stderr, "This program needs at least 2 nodes\n");
		ret = 77;
	}
	else
	{
		if (rank == 0)
		{
			double real[2] = {4.0, 2.0};
			double imaginary[2] = {7.0, 9.0};
			starpu_data_handle_t handle;

			starpu_complex_data_register(&handle, 0, real, imaginary, 2);
			starpu_insert_task(&cl_display, STARPU_R, handle, 0);
			starpu_mpi_send(handle, 1, 10, MPI_COMM_WORLD);
#warning todo: send data back and test equality
		}
		else if (rank == 1)
		{
			double real[2] = {0.0, 0.0};
			double imaginary[2] = {0.0, 0.0};
			starpu_data_handle_t handle;
			MPI_Status status;

			starpu_complex_data_register(&handle, 0, real, imaginary, 2);
			starpu_mpi_recv(handle, 0, 10, MPI_COMM_WORLD, &status);
			starpu_insert_task(&cl_display, STARPU_R, handle, 0);
		}
	}
	starpu_task_wait_for_all();
	starpu_mpi_shutdown();
	starpu_shutdown();

	return ret;
}
