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

void display_double_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	double *foo = (double *)STARPU_VARIABLE_GET_PTR(descr[0]);
	fprintf(stderr, "foo = %f\n", *foo);
}

struct starpu_codelet double_display =
{
	.cpu_funcs = {display_double_codelet, NULL},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

void test_handle(starpu_data_handle_t handle, struct starpu_codelet *codelet, int rank)
{
	starpu_data_set_rank(handle, 1);
	starpu_data_set_tag(handle, 42);

	if (rank == 0)
	{
		starpu_insert_task(codelet, STARPU_R, handle, 0);
	}
	starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, handle, 0, NULL, NULL);
	if (rank == 0)
	{
		starpu_insert_task(codelet, STARPU_R, handle, 0);
	}
}

int main(int argc, char **argv)
{
	int rank, nodes;
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(&argc, &argv);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2)
	{
		fprintf(stderr, "This program needs at least 2 nodes\n");
		ret = 77;
	}
	else
	{
		double real[2] = {0.0, 0.0};
		double imaginary[2] = {0.0, 0.0};
		double foo=8;
		starpu_data_handle_t handle_complex;
		starpu_data_handle_t handle_var;

		if (rank == 1)
		{
			foo = 42;
			real[0] = 12.0;
			real[1] = 45.0;
			imaginary[0] = 7.0;
			imaginary[1] = 42.0;
		}
		starpu_complex_data_register(&handle_complex, 0, real, imaginary, 2);
		starpu_variable_data_register(&handle_var, 0, (uintptr_t)&foo, sizeof(double));

		test_handle(handle_var, &double_display, rank);
		test_handle(handle_complex, &cl_display, rank);
	}
	starpu_task_wait_for_all();
	starpu_mpi_shutdown();
	starpu_shutdown();

	return 0;
}
