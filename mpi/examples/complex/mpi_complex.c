/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void display_foo_codelet(void *descr[], void *_args)
{
	(void)_args;
	int *foo = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	FPRINTF(stderr, "foo = %d\n", *foo);
}

struct starpu_codelet foo_display =
{
	.cpu_funcs = {display_foo_codelet},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int rank, nodes;
	int ret;
	int compare=0;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2 || (starpu_cpu_worker_get_count() == 0))
	{
		if (rank == 0)
		{
			if (nodes < 2)
				fprintf(stderr, "We need at least 2 processes.\n");
			else
				fprintf(stderr, "We need at least 1 CPU.\n");
		}
		starpu_mpi_shutdown();
		return 77;
	}

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

	// Ping-pong
	if (rank == 0)
	{
		int *compare_ptr = &compare;

		starpu_task_insert(&cl_display, STARPU_VALUE, "node0 initial value", strlen("node0 initial value")+1, STARPU_R, handle, 0);
		starpu_mpi_isend_detached(handle, 1, 10, MPI_COMM_WORLD, NULL, NULL);
		starpu_mpi_irecv_detached(handle2, 1, 20, MPI_COMM_WORLD, NULL, NULL);

		starpu_task_insert(&cl_display, STARPU_VALUE, "node0 received value", strlen("node0 received value")+1, STARPU_R, handle2, 0);
		starpu_task_insert(&cl_compare, STARPU_R, handle, STARPU_R, handle2, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), 0);
	}
	else if (rank == 1)
	{
		starpu_mpi_irecv_detached(handle, 0, 10, MPI_COMM_WORLD, NULL, NULL);
		starpu_task_insert(&cl_display, STARPU_VALUE, "node1 received value", strlen("node1 received value")+1, STARPU_R, handle, 0);
		starpu_mpi_isend_detached(handle, 0, 20, MPI_COMM_WORLD, NULL, NULL);
	}

	// Ping
	if (rank == 0)
	{
		starpu_data_handle_t xhandle;
		double xreal = 4.0;
		double ximaginary = 8.0;
		starpu_complex_data_register(&xhandle, STARPU_MAIN_RAM, &xreal, &ximaginary, 1);
		starpu_mpi_send(xhandle, 1, 10, MPI_COMM_WORLD);
		starpu_data_unregister(xhandle);
	}
	else if (rank == 1)
	{
		MPI_Status status;
		starpu_data_handle_t xhandle;
		double xreal = 14.0;
		double ximaginary = 18.0;
		starpu_complex_data_register(&xhandle, STARPU_MAIN_RAM, &xreal, &ximaginary, 1);
		starpu_mpi_recv(xhandle, 0, 10, MPI_COMM_WORLD, &status);
		starpu_data_unregister(xhandle);
		FPRINTF(stderr, "[received] real %f imaginary %f\n", xreal, ximaginary);
		STARPU_ASSERT_MSG(xreal == 4 && ximaginary == 8, "Incorrect received value\n");
	}

	starpu_task_wait_for_all();

	starpu_data_unregister(handle);
	starpu_data_unregister(handle2);

	starpu_mpi_shutdown();

	return (rank == 0) ? !compare : 0;
}
