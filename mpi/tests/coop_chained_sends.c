/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define NX 20

void scal_cpu_func(void *buffers[], void *cl_arg)
{
	unsigned i;
	struct starpu_vector_interface *vector = buffers[0];
	unsigned n = STARPU_VECTOR_GET_NX(vector);
	float *val = (float *) STARPU_VECTOR_GET_PTR(vector);

	/* scale the vector */
	for (i = 0; i < n; i++)
		val[i] *= 2;
}

static struct starpu_codelet cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = { scal_cpu_func },
	.cpu_funcs_name = { "scal_cpu_func" },
	.nbuffers = 1,
	.modes = { STARPU_RW }
};

int main(int argc, char **argv)
{
	int ret, rank, size;
	starpu_data_handle_t handle;
	int mpi_init;
	int i = 0, n = 0;
	MPI_Status status;
	struct starpu_conf conf;

	float* vector = malloc(NX * sizeof(float));

	for (i = 0; i < NX; i++)
	{
		vector[i] = 1.0f;
	}

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t) vector, NX, sizeof(float));

	if (rank == 0)
	{
		ret = starpu_task_insert(&cl, STARPU_RW, handle, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* The task previously inserted should be enough to detect the coop,
		 * but to be sure, indicate the number of sends requests before really
		 * sending the data: */
		starpu_mpi_coop_sends_data_handle_nb_sends(handle, size-1);

		for (n = 1 ; n < size ; n++)
		{
			FPRINTF_MPI(stderr, "sending data to %d\n", n);
			ret = starpu_mpi_isend_detached(handle, n, 0, MPI_COMM_WORLD, NULL, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
		}
	}
	else
	{
		ret = starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		FPRINTF_MPI(stderr, "received data\n");
		starpu_data_acquire(handle, STARPU_R);
		STARPU_ASSERT_MSG(vector[0] == 2, "vector[0] = %f, expected 2\n", vector[0]);
		STARPU_ASSERT_MSG(vector[NX-1] == 2, "vector[%d] = %f, expected 2\n", NX-1, vector[NX-1]);

		starpu_data_release(handle);

		if (rank == 1)
		{
			ret = starpu_task_insert(&cl, STARPU_RW, handle, 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

			starpu_mpi_coop_sends_data_handle_nb_sends(handle, size-2);

			for (i = 2; i < size; i++)
			{
				FPRINTF_MPI(stderr, "sending data to %d\n", i);
				ret = starpu_mpi_isend_detached(handle, i, 1, MPI_COMM_WORLD, NULL, NULL);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
			}
		}
		else
		{
			ret = starpu_mpi_recv(handle, 1, 1, MPI_COMM_WORLD, &status);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
			FPRINTF_MPI(stderr, "received data\n");
			starpu_data_acquire(handle, STARPU_R);
			STARPU_ASSERT_MSG(vector[0] == 4, "vector[0] = %f, expected 4\n", vector[0]);
			STARPU_ASSERT_MSG(vector[NX-1] == 4, "vector[%d] = %f, expected 4\n", NX-1, vector[NX-1]);
			starpu_data_release(handle);
		}
	}

	starpu_data_unregister(handle);

	printf("[%d] end\n", rank);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	free(vector);

	return 0;
}
