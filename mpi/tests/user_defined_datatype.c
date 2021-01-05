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
#include <user_defined_datatype_value.h>
#include "helper.h"

#ifdef STARPU_QUICK_CHECK
#  define ELEMENTS 10
#else
#  define ELEMENTS 1000
#endif

typedef void (*test_func)(starpu_data_handle_t *, int, int, starpu_mpi_tag_t);

void test_handle_irecv_isend_detached(starpu_data_handle_t *handles, int nb_handles, int rank, starpu_mpi_tag_t tag)
{
	int i;
	(void)rank;

	for(i=0 ; i<nb_handles ; i++)
	{
		starpu_mpi_data_register(handles[i], i+tag, 1);
	}

	for(i=0 ; i<nb_handles ; i++)
		starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, handles[i], 0, NULL, NULL);
}

void test_handle_recv_send(starpu_data_handle_t *handles, int nb_handles, int rank, starpu_mpi_tag_t tag)
{
	int i;

	if (rank == 1)
	{
		for(i=0 ; i<nb_handles ; i++)
			starpu_mpi_send(handles[i], 0, i+tag, MPI_COMM_WORLD);
	}
	else if (rank == 0)
	{
		MPI_Status statuses[nb_handles];
		for(i=0 ; i<nb_handles ; i++)
			starpu_mpi_recv(handles[i], 1, i+tag, MPI_COMM_WORLD, &statuses[i]);
	}
}


int main(int argc, char **argv)
{
	int rank, nodes;
	int ret;
	int compare = 0;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2)
	{
		fprintf(stderr, "This program needs at least 2 nodes\n");
		ret = 77;
	}
	else
	{
		test_func funcs[3] = {test_handle_recv_send, test_handle_irecv_isend_detached, NULL};
		test_func *func;
		for(func=funcs ; *func!=NULL ; func++)
		{
			test_func f = *func;
			int i;

			starpu_data_handle_t handle_complex[ELEMENTS];
			starpu_data_handle_t handle_values[ELEMENTS];
			starpu_data_handle_t handle_vars[ELEMENTS];

			double real[ELEMENTS][2];
			double imaginary[ELEMENTS][2];
			float foo[ELEMENTS];
			int values[ELEMENTS];

			double real_compare[2] = {12.0, 45.0};
			double imaginary_compare[2] = {7.0, 42.0};
			float foo_compare=42.0;
			int value_compare=36;

			FPRINTF_MPI(stderr, "\nTesting with function %p\n", f);

			if (rank == 0)
			{
				for(i=0 ; i<ELEMENTS; i++)
				{
					foo[i] = -1.0;
					real[i][0] = -1.0;
					real[i][1] = -1.0;
					imaginary[i][0] = -1.0;
					imaginary[i][1] = -1.0;
					values[i] = -1;
				}
			}
			if (rank == 1)
			{
				for(i=0 ; i<ELEMENTS; i++)
				{
					foo[i] = foo_compare;
					real[i][0] = real_compare[0];
					real[i][1] = real_compare[1];
					imaginary[i][0] = imaginary_compare[0];
					imaginary[i][1] = imaginary_compare[1];
					values[i] = value_compare;
				}
			}
			for(i=0 ; i<ELEMENTS ; i++)
			{
				starpu_complex_data_register(&handle_complex[i], STARPU_MAIN_RAM, real[i], imaginary[i], 2);
				starpu_value_data_register(&handle_values[i], STARPU_MAIN_RAM, &values[i]);
				starpu_variable_data_register(&handle_vars[i], STARPU_MAIN_RAM, (uintptr_t)&foo[i], sizeof(float));
			}

			f(handle_vars, ELEMENTS, rank, ELEMENTS);
			f(handle_complex, ELEMENTS, rank, 2*ELEMENTS);
			f(handle_values, ELEMENTS, rank, 4*ELEMENTS);

			starpu_task_wait_for_all();

			for(i=0 ; i<ELEMENTS ; i++)
			{
				starpu_data_unregister(handle_complex[i]);
				starpu_data_unregister(handle_values[i]);
				starpu_data_unregister(handle_vars[i]);
			}

			if (rank == 0)
			{
				for(i=0 ; i<ELEMENTS ; i++)
				{
					int j;
					compare = (foo[i] == foo_compare);
					FPRINTF_MPI(stderr, "%s. foo[%d] = %f %s %f\n", compare==0?"ERROR":"SUCCESS", i, foo[i], compare==0?"!=":"==", foo_compare);

					compare = (values[i] == value_compare);
					FPRINTF_MPI(stderr, "%s. value[%d] = %d %s %d\n", compare==0?"ERROR":"SUCCESS", i, values[i], compare==0?"!=":"==", value_compare);

					for(j=0 ; j<2 ; j++)
					{
						compare = (real[i][j] == real_compare[j]);
						FPRINTF_MPI(stderr, "%s. real[%d][%d] = %f %s %f\n", compare==0?"ERROR":"SUCCESS", i, j, real[i][j], compare==0?"!=":"==", real_compare[j]);
					}
					for(j=0 ; j<2 ; j++)
					{
						compare = (imaginary[i][j] == imaginary_compare[j]);
						FPRINTF_MPI(stderr, "%s. imaginary[%d][%d] = %f %s %f\n", compare==0?"ERROR":"SUCCESS", i, j, imaginary[i][j], compare==0?"!=":"==", imaginary_compare[j]);
					}
				}
			}
		}
	}

	starpu_mpi_shutdown();

	return (rank == 0) ? !compare : ret;
}
