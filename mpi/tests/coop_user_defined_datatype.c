/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Copy of test user_defined_datatype.c, but with coop */

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

static int my_rank, worldsize, is_sender;

void test_handle_recv_send(starpu_data_handle_t *handles, int nb_handles, starpu_mpi_tag_t tag)
{
	int i, j;
	int ret;

	if (is_sender)
	{
		for(i=0 ; i<nb_handles ; i++)
		{
			starpu_mpi_coop_sends_data_handle_nb_sends(handles[i], worldsize-1);
			for (j = 1; j < worldsize; j++)
			{
				ret = starpu_mpi_isend_detached(handles[i], j, i+tag, MPI_COMM_WORLD, NULL, NULL);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
			}
		}
	}
	else
	{
		for(i=0 ; i<nb_handles ; i++)
		{
			ret = starpu_mpi_recv(handles[i], 0, i+tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		}
	}
}

int main(int argc, char **argv)
{
	int ret;
	int compare = 0;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	if (worldsize < 4)
	{
		fprintf(stderr, "This program needs at least 4 nodes\n");
		ret = 77;
	}
	else
	{
		int i;
		is_sender = (my_rank == 0);

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

		if (is_sender)
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
		else
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
		for(i=0 ; i<ELEMENTS ; i++)
		{
			starpu_complex_data_register(&handle_complex[i], STARPU_MAIN_RAM, real[i], imaginary[i], 2);
			starpu_value_data_register(&handle_values[i], STARPU_MAIN_RAM, &values[i]);
			starpu_variable_data_register(&handle_vars[i], STARPU_MAIN_RAM, (uintptr_t)&foo[i], sizeof(float));
		}

		test_handle_recv_send(handle_vars, ELEMENTS, ELEMENTS);
		test_handle_recv_send(handle_complex, ELEMENTS, 2*ELEMENTS);
		test_handle_recv_send(handle_values, ELEMENTS, 4*ELEMENTS);

		starpu_task_wait_for_all();

		for(i=0 ; i<ELEMENTS ; i++)
		{
			starpu_data_unregister(handle_complex[i]);
			starpu_data_unregister(handle_values[i]);
			starpu_data_unregister(handle_vars[i]);
		}

		if (my_rank == worldsize-1) // the last process to receive data will check its content:
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

	starpu_mpi_barrier(MPI_COMM_WORLD);

	starpu_mpi_shutdown();

	return (my_rank == worldsize-1) ? !compare : ret;
}
