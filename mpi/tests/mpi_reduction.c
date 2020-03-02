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
#include <math.h>
#include "helper.h"

extern void init_cpu_func(void *descr[], void *cl_arg);
extern void redux_cpu_func(void *descr[], void *cl_arg);
extern void dot_cpu_func(void *descr[], void *cl_arg);
extern void display_cpu_func(void *descr[], void *cl_arg);

static struct starpu_codelet init_codelet =
{
	.cpu_funcs = {init_cpu_func},
	.nbuffers = 1,
	.modes = {STARPU_W},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "init_codelet"
};

static struct starpu_codelet redux_codelet =
{
	.cpu_funcs = {redux_cpu_func},
	.modes = {STARPU_RW, STARPU_R},
	.nbuffers = 2,
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "redux_codelet"
};

static struct starpu_codelet dot_codelet =
{
	.cpu_funcs = {dot_cpu_func},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_REDUX},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "dot_codelet"
};

static struct starpu_codelet display_codelet =
{
	.cpu_funcs = {display_cpu_func},
	.nbuffers = 1,
	.modes = {STARPU_R},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "display_codelet"
};

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int nb_nodes)
{
	return x % nb_nodes;
}

int main(int argc, char **argv)
{
	int my_rank, size, x, y, i;
	long int *vector;
	long int dot, sum=0;
	starpu_data_handle_t *handles;
	starpu_data_handle_t dot_handle;
	int ret;

	int nb_elements, step, loops;

	STARPU_SKIP_IF_VALGRIND_RETURN_SKIP;

	/* Not supported yet */
	if (starpu_get_env_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return STARPU_TEST_SKIPPED;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (starpu_cpu_worker_get_count() == 0)
	{
		if (my_rank == 0)
			FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	nb_elements = size*8000;
	step = 4;
	loops = 5;

	vector = (long int *) malloc(nb_elements*sizeof(vector[0]));
	for(x = 0; x < nb_elements; x+=step)
	{
		int mpi_rank = my_distrib(x/step, size);
		if (mpi_rank == my_rank)
		{
			for(y=0 ; y<step ; y++)
			{
				vector[x+y] = x+y+1;
			}
		}
	}
	if (my_rank == 0)
	{
		dot = 14;
		sum = (nb_elements * (nb_elements + 1)) / 2;
		sum *= loops;
		sum += dot;
		starpu_variable_data_register(&dot_handle, STARPU_MAIN_RAM, (uintptr_t)&dot, sizeof(dot));
	}
	else
	{
		starpu_variable_data_register(&dot_handle, -1, (uintptr_t)NULL, sizeof(dot));
	}


	handles = (starpu_data_handle_t *) malloc(nb_elements*sizeof(handles[0]));
	for(x = 0; x < nb_elements; x+=step)
	{
		handles[x] = NULL;
		int mpi_rank = my_distrib(x/step, size);
		if (mpi_rank == my_rank)
		{
			/* Owning data */
			starpu_vector_data_register(&handles[x], STARPU_MAIN_RAM, (uintptr_t)&(vector[x]), step, sizeof(vector[0]));
		}
		else
		{
			starpu_vector_data_register(&handles[x], -1, (uintptr_t)NULL, step, sizeof(vector[0]));
		}
		if (handles[x])
		{
			starpu_mpi_data_register(handles[x], x, mpi_rank);
		}
	}

	starpu_mpi_data_register(dot_handle, nb_elements+1, 0);
	starpu_data_set_reduction_methods(dot_handle, &redux_codelet, &init_codelet);

	for (i = 0; i < loops; i++)
	{
		for (x = 0; x < nb_elements; x+=step)
		{
			starpu_mpi_task_insert(MPI_COMM_WORLD,
					       &dot_codelet,
					       STARPU_R, handles[x],
					       STARPU_REDUX, dot_handle,
					       0);
		}
		starpu_mpi_redux_data(MPI_COMM_WORLD, dot_handle);
		starpu_mpi_task_insert(MPI_COMM_WORLD, &display_codelet, STARPU_R, dot_handle, 0);
	}

	FPRINTF_MPI(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	for(x = 0; x < nb_elements; x+=step)
	{
		if (handles[x])
			starpu_data_unregister(handles[x]);
	}
	starpu_data_unregister(dot_handle);
	free(vector);
	free(handles);

	starpu_mpi_shutdown();

	if (my_rank == 0)
	{
		FPRINTF(stderr, "[%d] sum=%ld\n", my_rank, sum);
	}

#ifndef STARPU_SIMGRID
	if (my_rank == 0)
	{
		FPRINTF(stderr, "[%d] dot=%ld\n", my_rank, dot);
		FPRINTF(stderr, "%s when computing reduction\n", (sum == dot) ? "Success" : "Error");
		if (sum != dot)
			return 1;
	}
#endif

	return 0;
}

