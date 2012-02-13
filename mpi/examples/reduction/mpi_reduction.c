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
#include <math.h>

#define X         7

int display = 0;

extern void init_cpu_func(void *descr[], void *cl_arg);
extern void redux_cpu_func(void *descr[], void *cl_arg);
extern void dot_cpu_func(void *descr[], void *cl_arg);

static struct starpu_codelet init_codelet =
{
	.where = STARPU_CPU,
	.cpu_funcs = {init_cpu_func, NULL},
	.nbuffers = 1,
	.name = "init_codelet"
};

static struct starpu_codelet redux_codelet =
{
	.where = STARPU_CPU,
	.cpu_funcs = {redux_cpu_func, NULL},
	.nbuffers = 2,
	.name = "redux_codelet"
};

static struct starpu_codelet dot_codelet =
{
	.where = STARPU_CPU,
	.cpu_funcs = {dot_cpu_func, NULL},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_REDUX},
	.name = "dot_codelet"
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-display") == 0)
		{
			display = 1;
		}
	}
}

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int nb_nodes)
{
	return x % nb_nodes;
}

int main(int argc, char **argv)
{
        int my_rank, size, x;
        int value=0;
        unsigned vector[X];
	unsigned dot, sum=0;
        starpu_data_handle_t handles[X];
	starpu_data_handle_t dot_handle;

	starpu_init(NULL);
	starpu_mpi_initialize_extended(&my_rank, &size);
        parse_args(argc, argv);

        for(x = 0; x < X; x++)
	{
		int mpi_rank = my_distrib(x, size);
		if (mpi_rank == my_rank)
		{
			vector[x] = x+1;
		}
		sum += x+1;
        }
	if (my_rank == 0) {
		dot = 14;
		sum+= dot;
	}

        for(x = 0; x < X; x++)
	{
		int mpi_rank = my_distrib(x, size);
		if (mpi_rank == my_rank)
		{
			/* Owning data */
			starpu_variable_data_register(&handles[x], 0, (uintptr_t)&(vector[x]), sizeof(unsigned));
		}
		else
		{
			starpu_variable_data_register(&handles[x], -1, (uintptr_t)NULL, sizeof(unsigned));
		}
		if (handles[x])
		{
			starpu_data_set_rank(handles[x], mpi_rank);
			starpu_data_set_tag(handles[x], x);
		}
	}

	starpu_variable_data_register(&dot_handle, 0, (uintptr_t)&dot, sizeof(unsigned));
	starpu_data_set_rank(dot_handle, 0);
	starpu_data_set_tag(dot_handle, X+1);
	starpu_data_set_reduction_methods(dot_handle, &redux_codelet, &init_codelet);

	for (x = 0; x < X; x++)
	{
		starpu_mpi_insert_task(MPI_COMM_WORLD,
				       &dot_codelet,
				       STARPU_R, handles[x],
				       STARPU_REDUX, dot_handle,
				       0);
	}
	starpu_mpi_redux_data(MPI_COMM_WORLD, dot_handle);

        fprintf(stderr, "Waiting ...\n");
        starpu_task_wait_for_all();

        for(x = 0; x < X; x++)
	{
		if (handles[x]) starpu_data_unregister(handles[x]);
	}
	if (dot_handle)
	{
		starpu_data_unregister(dot_handle);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();

	if (display && my_rank == 0)
	{
                fprintf(stderr, "[%d] sum=%d\n", my_rank, sum);
                fprintf(stderr, "[%d] dot=%d\n", my_rank, dot);
		if (sum != dot) fprintf(stderr, "Error when computing reduction\n");
        }

	return 0;
}

