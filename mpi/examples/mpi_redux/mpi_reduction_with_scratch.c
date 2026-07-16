/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This example illustrates how to use STARPU_MPI_REDUX mode
 * with sratch data allocated for the reduction codelets.
 */

// Headers

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"
#include <unistd.h>

// Macros

#define NX 32

// Util

static int max_array(int *array, int n)
{
	int max = -1;
	for (int i = 0; i < n; i++)
	{
		if (max < array[i]) max = array[i];
	}

	return max;
}

void scallback(void *arg)
{
	char *msg = arg;
	FPRINTF_MPI(stderr, "Sending completed for <%s>\n", msg);
}

// Codelets

static void cl_cpu_work(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vectors
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *v  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);
	int *w  = (int *) STARPU_VECTOR_GET_PTR(handles[1]);

	// Accumulate
	int max_v = max_array(v, nx);
	int max_w = max_array(w, nx);
	if (max_w > max_v) memcpy(v, w, nx*sizeof(int));
}

static struct starpu_codelet work_cl =
{
	.cpu_funcs = {cl_cpu_work},
	.nbuffers = 2,
	.modes = {STARPU_RW | STARPU_COMMUTE, STARPU_R},
	.name = "work"
};

static void cl_cpu_print(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vectors
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *v  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);

	// Check
	int n_rank = starpu_mpi_world_size();
	int check = EXIT_SUCCESS;
	for (int i = 0; i < NX; i++)
	{
		if (v[i] != (n_rank-1)) check = EXIT_FAILURE;
	}

	// Output
	v[0] = check;

	// Print
	printf("Return %d\n", check);
	fflush(stdout);
}

static struct starpu_codelet print_cl =
{
	.cpu_funcs = {cl_cpu_print},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.name = "print"
};

static void cl_cpu_task_init(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vector
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *v  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);

	// Init
	for (int i = 0; i < nx; i++)
	{
		v[i] = 0;
	}
}

static struct starpu_codelet task_init_cl =
{
	.cpu_funcs = {cl_cpu_task_init},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "task_init"
};

static void cl_cpu_task_red(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vectors
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *v  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);
	int *w  = (int *) STARPU_VECTOR_GET_PTR(handles[1]);

	// Get scratch data
	int  ny = (int)   STARPU_VECTOR_GET_NX(handles[2]);
	int *s  = (int *) STARPU_VECTOR_GET_PTR(handles[2]);

	// Force use of scratch data
	memcpy(s,    v, nx*sizeof(int));
	memcpy(s+nx, w, nx*sizeof(int));

	// Accumulate
	memcpy(s,    v, nx*sizeof(int));
	memcpy(s+nx, w, nx*sizeof(int));
	int max_s = max_array(s, ny);
	for (int i = 0; i < nx; i++)
	{
		v[i] = max_s;
	}
}

static struct starpu_codelet task_red_cl =
{
	.cpu_funcs = {cl_cpu_task_red},
	.nbuffers = 3,
	.modes = {STARPU_RW | STARPU_COMMUTE, STARPU_R, STARPU_SCRATCH},
	.name = "task_red"
};

// Example

int main(int argc, char *argv[])
{
	// MPI Init
	starpu_fxt_autostart_profiling(0);
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conft");

	// Workers
	int nworkers = starpu_cpu_worker_get_count();
	if (nworkers > 1)
	{
		FPRINTF(stderr, "We need only one worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	// MPI Parameters
	int n_rank = 0;
	starpu_mpi_comm_size(MPI_COMM_WORLD, &n_rank);
	if (n_rank < 2)
	{
		FPRINTF(stderr, "We need at least 2 nodes.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	int i_rank = -1;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &i_rank);

	// Init reduction data
	int v[NX];
	if (i_rank == 0)
	{
		for (int i = 0; i < NX; i++)
		{
			v[i] = 0;
		}
	}

	// Register reduction data
	starpu_data_handle_t v_h;
	if (i_rank == 0)
	{
		starpu_vector_data_register(&v_h, STARPU_MAIN_RAM, (uintptr_t)v, NX, sizeof(int));
	}
	else
	{
		starpu_vector_data_register(&v_h, -1, 0, NX, sizeof(int));
	}

	// Init accumulation data
	int **w = malloc(n_rank * sizeof(int *));
	for (int j_rank = 0; j_rank < n_rank; j_rank++)
	{
		if (j_rank == i_rank)
		{
			w[j_rank] = malloc(NX * sizeof(int));
		}
		else
		{
			w[j_rank] = NULL;
		}
	}

	for (int i = 0; i < NX; i++)
	{
		w[i_rank][i] = i_rank;
	}

	// Register accumulation data
	starpu_data_handle_t w_h[n_rank];
	for (int j_rank = 0; j_rank < n_rank; j_rank++)
	{
		if (j_rank == i_rank)
		{
			starpu_vector_data_register(&w_h[j_rank], STARPU_MAIN_RAM, (uintptr_t)w[i_rank], NX, sizeof(int));
		}
		else
		{
			starpu_vector_data_register(&w_h[j_rank], -1, 0, NX, sizeof(int));
		}
	}

	// Register scratch data
	starpu_data_handle_t red_scratch_h;
	starpu_vector_data_register(&red_scratch_h, -1, 0, 2*NX, sizeof(int));

	// MPI Registration
	starpu_mpi_tag_t tag = 0;
	starpu_mpi_data_register(v_h, tag++, 0);
	for (int j_rank = 0; j_rank < n_rank; j_rank++)
	{
		starpu_mpi_data_register(w_h[j_rank], tag++, j_rank);
	}

	// Set reduction methods
	starpu_data_set_reduction_methods(v_h, &task_red_cl, &task_init_cl);

	// Set reduction scratch data
	starpu_data_set_reduction_scratch(v_h, red_scratch_h);

	// Tasks
	for (int j_rank = 0; j_rank < n_rank; j_rank++)
	{

		starpu_mpi_task_insert(MPI_COMM_WORLD,
				       &work_cl,
				       STARPU_MPI_REDUX, v_h,
				       STARPU_R, w_h[j_rank],
				       STARPU_EXECUTE_ON_NODE, j_rank,
				       0);
	}

	// Trigger reduction
	ret = starpu_mpi_redux_data(MPI_COMM_WORLD, v_h);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_redux_data");
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);
	if (i_rank == 0)
	{
		starpu_task_insert(&print_cl, STARPU_R, v_h, 0);
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);
	int check = 1;
	if (i_rank == 0) check = v[0];
	MPI_Bcast(&check,
		  1,
		  MPI_INT,
		  0,
		  MPI_COMM_WORLD);

	// printf("%d: v[0] = %d\n", i_rank, check);

	// Unregister data
	starpu_mpi_barrier(MPI_COMM_WORLD);
	starpu_data_unregister(v_h);
	for (int j_rank = 0; j_rank < n_rank; j_rank++)
	{
		starpu_data_unregister(w_h[j_rank]);
	}

	// Free
	free(w[i_rank]);
	free(w);

	// MPI Finalize
	starpu_mpi_shutdown();

	return check;
}
