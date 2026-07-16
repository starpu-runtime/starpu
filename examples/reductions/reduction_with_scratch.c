/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This example illustrates how to use STARPU_REDUX mode
 * with sratch data allocated for the reduction codelets.
 */

// Headers

#include <starpu.h>
#include <assert.h>
#include <math.h>

// Macros

#define NX      16
#define NBLOCKS 4

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

// Codelets

static void cl_cpu_print(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vectors
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *u  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);

	// Check
	int check = EXIT_SUCCESS;
	for (int i = 0; i < NX; i++)
	{
		if (u[i] != (NBLOCKS-1)) check = EXIT_FAILURE;
	}

	// Output
	u[0] = check;

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

static void cl_cpu_work(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vectors
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *u  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);
	int *v  = (int *) STARPU_VECTOR_GET_PTR(handles[1]);

	// Accumulate
	int max_u = max_array(u, nx);
	int max_v = max_array(v, nx);
	if (max_v > max_u) memcpy(u, v, nx*sizeof(int));
}

static struct starpu_codelet work_cl =
{
	.cpu_funcs = {cl_cpu_work},
	.nbuffers = 2,
	.modes = {STARPU_REDUX, STARPU_R},
	.name = "work"
};

static void cl_cpu_task_init(void *handles[], void*arg)
{
	// UNUSED
	(void) arg;

	// Get vector
	int  nx = (int)   STARPU_VECTOR_GET_NX(handles[0]);
	int *u  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);

	// Init
	for (int i = 0; i < nx; i++)
	{
		u[i] = 0;
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
	int *u  = (int *) STARPU_VECTOR_GET_PTR(handles[0]);
	int *v  = (int *) STARPU_VECTOR_GET_PTR(handles[1]);

	// Get scratch data
	int  ny = (int)   STARPU_VECTOR_GET_NX(handles[2]);
	int *s  = (int *) STARPU_VECTOR_GET_PTR(handles[2]);

	// Accumulate
	memcpy(s,    u, nx*sizeof(int));
	memcpy(s+nx, v, nx*sizeof(int));
	int max_s = max_array(s, ny);
	for (int i = 0; i < nx; i++)
	{
		u[i] = max_s;
	}
}

static struct starpu_codelet task_red_cl =
{
	.cpu_funcs = {cl_cpu_task_red},
	.nbuffers = 3,
	.modes = {STARPU_RW|STARPU_COMMUTE, STARPU_R, STARPU_SCRATCH},
	// .nbuffers = 2,
	// .modes = {STARPU_RW|STARPU_COMMUTE, STARPU_R},
	.name = "task_red"
};

// Example

int main(int argc, char *argv[])
{
	// StarPU Init
	int ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	// Init reduction data
	int *u = NULL;
	starpu_malloc((void **) &u, NX * sizeof(int));
	for (int i = 0; i < NX; i++)
	{
		u[i] = 0;
	}

	// Register reduction data
	starpu_data_handle_t u_h;
	starpu_vector_data_register(&u_h, STARPU_MAIN_RAM, (uintptr_t) u, NX, sizeof(int));

	// Init accumulation data
	int n_elt = NX * NBLOCKS;
	int *v = NULL;
	starpu_malloc((void **) &v, n_elt * sizeof(int));
	for (int i_block = 0; i_block < NBLOCKS; i_block++)
	{
		for (int i = 0; i < NX; i++)
		{
			v[i_block*NX+i] = i_block;
		}
	}

	// Register accumulation data
	starpu_data_handle_t v_h[NBLOCKS];
	for (int i_block = 0; i_block < NBLOCKS; i_block++)
	{
		starpu_vector_data_register(&v_h[i_block], STARPU_MAIN_RAM, (uintptr_t) &v[i_block*NX], NX, sizeof(int));
	}

	// Register scratch data
	starpu_data_handle_t red_scratch_h;
	starpu_vector_data_register(&red_scratch_h, -1, 0, 2*NX, sizeof(int));

	// Set reduction methods
	starpu_data_set_reduction_methods(u_h, &task_red_cl, &task_init_cl);

	// Set reduction scratch
	starpu_data_set_reduction_scratch(u_h, red_scratch_h);

	// Task
	for (int i_block = 0; i_block < NBLOCKS; i_block++)
	{
	    starpu_task_insert(&work_cl,
			       STARPU_REDUX, u_h,
			       STARPU_R,     v_h[i_block],
			       0);
	}

	// Check
	starpu_task_insert(&print_cl, STARPU_R, u_h, 0);

	// Unregister data
	for (int i_block = 0; i_block < NBLOCKS; i_block++)
	{
		starpu_data_unregister(v_h[i_block]);
	}
	starpu_data_unregister(u_h);
	starpu_data_unregister(red_scratch_h);

	// Free
	int check = u[0];
	starpu_free_noflag(u, NX * sizeof(int));
	starpu_free_noflag(v, n_elt * sizeof(int));

	// StarPU Finalize
	starpu_shutdown();

	return check;
}
