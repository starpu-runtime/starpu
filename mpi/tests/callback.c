/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013, 2014  Centre National de la Recherche Scientifique
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

static
int expected_x=40;
static
int expected_y=12;

void my_func(STARPU_ATTRIBUTE_UNUSED void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	FPRINTF_MPI("i am here\n");
}

struct starpu_codelet my_codelet =
{
	.cpu_funcs = {my_func, NULL},
	.cuda_funcs = {my_func, NULL},
	.opencl_funcs = {my_func, NULL}
};

static
void callback(void *ptr)
{
	int *x = (int *)ptr;
	FPRINTF_MPI("x=%d\n", *x);
	STARPU_ASSERT_MSG(*x == expected_x, "%d != %d\n", *x, expected_x);
	(*x)++;
}

static
void prologue_callback(void *ptr)
{
	int *y = (int *)ptr;
	FPRINTF_MPI("y=%d\n", *y);
	STARPU_ASSERT_MSG(*y == expected_y, "%d != %d\n", *y, expected_y);
	(*y)++;
}

int main(int argc, char **argv)
{
	int ret;
	int x=40;
	int y=12;
	int rank, size;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_mpi_init(&argc, &argv, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
				     NULL,
				     STARPU_EXECUTE_ON_NODE, 0,
				     STARPU_CALLBACK_WITH_ARG, callback, &x,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");

	if (rank == 0) expected_x ++;
	ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
				     NULL,
				     STARPU_EXECUTE_ON_NODE, 0,
				     STARPU_CALLBACK, callback,
				     STARPU_CALLBACK_ARG, &x,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	if (rank == 0) expected_x ++;
	STARPU_ASSERT_MSG(x == expected_x, "x should be equal to %d and not %d\n", expected_x, x);

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
				     NULL,
				     STARPU_EXECUTE_ON_NODE, 0,
				     STARPU_PROLOGUE_CALLBACK, prologue_callback,
				     STARPU_PROLOGUE_CALLBACK_ARG, &y,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	if (rank == 0) expected_y ++;
	ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
				     &my_codelet,
				     STARPU_EXECUTE_ON_NODE, 0,
				     STARPU_PROLOGUE_CALLBACK_POP, prologue_callback,
				     STARPU_PROLOGUE_CALLBACK_POP_ARG, &y,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();
	if (rank == 0) expected_y ++;
	STARPU_ASSERT_MSG(y == expected_y, "y should be equal to %d and not %d\n", expected_y, y);

	starpu_mpi_shutdown();
	starpu_shutdown();

	return EXIT_SUCCESS;
}

