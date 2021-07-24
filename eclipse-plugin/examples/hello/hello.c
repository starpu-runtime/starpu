/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

void display_cpu_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	int nx, i;
	struct starpu_vector_interface *vector;
	int *val;

	vector = (struct starpu_vector_interface *) buffers[0];
	nx = STARPU_VECTOR_GET_NX(vector);
	val = (int *)STARPU_VECTOR_GET_PTR(vector);

	for (i = 0; i < nx; i++)
		fprintf(stdout, "V[%d] = %d\n", i, val[i]);
}

void scal_cpu_func(void *buffers[], void *cl_arg)
{
	int factor, nx, i;
	struct starpu_vector_interface *vector;
	int *val;

	vector = (struct starpu_vector_interface *) buffers[0];
	nx = STARPU_VECTOR_GET_NX(vector);
	val = (int *)STARPU_VECTOR_GET_PTR(vector);
	starpu_codelet_unpack_args(cl_arg, &factor);

	for (i = 0; i < nx; i++)
		val[i] *= factor;
}

void hello_cpu_func(void *buffers[], void *cl_arg)
{
	(void)buffers;
	int answer;

	starpu_codelet_unpack_args(cl_arg, &answer);
	fprintf(stdout, "Hello world, the answer is %d\n", answer);
}

struct starpu_codelet hello_codelet =
{
	.cpu_funcs = {hello_cpu_func},
	.cpu_funcs_name = {"hello_cpu_func"},
	.nbuffers = 0,
	.name = "hello"
};

struct starpu_codelet scal_codelet =
{
	.cpu_funcs = {scal_cpu_func},
	.cpu_funcs_name = {"scal_cpu_func"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "scal"
};

struct starpu_codelet display_codelet =
{
	.cpu_funcs = {display_cpu_func},
	.cpu_funcs_name = {"display_cpu_func"},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.name = "display"
};

#define NX 5

int main(void)
{
	int answer = 42;
	int ret;
	int vector[NX];
	unsigned i;
	starpu_data_handle_t vector_handle;

	setenv("STARPU_FXT_TRACE", "1", 1);

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	for (i = 0; i < NX; i++)
                vector[i] = i+1;
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

	ret = starpu_task_insert(&hello_codelet,
				 STARPU_VALUE, &answer, sizeof(answer),
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&scal_codelet,
				 STARPU_RW, vector_handle,
				 STARPU_VALUE, &answer, sizeof(answer),
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&display_codelet,
				 STARPU_R, vector_handle,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_data_unregister(vector_handle);
	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
