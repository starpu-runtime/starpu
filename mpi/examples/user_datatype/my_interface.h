/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <mpi.h>

#ifndef __DATA_INTERFACE_H
#define __DATA_INTERFACE_H

struct starpu_my_interface
{
	int d;
	char c;
};

void starpu_my_interface_data_register(starpu_data_handle_t *handle, unsigned home_node, struct starpu_my_interface *xc);

char starpu_my_interface_get_char(starpu_data_handle_t handle);
int starpu_my_interface_get_int(starpu_data_handle_t handle);

#define STARPU_MY_INTERFACE_GET_CHAR(interface)	(((struct starpu_my_interface *)(interface))->c)
#define STARPU_MY_INTERFACE_GET_INT(interface)	(((struct starpu_my_interface *)(interface))->d)

void _starpu_my_interface_datatype_allocate(MPI_Datatype *mpi_datatype);
void starpu_my_interface_datatype_allocate(starpu_data_handle_t handle, MPI_Datatype *mpi_datatype);
void starpu_my_interface_datatype_free(MPI_Datatype *mpi_datatype);

void starpu_my_interface_display_codelet_cpu(void *descr[], void *_args);
void starpu_my_interface_compare_codelet_cpu(void *descr[], void *_args);

static struct starpu_codelet starpu_my_interface_display_codelet =
{
	.cpu_funcs = {starpu_my_interface_display_codelet_cpu},
	.cpu_funcs_name = {"starpu_my_interface_display_codelet_cpu"},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.name = "starpu_my_interface_display_codelet"
};

static struct starpu_codelet starpu_my_interface_compare_codelet =
{
	.cpu_funcs = {starpu_my_interface_compare_codelet_cpu},
	.cpu_funcs_name = {"starpu_my_interface_compare_codelet_cpu"},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_R},
	.name = "starpu_my_interface_compare_codelet"
};

#endif /* __MY_INTERFACE_H */
