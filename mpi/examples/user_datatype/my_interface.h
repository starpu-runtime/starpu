/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

struct starpu_my_data_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */

	uintptr_t ptr;                    /**< local pointer of the data */
	uintptr_t dev_handle;             /**< device handle of the data. */
	size_t offset;                    /**< offset in the data */
};

struct starpu_my_data
{
	int d;
	char c;
};

void starpu_my_data_register(starpu_data_handle_t *handle, unsigned home_node, struct starpu_my_data *xc);
void starpu_my_data2_register(starpu_data_handle_t *handle, unsigned home_node, struct starpu_my_data *xc);

char starpu_my_data_get_char(starpu_data_handle_t handle);
int starpu_my_data_get_int(starpu_data_handle_t handle);

char starpu_my_data_interface_get_char(void *interface);
int starpu_my_data_interface_get_int(void *interface);

#define STARPU_MY_DATA_GET_CHAR(interface)	starpu_my_data_interface_get_char(interface)
#define STARPU_MY_DATA_GET_INT(interface)	starpu_my_data_interface_get_int(interface)

void _starpu_my_data_datatype_allocate(MPI_Datatype *mpi_datatype);
int starpu_my_data_datatype_allocate(starpu_data_handle_t handle, MPI_Datatype *mpi_datatype);
void starpu_my_data_datatype_free(MPI_Datatype *mpi_datatype);
int starpu_my_data2_datatype_allocate(starpu_data_handle_t handle, MPI_Datatype *mpi_datatype);
void starpu_my_data2_datatype_free(MPI_Datatype *mpi_datatype);

void starpu_my_data_display_codelet_cpu(void *descr[], void *_args);
void starpu_my_data_compare_codelet_cpu(void *descr[], void *_args);

static struct starpu_codelet starpu_my_data_display_codelet =
{
	.cpu_funcs = {starpu_my_data_display_codelet_cpu},
	.cpu_funcs_name = {"starpu_my_data_display_codelet_cpu"},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &starpu_perfmodel_nop,
	.name = "starpu_my_data_display_codelet"
};

static struct starpu_codelet starpu_my_data_compare_codelet =
{
	.cpu_funcs = {starpu_my_data_compare_codelet_cpu},
	.cpu_funcs_name = {"starpu_my_data_compare_codelet_cpu"},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_R},
	.model = &starpu_perfmodel_nop,
	.name = "starpu_my_data_compare_codelet"
};

void starpu_my_data_shutdown(void);
void starpu_my_data2_shutdown(void);

#endif /* __MY_INTERFACE_H */
