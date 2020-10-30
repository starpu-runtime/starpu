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

#ifndef __STARPU_MPI_SELECT_NODE_H__
#define __STARPU_MPI_SELECT_NODE_H__

#include <mpi.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

#define _STARPU_MPI_NODE_SELECTION_MAX_POLICY 24

void _starpu_mpi_select_node_init();
int _starpu_mpi_select_node(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data, int policy);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_SELECT_NODE_H__
