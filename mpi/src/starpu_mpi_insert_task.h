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

#ifndef __STARPU_MPI_INSERT_TASK_H__
#define __STARPU_MPI_INSERT_TASK_H__

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

void _starpu_mpi_tables_init(MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_INSERT_TASK_H__
