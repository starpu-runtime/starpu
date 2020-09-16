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

#ifndef __STARPU_MPI_TAG_H__
#define __STARPU_MPI_TAG_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>

/** @file */

#ifdef STARPU_USE_MPI_MPI

#ifdef __cplusplus
extern "C"
{
#endif

void _starpu_mpi_tag_init(void);
void _starpu_mpi_tag_shutdown(void);

void _starpu_mpi_tag_data_register(starpu_data_handle_t handle, starpu_mpi_tag_t data_tag);
int _starpu_mpi_tag_data_release(starpu_data_handle_t handle);
starpu_data_handle_t _starpu_mpi_tag_get_data_handle_from_tag(starpu_mpi_tag_t data_tag);

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_MPI_MPI
#endif // __STARPU_MPI_TAG_H__
