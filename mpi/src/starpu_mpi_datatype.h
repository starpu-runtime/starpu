/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_DATATYPE_H__
#define __STARPU_MPI_DATATYPE_H__

#include <starpu_mpi.h>
#include <starpu_mpi_private.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

void _starpu_mpi_datatype_init(void);
void _starpu_mpi_datatype_shutdown(void);

void _starpu_mpi_datatype_allocate(starpu_data_handle_t data_handle, struct _starpu_mpi_req *req);
void _starpu_mpi_datatype_free(starpu_data_handle_t data_handle, MPI_Datatype *datatype);

MPI_Datatype _starpu_mpi_datatype_get_user_defined_datatype(starpu_data_handle_t data_handle);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_DATATYPE_H__
