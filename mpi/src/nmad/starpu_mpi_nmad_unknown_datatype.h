/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_NMAD_UNKNOWN_DATATYPE_H__
#define __STARPU_MPI_NMAD_UNKNOWN_DATATYPE_H__

#include <common/config.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_USE_MPI_NMAD

#include <nm_sendrecv_interface.h>
#include <nm_mpi_nmad.h>


void _starpu_mpi_isend_unknown_datatype(struct _starpu_mpi_req *req);
void _starpu_mpi_irecv_unknown_datatype(struct _starpu_mpi_req *req);


#endif // STARPU_USE_MPI_NMAD

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_NMAD_UNKNOWN_DATATYPE_H__
