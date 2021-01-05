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

#ifndef __STARPU_MPI_COMM_H__
#define __STARPU_MPI_COMM_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>

#ifdef STARPU_USE_MPI_MPI

#include <mpi/starpu_mpi_mpi_backend.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

void _starpu_mpi_comm_init(MPI_Comm comm);
void _starpu_mpi_comm_shutdown();
void _starpu_mpi_comm_register(MPI_Comm comm);
void _starpu_mpi_comm_post_recv();
int _starpu_mpi_comm_test_recv(MPI_Status *status, struct _starpu_mpi_envelope **envelope, MPI_Comm *comm);
void _starpu_mpi_comm_cancel_recv();

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_MPI_MPI
#endif // __STARPU_MPI_COMM_H__
