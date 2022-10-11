/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_NMAD_H__
#define __STARPU_MPI_NMAD_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>
#include <common/config.h>
#include <common/list.h>

/** @file */

#ifdef STARPU_USE_MPI_NMAD

#ifdef __cplusplus
extern "C"
{
#endif

int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv);
void _starpu_mpi_progress_shutdown(void **value);

//#ifdef STARPU_SIMGRID
//void _starpu_mpi_wait_for_initialization();
//#endif

int _starpu_mpi_barrier(MPI_Comm comm);
int _starpu_mpi_wait_for_all(MPI_Comm comm);
int _starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status);
int _starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status);

void _starpu_mpi_isend_func(struct _starpu_mpi_req *req);
void _starpu_mpi_irecv_func(struct _starpu_mpi_req *req);
void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req);
void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req);
void _starpu_mpi_handle_received_data(struct _starpu_mpi_req* req);

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_MPI_NMAD */
#endif /* __STARPU_MPI_NMAD_H__ */
