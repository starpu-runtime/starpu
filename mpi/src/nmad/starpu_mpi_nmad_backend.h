/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_NMAD_BACKEND_H__
#define __STARPU_MPI_NMAD_BACKEND_H__

#include <common/config.h>
#include <common/starpu_spinlock.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_USE_MPI_NMAD

#include <nm_sendrecv_interface.h>
#include <nm_session_interface.h>
#include <nm_mpi_nmad.h>

struct _starpu_mpi_req_backend
{
	nm_gate_t gate;
	nm_session_t session;
	nm_sr_request_t data_request;
	piom_cond_t req_cond;

	int posted; // with coop, only one request is really posted, we need to know if the request was really posted to possibly free data
	int has_received_data; // tell if request went through _starpu_mpi_handle_received_data() to release write lock
	int finalized; // tell if _starpu_mpi_handle_request_termination() was called, so starpu_mpi_test() and starpu_mpi_wait() have to free the request
	int to_destroy; // tell if starpu_mpi_wait() or starpu_mpi_test() was called before _starpu_mpi_handle_request_termination() and thus this last function will have to free the request
	struct _starpu_spinlock finalized_to_destroy_lock;

	/** When datatype is unknown */
	struct nm_data_s unknown_datatype_data; // will contain size of the datatype and data itself
	struct iovec unknown_datatype_v[2];
};

#endif // STARPU_USE_MPI_NMAD

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_NMAD_BACKEND_H__
