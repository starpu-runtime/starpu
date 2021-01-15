/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	/** When datatype is unknown */
	struct nm_data_s unknown_datatype_data; // will contain size of the datatype and data itself
	struct iovec unknown_datatype_v[2];
};

#endif // STARPU_USE_MPI_NMAD

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_NMAD_BACKEND_H__
