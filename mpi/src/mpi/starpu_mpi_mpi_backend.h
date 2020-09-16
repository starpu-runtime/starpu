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

#ifndef __STARPU_MPI_MPI_BACKEND_H__
#define __STARPU_MPI_MPI_BACKEND_H__

#include <common/config.h>
#include <common/uthash.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_USE_MPI_MPI

extern int _starpu_mpi_tag;
#define _STARPU_MPI_TAG_ENVELOPE  _starpu_mpi_tag
#define _STARPU_MPI_TAG_DATA      _starpu_mpi_tag+1
#define _STARPU_MPI_TAG_SYNC_DATA _starpu_mpi_tag+2

enum _starpu_envelope_mode
{
	_STARPU_MPI_ENVELOPE_DATA=0,
	_STARPU_MPI_ENVELOPE_SYNC_READY=1
};

struct _starpu_mpi_envelope
{
	enum _starpu_envelope_mode mode;
	starpu_ssize_t size;
	starpu_mpi_tag_t data_tag;
	unsigned sync;
};

struct _starpu_mpi_req_backend
{
	MPI_Request data_request;

	starpu_pthread_mutex_t req_mutex;
	starpu_pthread_cond_t req_cond;
	starpu_pthread_mutex_t posted_mutex;
	starpu_pthread_cond_t posted_cond;
	/** In the case of a Wait/Test request, we are going to post a request
	 * to test the completion of another request */
	struct _starpu_mpi_req *other_request;

	MPI_Request size_req;

	struct _starpu_mpi_envelope* envelope;

	unsigned is_internal_req:1;
	unsigned to_destroy:1;
	struct _starpu_mpi_req *internal_req;
	struct _starpu_mpi_early_data_handle *early_data_handle;
     	UT_hash_handle hh;
};

#endif // STARPU_USE_MPI_MPI

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_MPI_BACKEND_H__
