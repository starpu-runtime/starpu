/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdlib.h>
#include <starpu_config.h>
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>

#ifdef STARPU_USE_MPI_MPI

#include <mpi/starpu_mpi_mpi_backend.h>
#include <mpi/starpu_mpi_tag.h>
#include <mpi/starpu_mpi_comm.h>
#include <mpi/starpu_mpi_comm.h>
#include <mpi/starpu_mpi_tag.h>
#include <mpi/starpu_mpi_driver.h>
#include <mpi/starpu_mpi_mpi.h>

static void starpu_mpi_mpi_backend_constructor(void) __attribute__((constructor));
static void starpu_mpi_mpi_backend_constructor(void)
{
#ifdef HAVE_PIOMAN
	/* We don't want progression in both PIOman and StarPU */
	setenv("PIOM_ENABLE_PROGRESSION", "0", 0);
#endif
}

void _starpu_mpi_mpi_backend_init(struct starpu_conf *conf)
{
	_starpu_mpi_driver_init(conf);
}

void _starpu_mpi_mpi_backend_shutdown(void)
{
	_starpu_mpi_tag_shutdown();
	_starpu_mpi_comm_shutdown();
	_starpu_mpi_driver_shutdown();
}

int _starpu_mpi_mpi_backend_reserve_core(void)
{
	return (starpu_getenv_number_default("STARPU_MPI_DRIVER_CALL_FREQUENCY", 0) <= 0);
}

void _starpu_mpi_mpi_backend_request_init(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_CALLOC(req->backend, 1, sizeof(struct _starpu_mpi_req_backend));

	//req->backend->data_request = 0;

	STARPU_PTHREAD_MUTEX_INIT0(&req->backend->req_mutex, NULL);
	STARPU_PTHREAD_COND_INIT0(&req->backend->req_cond, NULL);
	STARPU_PTHREAD_COND_INIT0(&req->backend->posted_cond, NULL);

	//req->backend->other_request = NULL;

	//req->backend->size_req = 0;
	//req->backend->internal_req = NULL;
	//req->backend->is_internal_req = 0;
	req->backend->to_destroy = 1;
	//req->backend->early_data_handle = NULL;
	//req->backend->envelope = NULL;
}

void _starpu_mpi_mpi_backend_request_fill(struct _starpu_mpi_req *req, int is_internal_req)
{
	_starpu_mpi_comm_register(req->node_tag.node.comm);
	req->backend->is_internal_req = is_internal_req;
	/* For internal requests, we wait for both the request completion and the matching application request completion */
	req->backend->to_destroy = !is_internal_req;
}

void _starpu_mpi_mpi_backend_request_destroy(struct _starpu_mpi_req *req)
{
	STARPU_PTHREAD_MUTEX_DESTROY(&req->backend->req_mutex);
	STARPU_PTHREAD_COND_DESTROY(&req->backend->req_cond);
	STARPU_PTHREAD_COND_DESTROY(&req->backend->posted_cond);
	free(req->backend);
	req->backend = NULL;
}

void _starpu_mpi_mpi_backend_data_clear(starpu_data_handle_t data_handle)
{
	_starpu_mpi_tag_data_release(data_handle);
}

void _starpu_mpi_mpi_backend_data_register(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag)
{
	_starpu_mpi_tag_data_register(data_handle, data_tag);
}

void _starpu_mpi_mpi_backend_comm_register(MPI_Comm comm)
{
	_starpu_mpi_comm_register(comm);
}

struct _starpu_mpi_backend _mpi_backend =
{
	._starpu_mpi_backend_init = _starpu_mpi_mpi_backend_init,
	._starpu_mpi_backend_shutdown = _starpu_mpi_mpi_backend_shutdown,
	._starpu_mpi_backend_reserve_core = _starpu_mpi_mpi_backend_reserve_core,
	._starpu_mpi_backend_request_init = _starpu_mpi_mpi_backend_request_init,
	._starpu_mpi_backend_request_fill = _starpu_mpi_mpi_backend_request_fill,
	._starpu_mpi_backend_request_destroy = _starpu_mpi_mpi_backend_request_destroy,
	._starpu_mpi_backend_data_clear = _starpu_mpi_mpi_backend_data_clear,
	._starpu_mpi_backend_data_register = _starpu_mpi_mpi_backend_data_register,
	._starpu_mpi_backend_comm_register = _starpu_mpi_mpi_backend_comm_register,

	._starpu_mpi_backend_progress_init = _starpu_mpi_progress_init,
	._starpu_mpi_backend_progress_shutdown = _starpu_mpi_progress_shutdown,
#ifdef STARPU_SIMGRID
	._starpu_mpi_backend_wait_for_initialization = _starpu_mpi_wait_for_initialization,
#endif

	._starpu_mpi_backend_barrier = _starpu_mpi_barrier,
	._starpu_mpi_backend_wait_for_all = _starpu_mpi_wait_for_all,
	._starpu_mpi_backend_wait = _starpu_mpi_wait,
	._starpu_mpi_backend_test = _starpu_mpi_test,

	._starpu_mpi_backend_isend_size_func = _starpu_mpi_isend_size_func,
	._starpu_mpi_backend_irecv_size_func = _starpu_mpi_irecv_size_func,
};

#endif /* STARPU_USE_MPI_MPI*/
