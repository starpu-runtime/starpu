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

#include <stdlib.h>
#include "starpu_mpi_nmad_backend.h"
#include <starpu_mpi_private.h>

#ifdef STARPU_USE_MPI_NMAD

static void starpu_mpi_nmad_backend_constructor(void) __attribute__((constructor));
static void starpu_mpi_nmad_backend_constructor(void)
{
	/* strat_prio is preferred for StarPU instead of default strat_aggreg */
	setenv("NMAD_STRATEGY", "prio", 0 /* do not overwrite user-supplied value, if set */);
	/* prefer rcache on ibverbs */
	setenv("NMAD_IBVERBS_RCACHE", "1", 0);
	/* use pioman dedicated thread */
	setenv("PIOM_DEDICATED", "1", 0);
	/* pioman waits for starpu to place its dedicated thread */
	setenv("PIOM_DEDICATED_WAIT", "1", 0);
}

void _starpu_mpi_nmad_backend_init(struct starpu_conf *conf)
{
	(void)conf;
}

void _starpu_mpi_nmad_backend_shutdown(void)
{
}

int _starpu_mpi_nmad_backend_reserve_core(void)
{
	return 1;
}

void _starpu_mpi_nmad_backend_request_init(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_CALLOC(req->backend, 1, sizeof(struct _starpu_mpi_req_backend));
	piom_cond_init(&req->backend->req_cond, 0);
}

void _starpu_mpi_nmad_backend_request_fill(struct _starpu_mpi_req *req, MPI_Comm comm, int is_internal_req)
{
	nm_mpi_nmad_dest(&req->backend->session, &req->backend->gate, comm, req->node_tag.node.rank);
}

void _starpu_mpi_nmad_backend_request_destroy(struct _starpu_mpi_req *req)
{
	piom_cond_destroy(&(req->backend->req_cond));
	free(req->backend);
}

void _starpu_mpi_nmad_backend_data_clear(starpu_data_handle_t data_handle)
{
	(void)data_handle;
}

void _starpu_mpi_nmad_backend_data_register(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag)
{
	(void)data_handle;
	(void)data_tag;
}

void _starpu_mpi_nmad_backend_comm_register(MPI_Comm comm)
{
	(void)comm;
}

struct _starpu_mpi_backend _mpi_backend =
{
 	._starpu_mpi_backend_init = _starpu_mpi_nmad_backend_init,
 	._starpu_mpi_backend_shutdown = _starpu_mpi_nmad_backend_shutdown,
	._starpu_mpi_backend_reserve_core = _starpu_mpi_nmad_backend_reserve_core,
	._starpu_mpi_backend_request_init = _starpu_mpi_nmad_backend_request_init,
	._starpu_mpi_backend_request_fill = _starpu_mpi_nmad_backend_request_fill,
	._starpu_mpi_backend_request_destroy = _starpu_mpi_nmad_backend_request_destroy,
	._starpu_mpi_backend_data_clear = _starpu_mpi_nmad_backend_data_clear,
	._starpu_mpi_backend_data_register = _starpu_mpi_nmad_backend_data_register,
	._starpu_mpi_backend_comm_register = _starpu_mpi_nmad_backend_comm_register
};

#endif /* STARPU_USE_MPI_NMAD*/
