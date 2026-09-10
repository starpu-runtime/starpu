/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "starpu_mpi_nmad.h"

#ifdef STARPU_USE_MPI_NMAD

#include <nm_public.h>

static void starpu_mpi_nmad_backend_constructor(void) __attribute__((constructor));
static void starpu_mpi_nmad_backend_constructor(void)
{
	/* strat_prio is preferred for StarPU instead of default strat_aggreg */
	setenv("NMAD_STRATEGY", "prio", 0 /* do not overwrite user-supplied value, if set */);
	/* use pioman dedicated thread */
	setenv("PIOM_DEDICATED", "1", 0);
	/* pioman waits for starpu to place its dedicated thread */
	setenv("PIOM_DEDICATED_WAIT", "1", 0);
}

static void _starpu_mpi_nmad_backend_init(struct starpu_conf *conf)
{
	(void)conf;
	nm_abi_config_check();
}

static void _starpu_mpi_nmad_backend_shutdown(void)
{
}

static int _starpu_mpi_nmad_backend_reserve_core(void)
{
	return 1;
}

static void _starpu_mpi_nmad_backend_request_init(struct _starpu_mpi_req *req)
{
	STARPU_MPI_ASSERT_MSG(req->backend == NULL, "MPI request backend already initialized");
	_STARPU_MPI_CALLOC(req->backend, 1, sizeof(struct _starpu_mpi_req_backend));
	piom_cond_init(&req->backend->req_cond, 0);
	req->backend->data_request = NM_SR_REQUEST_NULL;
	req->backend->posted = 0;
	req->backend->has_received_data = 0;
	req->backend->finalized = 0;
	req->backend->to_destroy = 0;
	_starpu_spin_init(&req->backend->finalized_to_destroy_lock);
	nm_datav_init(&req->backend->datav);
}

static void _starpu_mpi_nmad_backend_request_fill(struct _starpu_mpi_req *req, int is_internal_req STARPU_ATTRIBUTE_UNUSED, starpu_mpi_comm internal_comm STARPU_ATTRIBUTE_UNUSED)
{
	/* this function gives session and gate: */
	nm_mpi_nmad_dest(&req->backend->session, &req->backend->gate, req->node_tag.node.comm, req->node_tag.node.rank);
}

static void _starpu_mpi_nmad_backend_request_destroy(struct _starpu_mpi_req *req)
{
	piom_cond_destroy(&(req->backend->req_cond));
	_starpu_spin_destroy(&req->backend->finalized_to_destroy_lock);
	nm_datav_destroy(&req->backend->datav);
	free(req->backend);
	req->backend = NULL;
}

static void _starpu_mpi_nmad_backend_data_clear(starpu_data_handle_t data_handle)
{
	(void)data_handle;
}

static void _starpu_mpi_nmad_backend_data_register(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag)
{
	(void)data_handle;
	(void)data_tag;
}

static void _starpu_mpi_nmad_backend_comm_register(MPI_Comm comm)
{
	(void)comm;
}

static void _starpu_mpi_nmad_early_mem_reg(struct _starpu_mpi_req *req)
{
	STARPU_ASSERT(req->request_type == SEND_REQ);
	_starpu_mpi_init_nmad_send_req(req);
	_STARPU_MPI_DEBUG(21, "triggering NIC memory registration from soon callback\n");
	/* "memory registration" is often called "prefetch" in NewMadeleine */
	nm_sr_send_early_prefetch(req->backend->session, &req->backend->data_request);
}

static void _starpu_mpi_nmad_early_mem_unreg(struct _starpu_mpi_req *req)
{
	STARPU_ASSERT(req->request_type == SEND_REQ);
	_STARPU_MPI_DEBUG(22, "triggering NIC memory unregistration from acquired callback\n");
	nm_sr_send_early_unfetch(req->backend->session, &req->backend->data_request);
}

static void _starpu_mpi_nmad_send_notify_receiver(struct _starpu_mpi_req *req)
{
	STARPU_ASSERT(!req->notification_sent);
	_STARPU_MPI_DEBUG(23, "sending a notification\n");
	nm_session_t session = req->backend->session;
	nm_sr_request_t *nm_req = &req->backend->data_request;
	nm_sr_send_header(session, nm_req, sizeof(size_t));
	nm_sr_send_submit(session, nm_req);
	/* The header will be sent eagerly as soon as possible, and thus acts as
	   a notification */
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
	._starpu_mpi_backend_comm_register = _starpu_mpi_nmad_backend_comm_register,

	._starpu_mpi_backend_progress_init = _starpu_mpi_progress_init,
	._starpu_mpi_backend_progress_shutdown = _starpu_mpi_progress_shutdown,
//#ifdef STARPU_SIMGRID
//	._starpu_mpi_backend_wait_for_initialization = _starpu_mpi_wait_for_initialization,
//#endif

	._starpu_mpi_backend_barrier = _starpu_mpi_barrier,
	._starpu_mpi_backend_wait_for_all = _starpu_mpi_wait_for_all,
	._starpu_mpi_backend_wait_for_all_in_ctx = _starpu_mpi_wait_for_all_in_ctx,
	._starpu_mpi_backend_wait = _starpu_mpi_wait,
	._starpu_mpi_backend_test = _starpu_mpi_test,

	._starpu_mpi_backend_isend_size_func = _starpu_mpi_isend_func,
	._starpu_mpi_backend_irecv_size_func = _starpu_mpi_irecv_func,

	._starpu_mpi_backend_early_mem_reg = _starpu_mpi_nmad_early_mem_reg,
	._starpu_mpi_backend_early_mem_unreg = _starpu_mpi_nmad_early_mem_unreg,

	._starpu_mpi_backend_send_notify_receiver = _starpu_mpi_nmad_send_notify_receiver,
};

#endif /* STARPU_USE_MPI_NMAD*/
