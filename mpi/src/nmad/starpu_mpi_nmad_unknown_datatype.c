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


#include <common/config.h>

#ifdef STARPU_USE_MPI_NMAD
#include <starpu_mpi_private.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_datatype.h>
#include <nm_sendrecv_interface.h>
#include <nm_mpi_nmad.h>
#include "starpu_mpi_nmad_backend.h"
#include "starpu_mpi_nmad_unknown_datatype.h"

#if defined(STARPU_VERBOSE) || defined(STARPU_MPI_VERBOSE)
extern char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type);
#endif

extern void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req,nm_sr_event_t event);
extern void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req);

struct starpu_nm_datatype_unknown
{
	starpu_ssize_t* count;
	const struct nm_data_s* body;
};

static void starpu_nm_datatype_unknown_traversal(const void* _content, nm_data_apply_t apply, void* _context);
const struct nm_data_ops_s starpu_nm_datatype_unknown_ops =
{
	.p_traversal = &starpu_nm_datatype_unknown_traversal
};

NM_DATA_TYPE(datatype_unknown, struct starpu_nm_datatype_unknown, &starpu_nm_datatype_unknown_ops);

static void starpu_nm_datatype_unknown_traversal(const void* _content, nm_data_apply_t apply, void* _context)
{
	const struct starpu_nm_datatype_unknown* p_content = _content;

	(*apply)(p_content->count, sizeof(starpu_ssize_t), _context);

	nm_data_traversal_apply(p_content->body, apply, _context);
}

// warning: this function requires valid pointers for future usage
void starpu_nm_datatype_unknown_build(struct nm_data_s* datatype_unknown_data, starpu_ssize_t* count, const struct nm_data_s* body)
{
	nm_data_datatype_unknown_set(datatype_unknown_data, (struct starpu_nm_datatype_unknown)
			{
			.count = count,
			.body = body
			});
}

/**********************************************
* Send
**********************************************/

void _starpu_mpi_isend_unknown_datatype(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	STARPU_ASSERT_MSG(req->registered_datatype != 1, "Datatype is registered, no need to send it through this way !");

	_STARPU_MPI_DEBUG(30, "post NM isend (unknown datatype) request %p type %s tag %ld src %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.node.comm, req->node_tag.node.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag, 0);

	starpu_data_pack(req->data_handle, &req->ptr, &req->count);

	nm_mpi_nmad_data_get(&(req->backend->unknown_datatype_body), (void*)req->ptr, req->datatype, req->count);

	// warning: this function requires valid pointers for future usage
	starpu_nm_datatype_unknown_build(&(req->backend->unknown_datatype_data), &(req->count), &(req->backend->unknown_datatype_body));

	nm_sr_send_init(req->backend->session, &(req->backend->data_request));
	nm_sr_send_pack_data(req->backend->session, &(req->backend->data_request), &(req->backend->unknown_datatype_data));
	nm_sr_send_set_priority(req->backend->session, &(req->backend->data_request), req->prio);
	nm_sr_send_header(req->backend->session, &(req->backend->data_request), sizeof(starpu_ssize_t));

	if (req->sync == 0)
	{
		req->ret = nm_sr_send_isend(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "nm_sr_send_isend returning %d", req->ret);
	}
	else
	{
		req->ret = nm_sr_send_issend(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "nm_sr_send_issend returning %d", req->ret);
	}

	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag, starpu_data_get_size(req->data_handle), req->pre_sync_jobid, req->data_handle);

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}


/**********************************************
 * Receive
 **********************************************/

static void _starpu_mpi_unknown_datatype_recv_callback(nm_sr_event_t event, const nm_sr_event_info_t* p_info, void* ref)
{
	STARPU_ASSERT_MSG(!((event & NM_SR_EVENT_FINALIZED) && (event & NM_SR_EVENT_RECV_DATA)), "Both events can't be triggered at the same time !");

	struct _starpu_mpi_req* req = (struct _starpu_mpi_req*) ref;

	if (event & NM_SR_EVENT_RECV_DATA)
	{
		nm_data_contiguous_build(&(req->backend->unknown_datatype_size), &(req->count), sizeof(int));

		int ret = nm_sr_recv_peek(req->backend->session, &(req->backend->data_request), &(req->backend->unknown_datatype_size));
		STARPU_ASSERT_MSG(ret == NM_ESUCCESS, "nm_sr_recv_peek returned %d", ret);

		req->ptr = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, req->count, 0);
		STARPU_ASSERT_MSG(req->ptr, "cannot allocate message of size %ld", req->count);

		nm_mpi_nmad_data_get(&(req->backend->unknown_datatype_body), (void*) req->ptr, req->datatype, req->count);

		// warning: this function requires valid pointers for future usage
		starpu_nm_datatype_unknown_build(&(req->backend->unknown_datatype_data), &(req->count), &(req->backend->unknown_datatype_body));
		nm_sr_recv_unpack_data(req->backend->session, &(req->backend->data_request), &(req->backend->unknown_datatype_data));
	}
	else if (event & NM_SR_EVENT_FINALIZED)
	{
		_starpu_mpi_handle_request_termination(req, event);
	}
}

void _starpu_mpi_irecv_unknown_datatype(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	STARPU_ASSERT_MSG(req->registered_datatype != 1, "Datatype is registered, no need to receive it through this way !");

	_STARPU_MPI_DEBUG(20, "post NM irecv (datatype unknown) request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	nm_sr_recv_init(req->backend->session, &(req->backend->data_request));
	nm_sr_request_set_ref(&(req->backend->data_request), req);
	nm_sr_request_monitor(req->backend->session, &(req->backend->data_request), NM_SR_EVENT_FINALIZED | NM_SR_EVENT_RECV_DATA,
				&_starpu_mpi_unknown_datatype_recv_callback);
	nm_sr_recv_irecv(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag, NM_TAG_MASK_FULL);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

	_STARPU_MPI_LOG_OUT();
}

#endif //  STARPU_USE_MPI_NMAD
