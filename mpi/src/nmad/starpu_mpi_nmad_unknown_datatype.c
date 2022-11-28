/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "starpu_mpi_nmad.h"
#include "starpu_mpi_nmad_backend.h"
#include "starpu_mpi_nmad_unknown_datatype.h"



/**********************************************
* Send
**********************************************/
void _starpu_mpi_isend_prepare_unknown_datatype(struct _starpu_mpi_req* req, struct nm_data_s* data)
{
	STARPU_ASSERT_MSG(req->registered_datatype != 1, "Datatype is registered, no need to send it through this way !");

	starpu_data_pack_node(req->data_handle, req->node, &req->ptr, &req->count);

	req->backend->unknown_datatype_v[0].iov_base = &req->count;
	req->backend->unknown_datatype_v[0].iov_len = sizeof(starpu_ssize_t);
	req->backend->unknown_datatype_v[1].iov_base = req->ptr;
	req->backend->unknown_datatype_v[1].iov_len = req->count;
	nm_data_iov_build(data, req->backend->unknown_datatype_v, 2);
}

void _starpu_mpi_isend_unknown_datatype(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(30, "post NM isend (unknown datatype) request %p type %s tag %ld src %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.node.comm, req->node, req->node_tag.node.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag, 0);

	_starpu_mpi_isend_prepare_unknown_datatype(req, &req->backend->unknown_datatype_data);

	nm_sr_send_init(req->backend->session, &req->backend->data_request);
	nm_sr_send_pack_data(req->backend->session, &req->backend->data_request, &req->backend->unknown_datatype_data);
	nm_sr_send_set_priority(req->backend->session, &req->backend->data_request, req->prio);
	nm_sr_send_header(req->backend->session, &req->backend->data_request, sizeof(starpu_ssize_t));

	// this trace event is the start of the communication link:
	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(_STARPU_MPI_FUT_POINT_TO_POINT_SEND, req, req->prio);

	if (req->sync == 0)
	{
		req->ret = nm_sr_send_isend(req->backend->session, &req->backend->data_request, req->backend->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "nm_sr_send_isend returning %d", req->ret);
	}
	else
	{
		req->ret = nm_sr_send_issend(req->backend->session, &req->backend->data_request, req->backend->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "nm_sr_send_issend returning %d", req->ret);
	}

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}


/**********************************************
 * Receive
 **********************************************/

static void _starpu_mpi_unknown_datatype_recv_callback(nm_sr_event_t event, const nm_sr_event_info_t* p_info STARPU_ATTRIBUTE_UNUSED, void* ref)
{
	STARPU_ASSERT_MSG(!((event & NM_SR_EVENT_FINALIZED) && (event & NM_SR_EVENT_RECV_DATA)), "Both events can't be triggered at the same time !");

	struct _starpu_mpi_req* req = (struct _starpu_mpi_req*) ref;
	assert(req->request_type == RECV_REQ);
	assert(req->registered_datatype != 1);

	req->backend->posted = 1; // a network event was triggered for this request, so it was really posted

	if (event & NM_SR_EVENT_RECV_DATA)
	{
		// Header arrived, so get the size of the datatype and store it in req->count:
		struct nm_data_s data_header;
		nm_data_contiguous_build(&data_header, &req->count, sizeof(starpu_ssize_t));
		nm_sr_recv_peek(req->backend->session, &req->backend->data_request, &data_header);

		// Now we know the size, allocate the buffer:
		req->ptr = (void *)starpu_malloc_on_node_flags(req->node, req->count, 0);
		STARPU_ASSERT_MSG(req->ptr, "cannot allocate message of size %ld", req->count);

		/* Last step: give this buffer to NewMadeleine to receive data
		 * We need to use an iov to easily take into account the offset used
		 * during the peek. */
		req->backend->unknown_datatype_v[0].iov_base = &req->count;
		req->backend->unknown_datatype_v[0].iov_len = sizeof(starpu_ssize_t);
		req->backend->unknown_datatype_v[1].iov_base = req->ptr;
		req->backend->unknown_datatype_v[1].iov_len = req->count;
		nm_data_iov_build(&req->backend->unknown_datatype_data, req->backend->unknown_datatype_v, 2);
		nm_sr_recv_offset(req->backend->session, &req->backend->data_request, sizeof(starpu_ssize_t));
		nm_sr_recv_unpack_data(req->backend->session, &req->backend->data_request, &req->backend->unknown_datatype_data);
	}
	else if (event & NM_SR_EVENT_FINALIZED)
	{
		_starpu_mpi_handle_request_termination(req);
	}
	else if (event & NM_SR_EVENT_RECV_COMPLETED && !_starpu_mpi_recv_wait_finalize && req->sequential_consistency && starpu_data_get_interface_ops(req->data_handle)->peek_data)
	{
		_starpu_mpi_handle_received_data(req);
	}
}

void _starpu_mpi_irecv_unknown_datatype(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	STARPU_ASSERT_MSG(req->registered_datatype != 1, "Datatype is registered, no need to receive it through this way !");

	_STARPU_MPI_DEBUG(20, "post NM irecv (datatype unknown) request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	/* we post a recv without giving a buffer because we don't know the required size of this buffer,
	 * the buffer will be allocated and provided to nmad when the header of data will be received,
	 * in _starpu_mpi_unknown_datatype_recv_callback() */
	nm_sr_recv_init(req->backend->session, &req->backend->data_request);
	nm_sr_request_set_ref(&req->backend->data_request, req);
	nm_sr_request_monitor(req->backend->session, &req->backend->data_request,
						  NM_SR_EVENT_FINALIZED | NM_SR_EVENT_RECV_DATA | NM_SR_EVENT_RECV_COMPLETED,
						  &_starpu_mpi_unknown_datatype_recv_callback);
	nm_sr_recv_irecv(req->backend->session, &req->backend->data_request, req->backend->gate, req->node_tag.data_tag, NM_TAG_MASK_FULL);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

	_STARPU_MPI_LOG_OUT();
}

#endif //  STARPU_USE_MPI_NMAD
