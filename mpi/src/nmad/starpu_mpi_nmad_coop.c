/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/config.h>

#ifdef STARPU_USE_MPI_NMAD
#include <starpu_mpi_private.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_datatype.h>
#include <nm_launcher_interface.h>
#include <nm_mcast_interface.h>
#include <nm_mpi_nmad.h>
#include "starpu_mpi_nmad_coop.h"
#include "starpu_mpi_nmad_backend.h"
#include "starpu_mpi_nmad_unknown_datatype.h"


extern void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req);

struct mcast_send {
	nm_mcast_t mcast;
	int* dests;
	int* prios;
	struct _starpu_mpi_req* req;
	struct nm_data_s data;
};

static nm_mcast_service_t mcast_service;

void _starpu_mpi_nmad_coop_init(void)
{
	mcast_service = nm_mcast_init(nm_mpi_comm(MPI_COMM_WORLD));
}

void _starpu_mpi_nmad_end_coop_callback(void* arg)
{
	/* Callback called by the root node of the broadcast, when its job is done;
	 * not by receivers. */
	struct mcast_send* mcast = (struct mcast_send*) arg;

	mcast->req->backend->posted = 1;

	_starpu_mpi_handle_request_termination(mcast->req);

	nm_mcast_send_destroy(&mcast->mcast);
	free(mcast->dests);
	if (_starpu_mpi_use_prio)
	{
		free(mcast->prios);
	}
	free(mcast);
}

void _starpu_mpi_submit_coop_sends(struct _starpu_mpi_coop_sends *coop_sends, int submit_control STARPU_ATTRIBUTE_UNUSED, int submit_data)
{
	if (!submit_data)
	{
		return;
	}

	_STARPU_MPI_LOG_IN();

	unsigned n = coop_sends->n;
	assert(n >= 2);

#if 0 // sure, a tree does not worth it for only two recipients, but if the user wants a broadcast with a chain routing, he really wants only one request to be sent from this node.
	if (n == 2) // a broadcast tree does not worth it for only two recipients
	{
		assert(coop_sends->reqs_array[0]->request_type == SEND_REQ);
		_starpu_mpi_submit_ready_request(coop_sends->reqs_array[0]);

		assert(coop_sends->reqs_array[1]->request_type == SEND_REQ);
		_starpu_mpi_submit_ready_request(coop_sends->reqs_array[1]);
	}
	else
#endif
	{
		starpu_fxt_trace_user_event_string("collective send");

		unsigned i = 0;
		struct _starpu_mpi_req *starpu_req;

		struct mcast_send* mcast = malloc(sizeof(struct mcast_send));
		mcast->dests = malloc(n * sizeof(int));
		if (_starpu_mpi_use_prio)
		{
			mcast->prios = malloc(n * sizeof(int));
		}
		else
		{
			mcast->prios = NULL;
		}

		/* We don't increase the amount of communicated data, because we don't
		 * know which tree type will be executed to do the broadcast, so we
		 * don't know how many data will actually be sent from this node. */
		_starpu_mpi_nb_coop_inc(n);

		for (i = 0; i < n; i++)
		{
			starpu_req = coop_sends->reqs_array[i];

			assert(starpu_req->request_type == SEND_REQ);
			assert(starpu_req->coop_sends_head != NULL);
			mcast->dests[i] = starpu_req->node_tag.node.rank;
			if (_starpu_mpi_use_prio)
			{
				mcast->prios[i] = starpu_req->prio;
			}

			// this trace event is the start of the communication link:
			_STARPU_MPI_TRACE_ISEND_SUBMIT_END(_STARPU_MPI_FUT_COLLECTIVE_SEND, starpu_req, starpu_req->prio);

			// Keep the first request to do the mcast, but consider other as finished:
			if (i > 0)
			{
				_starpu_mpi_handle_request_termination(starpu_req);
			}
		}

		starpu_req = coop_sends->reqs_array[0];

		_starpu_mpi_datatype_allocate(starpu_req->data_handle, starpu_req);

		nm_len_t header_len = 0;

		if (starpu_req->registered_datatype == 1)
		{
			starpu_req->count = 1;
			starpu_req->ptr = starpu_data_handle_to_pointer(starpu_req->data_handle, STARPU_MAIN_RAM);
			nm_mpi_nmad_data_get(&mcast->data, (void*)starpu_req->ptr, starpu_req->datatype, starpu_req->count);
		}
		else
		{
			_starpu_mpi_isend_prepare_unknown_datatype(starpu_req, &mcast->data);
			header_len = sizeof(starpu_ssize_t); // we send the size of the data as a header
		}

		mcast->req = starpu_req;

		nm_comm_t comm = nm_comm_get_by_session(starpu_req->backend->session);
		assert(comm != NULL);

		nm_mcast_send_init(mcast_service, &mcast->mcast);
		nm_mcast_send_set_notifier(&mcast->mcast, _starpu_mpi_nmad_end_coop_callback, mcast);
		nm_mcast_isend(&mcast->mcast, comm, mcast->dests, mcast->prios, n, starpu_req->node_tag.data_tag, &mcast->data, header_len, NM_COLL_TREE_DEFAULT);
	}

	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_nmad_coop_shutdown(void)
{
	nm_mcast_finalize(mcast_service);
}

void _starpu_mpi_coop_sends_build_tree(struct _starpu_mpi_coop_sends *coop_sends STARPU_ATTRIBUTE_UNUSED)
{
	/* The NMAD implementation doesn't use this function. */
}

#endif /* STARPU_USE_MPI_NMAD*/
