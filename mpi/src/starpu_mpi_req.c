/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2017       Guillaume Beauchamp
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

#include <starpu.h>
#include <starpu_mpi_private.h>

void _starpu_mpi_request_init(struct _starpu_mpi_req **req)
{
	_STARPU_MPI_CALLOC(*req, 1, sizeof(struct _starpu_mpi_req));

	/* Initialize the request structure */
	(*req)->data_handle = NULL;
	(*req)->prio = 0;

	(*req)->datatype = 0;
	(*req)->datatype_name = NULL;
	(*req)->ptr = NULL;
	(*req)->count = -1;
	(*req)->registered_datatype = -1;

	(*req)->node_tag.node.rank = -1;
	(*req)->node_tag.data_tag = -1;
	(*req)->node_tag.node.comm = 0;

	(*req)->func = NULL;

	(*req)->status = NULL;
	(*req)->flag = NULL;
	_starpu_mpi_req_multilist_init_coop_sends(*req);

	(*req)->ret = -1;

	(*req)->request_type = UNKNOWN_REQ;

	(*req)->submitted = 0;
	(*req)->completed = 0;
	(*req)->posted = 0;

	(*req)->sync = 0;
	(*req)->detached = -1;
	(*req)->callback = NULL;
	(*req)->callback_arg = NULL;

	(*req)->sequential_consistency = 1;
	(*req)->pre_sync_jobid = -1;
	(*req)->post_sync_jobid = -1;

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_init(&((*req)->queue));
	starpu_pthread_queue_register(&_starpu_mpi_thread_wait, &((*req)->queue));
	(*req)->done = 0;
#endif
	_mpi_backend._starpu_mpi_backend_request_init(*req);
}

struct _starpu_mpi_req *_starpu_mpi_request_fill(starpu_data_handle_t data_handle,
						 int srcdst, starpu_mpi_tag_t data_tag, MPI_Comm comm,
						 unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg,
						 enum _starpu_mpi_request_type request_type, void (*func)(struct _starpu_mpi_req *),
						 int sequential_consistency,
						 int is_internal_req,
						 starpu_ssize_t count)
{
	struct _starpu_mpi_req *req;

	/* Initialize the request structure */
	_starpu_mpi_request_init(&req);
	req->request_type = request_type;
	/* prio_list is sorted by increasing values */
	if (_starpu_mpi_use_prio)
		req->prio = prio;
	req->data_handle = data_handle;
	req->node_tag.node.rank = srcdst;
	req->node_tag.data_tag = data_tag;
	req->node_tag.node.comm = comm;
	req->detached = detached;
	req->sync = sync;
	req->callback = callback;
	req->callback_arg = arg;
	req->func = func;
	req->sequential_consistency = sequential_consistency;
	req->count = count;

	_mpi_backend._starpu_mpi_backend_request_fill(req, comm, is_internal_req);

	return req;
}

void _starpu_mpi_request_destroy(struct _starpu_mpi_req *req)
{
	_mpi_backend._starpu_mpi_backend_request_destroy(req);
	free(req->datatype_name);
	req->datatype_name = NULL;
#ifdef STARPU_SIMGRID
	starpu_pthread_queue_unregister(&_starpu_mpi_thread_wait, &req->queue);
	starpu_pthread_queue_destroy(&req->queue);
#endif
	free(req);
}

