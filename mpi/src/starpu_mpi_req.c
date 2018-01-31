/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2013,2016-2017                      Inria
 * Copyright (C) 2009-2018                                Université de Bordeaux
 * Copyright (C) 2017                                     Guillaume Beauchamp
 * Copyright (C) 2010-2017                                CNRS
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

	(*req)->node_tag.rank = -1;
	(*req)->node_tag.data_tag = -1;
	(*req)->node_tag.comm = 0;

	(*req)->func = NULL;

	(*req)->status = NULL;
#ifdef STARPU_USE_MPI_MPI
	(*req)->data_request = 0;
#endif
	(*req)->flag = NULL;

	(*req)->ret = -1;
#ifdef STARPU_USE_MPI_NMAD
	piom_cond_init(&((*req)->req_cond), 0);
#elif defined(STARPU_USE_MPI_MPI)
	STARPU_PTHREAD_MUTEX_INIT(&((*req)->req_mutex), NULL);
	STARPU_PTHREAD_COND_INIT(&((*req)->req_cond), NULL);
	STARPU_PTHREAD_MUTEX_INIT(&((*req)->posted_mutex), NULL);
	STARPU_PTHREAD_COND_INIT(&((*req)->posted_cond), NULL);
#endif

	(*req)->request_type = UNKNOWN_REQ;

	(*req)->submitted = 0;
	(*req)->completed = 0;
	(*req)->posted = 0;

#ifdef STARPU_USE_MPI_MPI
	(*req)->other_request = NULL;
#endif

	(*req)->sync = 0;
	(*req)->detached = -1;
	(*req)->callback = NULL;
	(*req)->callback_arg = NULL;

#ifdef STARPU_USE_MPI_MPI
	(*req)->size_req = 0;
	(*req)->internal_req = NULL;
	(*req)->is_internal_req = 0;
	(*req)->to_destroy = 1;
	(*req)->early_data_handle = NULL;
	(*req)->envelope = NULL;
#endif
	(*req)->sequential_consistency = 1;
	(*req)->pre_sync_jobid = -1;
	(*req)->post_sync_jobid = -1;

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_init(&((*req)->queue));
	starpu_pthread_queue_register(&wait, &((*req)->queue));
	(*req)->done = 0;
#endif
}

void _starpu_mpi_request_destroy(struct _starpu_mpi_req *req)
{
#ifdef STARPU_USE_MPI_NMAD
	piom_cond_destroy(&(req->req_cond));
#elif defined(STARPU_USE_MPI_MPI)
	STARPU_PTHREAD_MUTEX_DESTROY(&req->req_mutex);
	STARPU_PTHREAD_COND_DESTROY(&req->req_cond);
	STARPU_PTHREAD_MUTEX_DESTROY(&req->posted_mutex);
	STARPU_PTHREAD_COND_DESTROY(&req->posted_cond);
	free(req->datatype_name);
	req->datatype_name = NULL;
#endif
#ifdef STARPU_SIMGRID
	starpu_pthread_queue_unregister(&wait, &req->queue);
	starpu_pthread_queue_destroy(&req->queue);
#endif
	free(req);
}

