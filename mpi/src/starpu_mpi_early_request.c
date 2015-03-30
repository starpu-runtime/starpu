/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2014  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  CNRS
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
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_early_request.h>
#include <common/uthash.h>

/** stores application requests for which data have not been received yet */
struct _starpu_mpi_req *_starpu_mpi_early_request_hash;
int _starpu_mpi_early_request_hash_count;

void _starpu_mpi_early_request_init()
{
	_starpu_mpi_early_request_hash = NULL;
	_starpu_mpi_early_request_hash_count = 0;
}

void _starpu_mpi_early_request_free()
{
	free(_starpu_mpi_early_request_hash);
}

int _starpu_mpi_early_request_count()
{
	return _starpu_mpi_early_request_hash_count;
}

void _starpu_mpi_early_request_check_termination()
{
	STARPU_ASSERT_MSG(_starpu_mpi_early_request_count() == 0, "Number of early requests left is not zero");
}

struct _starpu_mpi_req* _starpu_mpi_early_request_find(int data_tag, int source, MPI_Comm comm)
{
	struct _starpu_mpi_node_tag node_tag;
	struct _starpu_mpi_req *found;

	memset(&node_tag, 0, sizeof(struct _starpu_mpi_node_tag));
	node_tag.comm = comm;
	node_tag.rank = source;
	node_tag.data_tag = data_tag;

	HASH_FIND(hh, _starpu_mpi_early_request_hash, &node_tag, sizeof(struct _starpu_mpi_node_tag), found);

	return found;
}

void _starpu_mpi_early_request_add(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_req *test_req;

	test_req = _starpu_mpi_early_request_find(req->node_tag.data_tag, req->node_tag.rank, req->node_tag.comm);

	if (test_req == NULL)
	{
		HASH_ADD(hh, _starpu_mpi_early_request_hash, node_tag, sizeof(req->node_tag), req);
		_starpu_mpi_early_request_hash_count ++;
		_STARPU_MPI_DEBUG(3, "Adding request %p with comm %p source %d tag %d in the application request hashmap\n", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "[Error] request %p with comm %p source %d tag %d already in the application request hashmap\n", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag);
		int seq_const = starpu_data_get_sequential_consistency_flag(req->data_handle);
		if (seq_const &&  req->sequential_consistency)
		{
			STARPU_ASSERT_MSG(!test_req, "[Error] request %p with comm %p source %d tag %d wanted to be added to the application request hashmap, while another request %p with the same tag is already in it. \n Sequential consistency is activated : this is not supported by StarPU.", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag, test_req);
		}
		else
		{
			STARPU_ASSERT_MSG(!test_req, "[Error] request %p with comm %p source %d tag %d wanted to be added to the application request hashmap, while another request %p with the same tag is already in it. \n Sequential consistency isn't activated for this handle : you should want to add dependencies between requests for which the sequential consistency is deactivated.", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag, test_req);
		}
	}
}

void _starpu_mpi_early_request_delete(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_req *test_req;

	test_req = _starpu_mpi_early_request_find(req->node_tag.data_tag, req->node_tag.rank, req->node_tag.comm);

	if (test_req != NULL)
	{
		HASH_DEL(_starpu_mpi_early_request_hash, req);
		_starpu_mpi_early_request_hash_count --;
		_STARPU_MPI_DEBUG(3, "Deleting application request %p with comm %p source %d tag %d from the application request hashmap\n", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "[Warning] request %p with comm %p source %d tag %d is NOT in the application request hashmap\n", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag);
	}
}

