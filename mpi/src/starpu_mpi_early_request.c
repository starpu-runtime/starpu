/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
struct _starpu_mpi_early_request_hashlist
{
	struct _starpu_mpi_req_list list;
	UT_hash_handle hh;
	struct _starpu_mpi_node_tag node_tag;
};

static starpu_pthread_mutex_t _starpu_mpi_early_request_mutex;
struct _starpu_mpi_early_request_hashlist *_starpu_mpi_early_request_hash;
int _starpu_mpi_early_request_hash_count;

void _starpu_mpi_early_request_init()
{
	_starpu_mpi_early_request_hash = NULL;
	_starpu_mpi_early_request_hash_count = 0;
	STARPU_PTHREAD_MUTEX_INIT(&_starpu_mpi_early_request_mutex, NULL);
}

void _starpu_mpi_early_request_free()
{
	struct _starpu_mpi_early_request_hashlist *entry, *tmp;
	HASH_ITER(hh, _starpu_mpi_early_request_hash, entry, tmp)
	{
		STARPU_ASSERT(_starpu_mpi_req_list_empty(&entry->list));
		HASH_DEL(_starpu_mpi_early_request_hash, entry);
		free(entry);
	}
	STARPU_PTHREAD_MUTEX_DESTROY(&_starpu_mpi_early_request_mutex);
}

int _starpu_mpi_early_request_count()
{
	return _starpu_mpi_early_request_hash_count;
}

void _starpu_mpi_early_request_check_termination()
{
	STARPU_ASSERT_MSG(_starpu_mpi_early_request_count() == 0, "Number of early requests left is not zero");
}

struct _starpu_mpi_req* _starpu_mpi_early_request_dequeue(int data_tag, int source, MPI_Comm comm)
{
	struct _starpu_mpi_node_tag node_tag;
	struct _starpu_mpi_req *found;
	struct _starpu_mpi_early_request_hashlist *hashlist;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	memset(&node_tag, 0, sizeof(struct _starpu_mpi_node_tag));
	node_tag.comm = comm;
	node_tag.rank = source;
	node_tag.data_tag = data_tag;

	_STARPU_MPI_DEBUG(100, "Looking for early_request with comm %d source %d tag %d\n", node_tag.comm, node_tag.rank, node_tag.data_tag);
	HASH_FIND(hh, _starpu_mpi_early_request_hash, &node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		found = NULL;
	}
	else
	{
		if (_starpu_mpi_req_list_empty(&hashlist->list))
		{
			found = NULL;
		}
		else
		{
			found = _starpu_mpi_req_list_pop_front(&hashlist->list);
			_starpu_mpi_early_request_hash_count --;
		}
	}
	_STARPU_MPI_DEBUG(100, "Found early_request %p with comm %d source %d tag %d\n", found, node_tag.comm, node_tag.rank, node_tag.data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_request_mutex);
	return found;
}

void _starpu_mpi_early_request_enqueue(struct _starpu_mpi_req *req)
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	_STARPU_MPI_DEBUG(100, "Adding request %p with comm %d source %d tag %d in the application request hashmap\n", req, req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag);

	struct _starpu_mpi_early_request_hashlist *hashlist;
	HASH_FIND(hh, _starpu_mpi_early_request_hash, &req->node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		_STARPU_MPI_MALLOC(hashlist, sizeof(struct _starpu_mpi_early_request_hashlist));
		_starpu_mpi_req_list_init(&hashlist->list);
		hashlist->node_tag = req->node_tag;
		HASH_ADD(hh, _starpu_mpi_early_request_hash, node_tag, sizeof(hashlist->node_tag), hashlist);
	}
	_starpu_mpi_req_list_push_back(&hashlist->list, req);
	_starpu_mpi_early_request_hash_count ++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_request_mutex);
}
