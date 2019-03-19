/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2017,2019                           CNRS
 * Copyright (C) 2009-2014,2016,2017                      Université de Bordeaux
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
#include <mpi/starpu_mpi_early_request.h>
#include <common/uthash.h>

#ifdef STARPU_USE_MPI_MPI

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

void _starpu_mpi_early_request_shutdown()
{
	struct _starpu_mpi_early_request_hashlist *entry=NULL, *tmp=NULL;
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

struct _starpu_mpi_req* _starpu_mpi_early_request_dequeue(starpu_mpi_tag_t data_tag, int source, MPI_Comm comm)
{
	struct _starpu_mpi_node_tag node_tag;
	struct _starpu_mpi_req *found;
	struct _starpu_mpi_early_request_hashlist *hashlist;

	memset(&node_tag, 0, sizeof(struct _starpu_mpi_node_tag));
	node_tag.comm = comm;
	node_tag.rank = source;
	node_tag.data_tag = data_tag;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	_STARPU_MPI_DEBUG(100, "Looking for early_request with comm %ld source %d tag %ld\n", (long int)node_tag.comm, node_tag.rank, node_tag.data_tag);
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
	_STARPU_MPI_DEBUG(100, "Found early_request %p with comm %ld source %d tag %ld\n", found, (long int)node_tag.comm, node_tag.rank, node_tag.data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_request_mutex);
	return found;
}

void _starpu_mpi_early_request_enqueue(struct _starpu_mpi_req *req)
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	_STARPU_MPI_DEBUG(100, "Adding request %p with comm %ld source %d tag %ld in the application request hashmap\n", req, (long int)req->node_tag.comm, req->node_tag.rank, req->node_tag.data_tag);

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

#endif // STARPU_USE_MPI_MPI
