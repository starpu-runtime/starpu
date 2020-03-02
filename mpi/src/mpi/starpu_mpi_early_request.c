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
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <mpi/starpu_mpi_early_request.h>
#include <common/uthash.h>

#ifdef STARPU_USE_MPI_MPI

/** stores application requests for which data have not been received yet */
/** the hashlist is on 2 levels, the first top level is indexed on (node, rank), the second lower level is indexed on the data tag */
struct _starpu_mpi_early_request_hashlist
{
	struct _starpu_mpi_early_request_tag_hashlist *datahash;
	UT_hash_handle hh;
	struct _starpu_mpi_node node;
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
		struct _starpu_mpi_early_request_tag_hashlist *tag_entry=NULL, *tag_tmp=NULL;
		HASH_ITER(hh, entry->datahash, tag_entry, tag_tmp)
		{
			STARPU_ASSERT(_starpu_mpi_req_list_empty(&tag_entry->list));
			HASH_DEL(entry->datahash, tag_entry);
			free(tag_entry);
		}

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
	node_tag.node.comm = comm;
	node_tag.node.rank = source;
	node_tag.data_tag = data_tag;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	_STARPU_MPI_DEBUG(100, "Looking for early_request with comm %ld source %d tag %ld\n", (long int)node_tag.node.comm, node_tag.node.rank, node_tag.data_tag);
	HASH_FIND(hh, _starpu_mpi_early_request_hash, &node_tag.node, sizeof(struct _starpu_mpi_node), hashlist);
	if (hashlist == NULL)
	{
		found = NULL;
	}
	else
	{
		struct _starpu_mpi_early_request_tag_hashlist *tag_hashlist;
		HASH_FIND(hh, hashlist->datahash, &node_tag.data_tag, sizeof(starpu_mpi_tag_t), tag_hashlist);
		if (tag_hashlist == NULL)
		{
			found = NULL;
		}
		else if (_starpu_mpi_req_list_empty(&tag_hashlist->list))
		{
			found = NULL;
		}
		else
		{
			found = _starpu_mpi_req_list_pop_front(&tag_hashlist->list);
			_starpu_mpi_early_request_hash_count --;
		}
	}
	_STARPU_MPI_DEBUG(100, "Found early_request %p with comm %ld source %d tag %ld\n", found, (long int)node_tag.node.comm, node_tag.node.rank, node_tag.data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_request_mutex);
	return found;
}

struct _starpu_mpi_early_request_tag_hashlist *_starpu_mpi_early_request_extract(starpu_mpi_tag_t data_tag, int source, MPI_Comm comm)
{
	struct _starpu_mpi_node_tag node_tag;
	struct _starpu_mpi_early_request_hashlist *hashlist;
	struct _starpu_mpi_early_request_tag_hashlist *tag_hashlist = NULL;

	memset(&node_tag, 0, sizeof(struct _starpu_mpi_node_tag));
	node_tag.node.comm = comm;
	node_tag.node.rank = source;
	node_tag.data_tag = data_tag;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	_STARPU_MPI_DEBUG(100, "Looking for early_request with comm %ld source %d tag %ld\n", (long int)node_tag.node.comm, node_tag.node.rank, node_tag.data_tag);
	HASH_FIND(hh, _starpu_mpi_early_request_hash, &node_tag.node, sizeof(struct _starpu_mpi_node), hashlist);
	if (hashlist)
	{
		HASH_FIND(hh, hashlist->datahash, &node_tag.data_tag, sizeof(starpu_mpi_tag_t), tag_hashlist);
		if (tag_hashlist)
		{
			_starpu_mpi_early_request_hash_count -= _starpu_mpi_req_list_size(&tag_hashlist->list);
			HASH_DEL(hashlist->datahash, tag_hashlist);
		}
	}
	_STARPU_MPI_DEBUG(100, "Found hashlist %p with comm %ld source %d tag %ld\n", hashlist, (long int)node_tag.node.comm, node_tag.node.rank, node_tag.data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_request_mutex);
	return tag_hashlist;
}

void _starpu_mpi_early_request_enqueue(struct _starpu_mpi_req *req)
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_request_mutex);
	_STARPU_MPI_DEBUG(100, "Adding request %p with comm %ld source %d tag %ld in the application request hashmap\n", req, (long int)req->node_tag.node.comm, req->node_tag.node.rank, req->node_tag.data_tag);

	struct _starpu_mpi_early_request_hashlist *hashlist;
	HASH_FIND(hh, _starpu_mpi_early_request_hash, &req->node_tag.node, sizeof(struct _starpu_mpi_node), hashlist);
	if (hashlist == NULL)
	{
		_STARPU_MPI_MALLOC(hashlist, sizeof(struct _starpu_mpi_early_request_hashlist));
		hashlist->node = req->node_tag.node;
		hashlist->datahash = NULL;
		HASH_ADD(hh, _starpu_mpi_early_request_hash, node, sizeof(hashlist->node), hashlist);
	}

	struct _starpu_mpi_early_request_tag_hashlist *tag_hashlist;
	HASH_FIND(hh, hashlist->datahash, &req->node_tag.data_tag, sizeof(starpu_mpi_tag_t), tag_hashlist);
	if (tag_hashlist == NULL)
	{
		_STARPU_MPI_MALLOC(tag_hashlist, sizeof(struct _starpu_mpi_early_request_tag_hashlist));
		tag_hashlist->data_tag = req->node_tag.data_tag;
		HASH_ADD(hh, hashlist->datahash, data_tag, sizeof(tag_hashlist->data_tag), tag_hashlist);
		_starpu_mpi_req_list_init(&tag_hashlist->list);
	}

	_starpu_mpi_req_list_push_back(&tag_hashlist->list, req);
	_starpu_mpi_early_request_hash_count ++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_request_mutex);
}

#endif // STARPU_USE_MPI_MPI
