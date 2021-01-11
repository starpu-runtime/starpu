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
#include <starpu_mpi_early_data.h>
#include <starpu_mpi_private.h>
#include <common/uthash.h>

struct _starpu_mpi_early_data_handle_hashlist
{
	struct _starpu_mpi_early_data_handle_list list;
	UT_hash_handle hh;
	struct _starpu_mpi_node_tag node_tag;
};

/** stores data which have been received by MPI but have not been requested by the application */
static starpu_pthread_mutex_t _starpu_mpi_early_data_handle_mutex;
static struct _starpu_mpi_early_data_handle_hashlist *_starpu_mpi_early_data_handle_hashmap = NULL;
static int _starpu_mpi_early_data_handle_hashmap_count = 0;

void _starpu_mpi_early_data_init(void)
{
	_starpu_mpi_early_data_handle_hashmap = NULL;
	_starpu_mpi_early_data_handle_hashmap_count = 0;
	STARPU_PTHREAD_MUTEX_INIT(&_starpu_mpi_early_data_handle_mutex, NULL);
}

void _starpu_mpi_early_data_check_termination(void)
{
	if (_starpu_mpi_early_data_handle_hashmap_count != 0)
	{
		struct _starpu_mpi_early_data_handle_hashlist *current, *tmp;
		HASH_ITER(hh, _starpu_mpi_early_data_handle_hashmap, current, tmp)
		{
			_STARPU_MSG("Unexpected message with comm %ld source %d tag %ld\n", (long int)current->node_tag.comm, current->node_tag.rank, current->node_tag.data_tag);
		}
		STARPU_ASSERT_MSG(_starpu_mpi_early_data_handle_hashmap_count == 0, "Number of unexpected received messages left is not 0 (but %d), did you forget to post a receive corresponding to a send?", _starpu_mpi_early_data_handle_hashmap_count);
	}
}

void _starpu_mpi_early_data_free(void)
{
	struct _starpu_mpi_early_data_handle_hashlist *current, *tmp;
	HASH_ITER(hh, _starpu_mpi_early_data_handle_hashmap, current, tmp)
	{
		STARPU_ASSERT(_starpu_mpi_early_data_handle_list_empty(&current->list));
		HASH_DEL(_starpu_mpi_early_data_handle_hashmap, current);
		free(current);
	}
	STARPU_PTHREAD_MUTEX_DESTROY(&_starpu_mpi_early_data_handle_mutex);
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_create(struct _starpu_mpi_envelope *envelope, int source, MPI_Comm comm)
{
	struct _starpu_mpi_early_data_handle *early_data_handle;
	_STARPU_MPI_CALLOC(early_data_handle, 1, sizeof(struct _starpu_mpi_early_data_handle));
	STARPU_PTHREAD_MUTEX_INIT(&early_data_handle->req_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&early_data_handle->req_cond, NULL);
	early_data_handle->env = envelope;
	early_data_handle->node_tag.comm = comm;
	early_data_handle->node_tag.rank = source;
	early_data_handle->node_tag.data_tag = envelope->data_tag;
	return early_data_handle;
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_find(struct _starpu_mpi_node_tag *node_tag)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	struct _starpu_mpi_early_data_handle *early_data_handle;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Looking for early_data_handle with comm %d source %d tag %d\n", node_tag->comm, node_tag->rank, node_tag->data_tag);
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		early_data_handle = NULL;
	}
	else
	{
		if (_starpu_mpi_early_data_handle_list_empty(&hashlist->list))
		{
			early_data_handle = NULL;
		}
		else
		{
			_starpu_mpi_early_data_handle_hashmap_count --;
			early_data_handle = _starpu_mpi_early_data_handle_list_pop_front(&hashlist->list);
		}
	}
	_STARPU_MPI_DEBUG(60, "Found early_data_handle %p with comm %d source %d tag %d\n", early_data_handle, node_tag->comm, node_tag->rank, node_tag->data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
	return early_data_handle;
}

void _starpu_mpi_early_data_add(struct _starpu_mpi_early_data_handle *early_data_handle)
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Trying to add early_data_handle %p with comm %d source %d tag %d\n", early_data_handle, early_data_handle->node_tag.comm,
			  early_data_handle->node_tag.rank, early_data_handle->node_tag.data_tag);

	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, &early_data_handle->node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		_STARPU_MPI_MALLOC(hashlist, sizeof(struct _starpu_mpi_early_data_handle_hashlist));
		_starpu_mpi_early_data_handle_list_init(&hashlist->list);
		hashlist->node_tag = early_data_handle->node_tag;
		HASH_ADD(hh, _starpu_mpi_early_data_handle_hashmap, node_tag, sizeof(hashlist->node_tag), hashlist);
	}
	_starpu_mpi_early_data_handle_list_push_back(&hashlist->list, early_data_handle);
	_starpu_mpi_early_data_handle_hashmap_count ++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
}

