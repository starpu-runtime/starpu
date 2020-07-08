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
#include <mpi/starpu_mpi_early_data.h>
#include <mpi/starpu_mpi_mpi_backend.h>
#include <starpu_mpi_private.h>

#ifdef STARPU_USE_MPI_MPI

/** the hashlist is on 2 levels, the first top level is indexed on (node, rank), the second lower level is indexed on the data tag */

struct _starpu_mpi_early_data_handle_hashlist
{
	struct _starpu_mpi_early_data_handle_tag_hashlist *datahash;
	UT_hash_handle hh;
	struct _starpu_mpi_node node;
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
		struct _starpu_mpi_early_data_handle_hashlist *current=NULL, *tmp=NULL;
		HASH_ITER(hh, _starpu_mpi_early_data_handle_hashmap, current, tmp)
		{
			struct _starpu_mpi_early_data_handle_tag_hashlist *tag_current=NULL, *tag_tmp=NULL;
			HASH_ITER(hh, current->datahash, tag_current, tag_tmp)
			{
				_STARPU_MSG("Unexpected message with comm %ld source %d tag %ld\n", (long int)current->node.comm, current->node.rank, tag_current->data_tag);
			}
		}
		STARPU_ASSERT_MSG(_starpu_mpi_early_data_handle_hashmap_count == 0, "Number of unexpected received messages left is not 0 (but %d), did you forget to post a receive corresponding to a send?", _starpu_mpi_early_data_handle_hashmap_count);
	}
}

void _starpu_mpi_early_data_shutdown(void)
{
	struct _starpu_mpi_early_data_handle_hashlist *current=NULL, *tmp=NULL;
	HASH_ITER(hh, _starpu_mpi_early_data_handle_hashmap, current, tmp)
	{
		_STARPU_MPI_DEBUG(600, "Hash early_data with comm %ld source %d\n", (long int) current->node.comm, current->node.rank);
		struct _starpu_mpi_early_data_handle_tag_hashlist *tag_entry=NULL, *tag_tmp=NULL;
		HASH_ITER(hh, current->datahash, tag_entry, tag_tmp)
		{
			_STARPU_MPI_DEBUG(600, "Hash 2nd level with tag %ld\n", tag_entry->data_tag);
			STARPU_ASSERT(_starpu_mpi_early_data_handle_list_empty(&tag_entry->list));
			HASH_DEL(current->datahash, tag_entry);
			free(tag_entry);
		}
		HASH_DEL(_starpu_mpi_early_data_handle_hashmap, current);
		free(current);
	}
	STARPU_PTHREAD_MUTEX_DESTROY(&_starpu_mpi_early_data_handle_mutex);
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_create(struct _starpu_mpi_envelope *envelope, int source, MPI_Comm comm)
{
	struct _starpu_mpi_early_data_handle* early_data_handle;
	_STARPU_MPI_CALLOC(early_data_handle, 1, sizeof(struct _starpu_mpi_early_data_handle));
	STARPU_PTHREAD_MUTEX_INIT(&early_data_handle->req_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&early_data_handle->req_cond, NULL);
	early_data_handle->node_tag.node.comm = comm;
	early_data_handle->node_tag.node.rank = source;
	early_data_handle->node_tag.data_tag = envelope->data_tag;
	return early_data_handle;
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_find(struct _starpu_mpi_node_tag *node_tag)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	struct _starpu_mpi_early_data_handle *early_data_handle;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Looking for early_data_handle with comm %ld source %d tag %ld\n", (long int)node_tag->node.comm, node_tag->node.rank, node_tag->data_tag);
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, &node_tag->node, sizeof(struct _starpu_mpi_node), hashlist);
	if (hashlist == NULL)
	{
		_STARPU_MPI_DEBUG(600, "No entry for (comm %ld, source %d)\n", (long int)node_tag->node.comm, node_tag->node.rank);
		early_data_handle = NULL;
	}
	else
	{
		struct _starpu_mpi_early_data_handle_tag_hashlist *tag_hashlist;
		HASH_FIND(hh, hashlist->datahash, &node_tag->data_tag, sizeof(starpu_mpi_tag_t), tag_hashlist);
		if (tag_hashlist == NULL)
		{
			_STARPU_MPI_DEBUG(600, "No entry for tag %ld\n", node_tag->data_tag);
			early_data_handle = NULL;
		}
		else if (_starpu_mpi_early_data_handle_list_empty(&tag_hashlist->list))
		{
			_STARPU_MPI_DEBUG(600, "List empty for tag %ld\n", node_tag->data_tag);
			early_data_handle = NULL;
		}
		else
		{
			_starpu_mpi_early_data_handle_hashmap_count --;
			early_data_handle = _starpu_mpi_early_data_handle_list_pop_front(&tag_hashlist->list);
		}
	}
	_STARPU_MPI_DEBUG(60, "Found early_data_handle %p with comm %ld source %d tag %ld\n", early_data_handle, (long int)node_tag->node.comm, node_tag->node.rank, node_tag->data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
	return early_data_handle;
}

struct _starpu_mpi_early_data_handle_tag_hashlist *_starpu_mpi_early_data_extract(struct _starpu_mpi_node_tag *node_tag)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	struct _starpu_mpi_early_data_handle_tag_hashlist *tag_hashlist = NULL;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Looking for hashlist for (comm %ld, source %d)\n", (long int)node_tag->node.comm, node_tag->node.rank);
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, &node_tag->node, sizeof(struct _starpu_mpi_node), hashlist);
	if (hashlist)
	{
		_STARPU_MPI_DEBUG(60, "Looking for hashlist for (tag %ld)\n", node_tag->data_tag);
		HASH_FIND(hh, hashlist->datahash, &node_tag->data_tag, sizeof(starpu_mpi_tag_t), tag_hashlist);
		if (tag_hashlist)
		{
			_starpu_mpi_early_data_handle_hashmap_count -= _starpu_mpi_early_data_handle_list_size(&tag_hashlist->list);
			HASH_DEL(hashlist->datahash, tag_hashlist);
		}
	}
	_STARPU_MPI_DEBUG(60, "Found hashlist %p for (comm %ld, source %d) and (tag %ld)\n", tag_hashlist, (long int)node_tag->node.comm, node_tag->node.rank, node_tag->data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
	return tag_hashlist;
}

void _starpu_mpi_early_data_add(struct _starpu_mpi_early_data_handle *early_data_handle)
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Adding early_data_handle %p with comm %ld source %d tag %ld (%p)\n", early_data_handle, (long int)early_data_handle->node_tag.node.comm, early_data_handle->node_tag.node.rank, early_data_handle->node_tag.data_tag, &early_data_handle->node_tag.node);

	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, &early_data_handle->node_tag.node, sizeof(struct _starpu_mpi_node), hashlist);
	if (hashlist == NULL)
	{
		_STARPU_MPI_MALLOC(hashlist, sizeof(struct _starpu_mpi_early_data_handle_hashlist));
		hashlist->node = early_data_handle->node_tag.node;
		hashlist->datahash = NULL;
		HASH_ADD(hh, _starpu_mpi_early_data_handle_hashmap, node, sizeof(hashlist->node), hashlist);
	}

	struct _starpu_mpi_early_data_handle_tag_hashlist *tag_hashlist;
	HASH_FIND(hh, hashlist->datahash, &early_data_handle->node_tag.data_tag, sizeof(starpu_mpi_tag_t), tag_hashlist);
	if (tag_hashlist == NULL)
	{
		_STARPU_MPI_MALLOC(tag_hashlist, sizeof(struct _starpu_mpi_early_data_handle_tag_hashlist));
		tag_hashlist->data_tag = early_data_handle->node_tag.data_tag;
		HASH_ADD(hh, hashlist->datahash, data_tag, sizeof(tag_hashlist->data_tag), tag_hashlist);
		_starpu_mpi_early_data_handle_list_init(&tag_hashlist->list);
	}

	_starpu_mpi_early_data_handle_list_push_back(&tag_hashlist->list, early_data_handle);
	_starpu_mpi_early_data_handle_hashmap_count ++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
}

#endif // STARPU_USE_MPI_MPI
