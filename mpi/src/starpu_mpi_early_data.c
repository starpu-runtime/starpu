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
#include <starpu_mpi_early_data.h>
#include <starpu_mpi_private.h>
#include <common/uthash.h>

struct _starpu_mpi_early_data_handle_hashlist
{
	struct _starpu_mpi_early_data_handle_list *list;
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
	STARPU_ASSERT_MSG(_starpu_mpi_early_data_handle_hashmap_count == 0, "Number of unexpected received messages left is not zero, did you forget to post a receive corresponding to a send?");
}

void _starpu_mpi_early_data_free(void)
{
	struct _starpu_mpi_early_data_handle_hashlist *current, *tmp;
	HASH_ITER(hh, _starpu_mpi_early_data_handle_hashmap, current, tmp)
	{
		_starpu_mpi_early_data_handle_list_delete(current->list);
		HASH_DEL(_starpu_mpi_early_data_handle_hashmap, current);
		free(current);
	}
	STARPU_PTHREAD_MUTEX_DESTROY(&_starpu_mpi_early_data_handle_mutex);
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_create(struct _starpu_mpi_envelope *envelope, int source, MPI_Comm comm)
{
	struct _starpu_mpi_early_data_handle* early_data_handle = calloc(1, sizeof(struct _starpu_mpi_early_data_handle));
	STARPU_ASSERT(early_data_handle);
	STARPU_PTHREAD_MUTEX_INIT(&early_data_handle->req_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&early_data_handle->req_cond, NULL);
	early_data_handle->env = envelope;
	early_data_handle->node_tag.comm = comm;
	early_data_handle->node_tag.rank = source;
	early_data_handle->node_tag.data_tag = envelope->data_tag;
	return early_data_handle;
}

#ifdef STARPU_VERBOSE
static void _starpu_mpi_early_data_handle_display_hash(struct _starpu_mpi_node_tag *node_tag)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;

	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		_STARPU_MPI_DEBUG(60, "Hashlist for comm %p source %d and tag %d does not exist\n", node_tag->comm, node_tag->rank, node_tag->data_tag);
	}
	else if (_starpu_mpi_early_data_handle_list_empty(hashlist->list))
	{
		_STARPU_MPI_DEBUG(60, "Hashlist for comm %p source %d and tag %d is empty\n", node_tag->comm, node_tag->rank, node_tag->data_tag);
	}
	else
	{
		struct _starpu_mpi_early_data_handle *cur;
		for (cur = _starpu_mpi_early_data_handle_list_begin(hashlist->list) ;
		     cur != _starpu_mpi_early_data_handle_list_end(hashlist->list);
		     cur = _starpu_mpi_early_data_handle_list_next(cur))
		{
			_STARPU_MPI_DEBUG(60, "Element for comm %p source %d and tag %d: %p\n", node_tag->comm, node_tag->rank, node_tag->data_tag, cur);
		}
	}
}
#endif

static
struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_pop(struct _starpu_mpi_node_tag *node_tag, int delete)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	struct _starpu_mpi_early_data_handle *early_data_handle;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Looking for early_data_handle with comm %p source %d tag %d\n", node_tag->comm, node_tag->rank, node_tag->data_tag);
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		early_data_handle = NULL;
	}
	else
	{
		if (_starpu_mpi_early_data_handle_list_empty(hashlist->list))
		{
			early_data_handle = NULL;
		}
		else
		{
			if (delete == 1)
			{
				early_data_handle = _starpu_mpi_early_data_handle_list_pop_front(hashlist->list);
			}
			else
			{
				early_data_handle = _starpu_mpi_early_data_handle_list_front(hashlist->list);
			}
		}
	}
	_STARPU_MPI_DEBUG(60, "Found early_data_handle %p with comm %p source %d tag %d\n", early_data_handle, node_tag->comm, node_tag->rank, node_tag->data_tag);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
	return early_data_handle;
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_find(struct _starpu_mpi_node_tag *node_tag)
{
	return _starpu_mpi_early_data_pop(node_tag, 0);
}

void _starpu_mpi_early_data_add(struct _starpu_mpi_early_data_handle *early_data_handle)
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_early_data_handle_mutex);
	_STARPU_MPI_DEBUG(60, "Trying to add early_data_handle %p with comm %p source %d tag %d\n", early_data_handle, early_data_handle->node_tag.comm,
			  early_data_handle->node_tag.rank, early_data_handle->node_tag.data_tag);

	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	HASH_FIND(hh, _starpu_mpi_early_data_handle_hashmap, &early_data_handle->node_tag, sizeof(struct _starpu_mpi_node_tag), hashlist);
	if (hashlist == NULL)
	{
		hashlist = malloc(sizeof(struct _starpu_mpi_early_data_handle_hashlist));
		hashlist->list = _starpu_mpi_early_data_handle_list_new();
		hashlist->node_tag = early_data_handle->node_tag;
		HASH_ADD(hh, _starpu_mpi_early_data_handle_hashmap, node_tag, sizeof(hashlist->node_tag), hashlist);
	}
	_starpu_mpi_early_data_handle_list_push_back(hashlist->list, early_data_handle);
	_starpu_mpi_early_data_handle_hashmap_count ++;
#ifdef STARPU_VERBOSE
	_starpu_mpi_early_data_handle_display_hash(&hashlist->node_tag);
#endif
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_early_data_handle_mutex);
}

void _starpu_mpi_early_data_delete(struct _starpu_mpi_early_data_handle *early_data_handle)
{
	_STARPU_MPI_DEBUG(60, "Trying to delete early_data_handle %p with comm %p source %d tag %d\n", early_data_handle, early_data_handle->node_tag.comm,
			  early_data_handle->node_tag.rank, early_data_handle->node_tag.data_tag);
	struct _starpu_mpi_early_data_handle *found = _starpu_mpi_early_data_pop(&early_data_handle->node_tag, 1);

	STARPU_ASSERT_MSG(found == early_data_handle,
			  "[_starpu_mpi_early_data_delete][error] early_data_handle %p with comm %p source %d tag %d is NOT available\n",
			  early_data_handle, early_data_handle->node_tag.comm, early_data_handle->node_tag.rank, early_data_handle->node_tag.data_tag);

	_starpu_mpi_early_data_handle_hashmap_count --;
#ifdef STARPU_VERBOSE
	_starpu_mpi_early_data_handle_display_hash(&early_data_handle->node_tag);
#endif
}

