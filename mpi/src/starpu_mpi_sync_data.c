/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Centre National de la Recherche Scientifique
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
#include <starpu_mpi_sync_data.h>
#include <starpu_mpi_private.h>
#include <common/uthash.h>

struct _starpu_mpi_sync_data_handle_hashlist
{
	struct _starpu_mpi_sync_data_handle_list *list;
	UT_hash_handle hh;
	int data_tag;
};

/** stores data which have been received by MPI but have not been requested by the application */
static starpu_pthread_mutex_t *_starpu_mpi_sync_data_handle_mutex;
static struct _starpu_mpi_sync_data_handle_hashlist **_starpu_mpi_sync_data_handle_hashmap = NULL;
static int _starpu_mpi_sync_data_handle_hashmap_count = 0;

#ifdef STARPU_VERBOSE
static
void _starpu_mpi_sync_data_handle_display_hash(int source, int tag)
{
	struct _starpu_mpi_sync_data_handle_hashlist *hashlist;
	HASH_FIND_INT(_starpu_mpi_sync_data_handle_hashmap[source], &tag, hashlist);

	if (hashlist == NULL)
	{
		_STARPU_MPI_DEBUG(60, "Hashlist for source %d and tag %d does not exist\n", source, tag);
	}
	else if (_starpu_mpi_sync_data_handle_list_empty(hashlist->list))
	{
		_STARPU_MPI_DEBUG(60, "Hashlist for source %d and tag %d is empty\n", source, tag);
	}
	else
	{
		struct _starpu_mpi_sync_data_handle *cur;
		for (cur = _starpu_mpi_sync_data_handle_list_begin(hashlist->list) ;
		     cur != _starpu_mpi_sync_data_handle_list_end(hashlist->list);
		     cur = _starpu_mpi_sync_data_handle_list_next(cur))
		{
			_STARPU_MPI_DEBUG(60, "Element for source %d and tag %d: %p\n", source, tag, cur);
		}
	}
}
#endif

void _starpu_mpi_sync_data_init(int world_size)
{
	int k;

	_starpu_mpi_sync_data_handle_hashmap = malloc(world_size * sizeof(struct _starpu_mpi_sync_data_handle_hash_list *));
	_starpu_mpi_sync_data_handle_mutex = malloc(world_size * sizeof(starpu_pthread_mutex_t));
	for(k=0 ; k<world_size ; k++)
	{
		_starpu_mpi_sync_data_handle_hashmap[k] = NULL;
		STARPU_PTHREAD_MUTEX_INIT(&_starpu_mpi_sync_data_handle_mutex[k], NULL);
	}
}

void _starpu_mpi_sync_data_check_termination()
{
	STARPU_ASSERT_MSG(_starpu_mpi_sync_data_handle_hashmap_count == 0, "Number of sync received messages left is not zero, did you forget to post a receive corresponding to a send?");
}

void _starpu_mpi_sync_data_free(int world_size)
{
	int n;
	struct _starpu_mpi_sync_data_handle_hashlist *hashlist;

	for(n=0 ; n<world_size; n++)
	{
		for(hashlist=_starpu_mpi_sync_data_handle_hashmap[n]; hashlist != NULL; hashlist=hashlist->hh.next)
		{
			_starpu_mpi_sync_data_handle_list_delete(hashlist->list);
		}
		struct _starpu_mpi_sync_data_handle_hashlist *current, *tmp;
		HASH_ITER(hh, _starpu_mpi_sync_data_handle_hashmap[n], current, tmp)
		{
			HASH_DEL(_starpu_mpi_sync_data_handle_hashmap[n], current);
			free(current);
		}
		STARPU_PTHREAD_MUTEX_DESTROY(&_starpu_mpi_sync_data_handle_mutex[n]);
	}
	free(_starpu_mpi_sync_data_handle_hashmap);
	free(_starpu_mpi_sync_data_handle_mutex);
}

int _starpu_mpi_sync_data_count()
{
	return _starpu_mpi_sync_data_handle_hashmap_count;
}

struct _starpu_mpi_sync_data_handle *_starpu_mpi_sync_data_create(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_sync_data_handle* sync_data_handle = calloc(1, sizeof(struct _starpu_mpi_sync_data_handle));
	STARPU_ASSERT(sync_data_handle);
	sync_data_handle->data_tag = req->data_tag;
	sync_data_handle->source = req->srcdst;
	sync_data_handle->req = req;
	return sync_data_handle;
}

struct _starpu_mpi_sync_data_handle *_starpu_mpi_sync_data_find(int data_tag, int source)
{
	struct _starpu_mpi_sync_data_handle_hashlist *hashlist;
	struct _starpu_mpi_sync_data_handle *sync_data_handle;

	_STARPU_MPI_DEBUG(60, "Looking for sync_data_handle with tag %d in the hashmap[%d]\n", data_tag, source);
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_sync_data_handle_mutex[source]);
	HASH_FIND_INT(_starpu_mpi_sync_data_handle_hashmap[source], &data_tag, hashlist);
	if (hashlist == NULL)
	{
		sync_data_handle = NULL;
	}
	else
	{
		if (_starpu_mpi_sync_data_handle_list_empty(hashlist->list))
		{
			sync_data_handle = NULL;
		}
		else
		{
			sync_data_handle = _starpu_mpi_sync_data_handle_list_pop_front(hashlist->list);
			_starpu_mpi_sync_data_handle_hashmap_count --;
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_sync_data_handle_mutex[source]);
	_STARPU_MPI_DEBUG(60, "Found sync_data_handle %p with tag %d in the hashmap[%d]\n", sync_data_handle, data_tag, source);
	return sync_data_handle;
}

void _starpu_mpi_sync_data_add(struct _starpu_mpi_sync_data_handle *sync_data_handle)
{
	_STARPU_MPI_DEBUG(2000, "Adding sync_data_handle %p with tag %d in the hashmap[%d]\n", sync_data_handle, sync_data_handle->data_tag, sync_data_handle->source);

	struct _starpu_mpi_sync_data_handle_hashlist *hashlist;
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_sync_data_handle_mutex[sync_data_handle->source]);
	HASH_FIND_INT(_starpu_mpi_sync_data_handle_hashmap[sync_data_handle->source], &sync_data_handle->data_tag, hashlist);
	if (hashlist == NULL)
	{
		hashlist = malloc(sizeof(struct _starpu_mpi_sync_data_handle_hashlist));
		hashlist->list = _starpu_mpi_sync_data_handle_list_new();
		hashlist->data_tag = sync_data_handle->data_tag;
		HASH_ADD_INT(_starpu_mpi_sync_data_handle_hashmap[sync_data_handle->source], data_tag, hashlist);
	}
	_starpu_mpi_sync_data_handle_list_push_back(hashlist->list, sync_data_handle);
	_starpu_mpi_sync_data_handle_hashmap_count ++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_sync_data_handle_mutex[sync_data_handle->source]);
#ifdef STARPU_VERBOSE
	_starpu_mpi_sync_data_handle_display_hash(sync_data_handle->source, sync_data_handle->data_tag);
#endif
}

