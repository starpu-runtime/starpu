/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2014  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
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
	int mpi_tag;
};

/** stores data which have been received by MPI but have not been requested by the application */
static struct _starpu_mpi_early_data_handle_hashlist **_starpu_mpi_early_data_handle_hashmap = NULL;
static int _starpu_mpi_early_data_handle_hashmap_count = 0;

void _starpu_mpi_early_data_init(int world_size)
{
	int k;

	_starpu_mpi_early_data_handle_hashmap = malloc(world_size * sizeof(struct _starpu_mpi_early_data_handle_hash_list *));
	for(k=0 ; k<world_size ; k++) _starpu_mpi_early_data_handle_hashmap[k] = NULL;
}

void _starpu_mpi_early_data_check_termination()
{
	STARPU_ASSERT_MSG(_starpu_mpi_early_data_handle_hashmap_count == 0, "Number of copy requests left is not zero, did you forget to post a receive corresponding to a send?");
}

void _starpu_mpi_early_data_free(int world_size)
{
	int n;
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;

	for(n=0 ; n<world_size; n++)
	{
		for(hashlist=_starpu_mpi_early_data_handle_hashmap[n]; hashlist != NULL; hashlist=hashlist->hh.next)
		{
			_starpu_mpi_early_data_handle_list_delete(hashlist->list);
		}
		struct _starpu_mpi_early_data_handle_hashlist *current, *tmp;
		HASH_ITER(hh, _starpu_mpi_early_data_handle_hashmap[n], current, tmp)
		{
			HASH_DEL(_starpu_mpi_early_data_handle_hashmap[n], current);
			free(current);
		}
	}
	free(_starpu_mpi_early_data_handle_hashmap);
}

#ifdef STARPU_VERBOSE
static void _starpu_mpi_early_data_handle_display_hash(int source, int tag)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	HASH_FIND_INT(_starpu_mpi_early_data_handle_hashmap[source], &tag, hashlist);

	if (hashlist == NULL)
	{
		_STARPU_MPI_DEBUG(60, "Hashlist for source %d and tag %d does not exist\n", source, tag);
	}
	else if (_starpu_mpi_early_data_handle_list_empty(hashlist->list))
	{
		_STARPU_MPI_DEBUG(60, "Hashlist for source %d and tag %d is empty\n", source, tag);
	}
	else
	{
		struct _starpu_mpi_early_data_handle *cur;
		for (cur = _starpu_mpi_early_data_handle_list_begin(hashlist->list) ;
		     cur != _starpu_mpi_early_data_handle_list_end(hashlist->list);
		     cur = _starpu_mpi_early_data_handle_list_next(cur))
		{
			_STARPU_MPI_DEBUG(60, "Element for source %d and tag %d: %p\n", source, tag, cur);
		}
	}
}
#endif

static
struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_pop(int mpi_tag, int source, int delete)
{
	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	struct _starpu_mpi_early_data_handle *early_data_handle;

	_STARPU_MPI_DEBUG(60, "Looking for early_data_handle with tag %d in the hashmap[%d]\n", mpi_tag, source);
	HASH_FIND_INT(_starpu_mpi_early_data_handle_hashmap[source], &mpi_tag, hashlist);
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
	_STARPU_MPI_DEBUG(60, "Found early_data_handle %p with tag %d in the hashmap[%d]\n", early_data_handle, mpi_tag, source);
	return early_data_handle;
}

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_find(int mpi_tag, int source)
{
	return _starpu_mpi_early_data_pop(mpi_tag, source, 0);
}

void _starpu_mpi_early_data_add(struct _starpu_mpi_early_data_handle *early_data_handle)
{
	_STARPU_MPI_DEBUG(60, "Trying to add early_data_handle %p with tag %d in the hashmap[%d]\n", early_data_handle, early_data_handle->mpi_tag, early_data_handle->source);

	struct _starpu_mpi_early_data_handle_hashlist *hashlist;
	HASH_FIND_INT(_starpu_mpi_early_data_handle_hashmap[early_data_handle->source], &early_data_handle->mpi_tag, hashlist);
	if (hashlist == NULL)
	{
		hashlist = malloc(sizeof(struct _starpu_mpi_early_data_handle_hashlist));
		hashlist->list = _starpu_mpi_early_data_handle_list_new();
		hashlist->mpi_tag = early_data_handle->mpi_tag;
		HASH_ADD_INT(_starpu_mpi_early_data_handle_hashmap[early_data_handle->source], mpi_tag, hashlist);
	}
	_starpu_mpi_early_data_handle_list_push_back(hashlist->list, early_data_handle);
	_starpu_mpi_early_data_handle_hashmap_count ++;
#ifdef STARPU_VERBOSE
	_starpu_mpi_early_data_handle_display_hash(early_data_handle->source, early_data_handle->mpi_tag);
#endif
}

void _starpu_mpi_early_data_delete(struct _starpu_mpi_early_data_handle *early_data_handle)
{
	_STARPU_MPI_DEBUG(60, "Trying to delete early_data_handle %p with tag %d in the hashmap[%d]\n", early_data_handle, early_data_handle->mpi_tag, early_data_handle->source);
	struct _starpu_mpi_early_data_handle *found = _starpu_mpi_early_data_pop(early_data_handle->mpi_tag, early_data_handle->source, 1);

	STARPU_ASSERT_MSG(found == early_data_handle,
			  "[_starpu_mpi_early_data_delete][error] early_data_handle %p with tag %d is NOT in the hashmap[%d]\n", early_data_handle, early_data_handle->mpi_tag, early_data_handle->source);

	_starpu_mpi_early_data_handle_hashmap_count --;
#ifdef STARPU_VERBOSE
	_starpu_mpi_early_data_handle_display_hash(early_data_handle->source, early_data_handle->mpi_tag);
#endif
}

