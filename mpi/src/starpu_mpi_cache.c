/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/uthash.h>
#include <datawizard/coherency.h>

#include <starpu_mpi_cache.h>
#include <starpu_mpi_cache_stats.h>
#include <starpu_mpi_private.h>

/* Whether we are allowed to keep copies of remote data. */
struct _starpu_data_entry
{
	UT_hash_handle hh;
	starpu_data_handle_t data_handle;
};

static starpu_pthread_mutex_t _cache_mutex;
static struct _starpu_data_entry *_cache_data = NULL;
int _starpu_cache_enabled=1;
static MPI_Comm _starpu_cache_comm;
static int _starpu_cache_comm_size;

static void _starpu_mpi_cache_flush_nolock(starpu_data_handle_t data_handle);

int starpu_mpi_cache_is_enabled()
{
	return _starpu_cache_enabled==1;
}

int starpu_mpi_cache_set(int enabled)
{
	if (enabled == 1)
	{
		_starpu_cache_enabled = 1;
	}
	else
	{
		if (_starpu_cache_enabled)
		{
			// We need to clean the cache
			starpu_mpi_cache_flush_all_data(_starpu_cache_comm);
			_starpu_mpi_cache_shutdown();
		}
		_starpu_cache_enabled = 0;
	}
	return 0;
}

void _starpu_mpi_cache_init(MPI_Comm comm)
{
	_starpu_cache_enabled = starpu_get_env_number("STARPU_MPI_CACHE");
	if (_starpu_cache_enabled == -1)
	{
		_starpu_cache_enabled = 1;
	}

	if (_starpu_cache_enabled == 0)
	{
		_STARPU_DISP("Warning: StarPU MPI Communication cache is disabled\n");
		return;
	}

	_starpu_cache_comm = comm;
	starpu_mpi_comm_size(comm, &_starpu_cache_comm_size);
	_starpu_mpi_cache_stats_init();
	STARPU_PTHREAD_MUTEX_INIT(&_cache_mutex, NULL);
}

void _starpu_mpi_cache_shutdown()
{
	if (_starpu_cache_enabled == 0)
		return;

	struct _starpu_data_entry *entry=NULL, *tmp=NULL;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	HASH_ITER(hh, _cache_data, entry, tmp)
	{
		HASH_DEL(_cache_data, entry);
		free(entry);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&_cache_mutex);
	_starpu_mpi_cache_stats_shutdown();
}

void _starpu_mpi_cache_data_clear(starpu_data_handle_t data_handle)
{
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 1)
	{
		struct _starpu_data_entry *entry;
		STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
		_starpu_mpi_cache_flush_nolock(data_handle);
		HASH_FIND_PTR(_cache_data, &data_handle, entry);
		if (entry != NULL)
		{
			HASH_DEL(_cache_data, entry);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
	}

	free(mpi_data->cache_sent);
}

void _starpu_mpi_cache_data_init(starpu_data_handle_t data_handle)
{
	int i;
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 0)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	mpi_data->cache_received = 0;
	_STARPU_MALLOC(mpi_data->cache_sent, _starpu_cache_comm_size*sizeof(mpi_data->cache_sent[0]));
	for(i=0 ; i<_starpu_cache_comm_size ; i++)
	{
		mpi_data->cache_sent[i] = 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
}

static void _starpu_mpi_cache_data_add_nolock(starpu_data_handle_t data_handle)
{
	struct _starpu_data_entry *entry;

	if (_starpu_cache_enabled == 0)
		return;

	HASH_FIND_PTR(_cache_data, &data_handle, entry);
	if (entry == NULL)
	{
		_STARPU_MPI_MALLOC(entry, sizeof(*entry));
		entry->data_handle = data_handle;
		HASH_ADD_PTR(_cache_data, data_handle, entry);
	}
}

static void _starpu_mpi_cache_data_remove_nolock(starpu_data_handle_t data_handle)
{
	struct _starpu_data_entry *entry;

	if (_starpu_cache_enabled == 0)
		return;

	HASH_FIND_PTR(_cache_data, &data_handle, entry);
	if (entry)
	{
		HASH_DEL(_cache_data, entry);
		free(entry);
	}
}

/**************************************
 * Received cache
 **************************************/
void starpu_mpi_cached_receive_clear(starpu_data_handle_t data_handle)
{
	int mpi_rank = starpu_mpi_data_get_rank(data_handle);
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 0)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	STARPU_ASSERT(mpi_data->magic == 42);
	STARPU_MPI_ASSERT_MSG(mpi_rank < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", mpi_rank, _starpu_cache_comm_size);

	if (mpi_data->cache_received == 1)
	{
#ifdef STARPU_DEVEL
#  warning TODO: Somebody else will write to the data, so discard our cached copy if any. starpu_mpi could just remember itself.
#endif
		_STARPU_MPI_DEBUG(2, "Clearing receive cache for data %p\n", data_handle);
		mpi_data->cache_received = 0;
		starpu_data_invalidate_submit(data_handle);
		_starpu_mpi_cache_data_remove_nolock(data_handle);
		_starpu_mpi_cache_stats_dec(mpi_rank, data_handle);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
}

int starpu_mpi_cached_receive_set(starpu_data_handle_t data_handle)
{
	int mpi_rank = starpu_mpi_data_get_rank(data_handle);
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 0)
		return 0;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	STARPU_ASSERT(mpi_data->magic == 42);
	STARPU_MPI_ASSERT_MSG(mpi_rank < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", mpi_rank, _starpu_cache_comm_size);

	int already_received = mpi_data->cache_received;
	if (already_received == 0)
	{
		_STARPU_MPI_DEBUG(2, "Noting that data %p has already been received by %d\n", data_handle, mpi_rank);
		mpi_data->cache_received = 1;
		_starpu_mpi_cache_data_add_nolock(data_handle);
		_starpu_mpi_cache_stats_inc(mpi_rank, data_handle);
	}
	else
	{
		_STARPU_MPI_DEBUG(2, "Do not receive data %p from node %d as it is already available\n", data_handle, mpi_rank);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
	return already_received;
}

int starpu_mpi_cached_receive(starpu_data_handle_t data_handle)
{
	int already_received;
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 0)
		return 0;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	STARPU_ASSERT(mpi_data->magic == 42);
	already_received = mpi_data->cache_received;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
	return already_received;
}

/**************************************
 * Send cache
 **************************************/
void starpu_mpi_cached_send_clear(starpu_data_handle_t data_handle)
{
	int n, size;
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 0)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	starpu_mpi_comm_size(mpi_data->node_tag.node.comm, &size);
	for(n=0 ; n<size ; n++)
	{
		if (mpi_data->cache_sent[n] == 1)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data_handle);
			mpi_data->cache_sent[n] = 0;
			_starpu_mpi_cache_data_remove_nolock(data_handle);
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
}

int starpu_mpi_cached_send_set(starpu_data_handle_t data_handle, int dest)
{
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;

	if (_starpu_cache_enabled == 0)
		return 0;

	STARPU_MPI_ASSERT_MSG(dest < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", dest, _starpu_cache_comm_size);

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	int already_sent = mpi_data->cache_sent[dest];
	if (mpi_data->cache_sent[dest] == 0)
	{
		mpi_data->cache_sent[dest] = 1;
		_starpu_mpi_cache_data_add_nolock(data_handle);
		_STARPU_MPI_DEBUG(2, "Noting that data %p has already been sent to %d\n", data_handle, dest);
	}
	else
	{
		_STARPU_MPI_DEBUG(2, "Do not send data %p to node %d as it has already been sent\n", data_handle, dest);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
	return already_sent;
}

int starpu_mpi_cached_send(starpu_data_handle_t data_handle, int dest)
{
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;
	int already_sent;

	if (_starpu_cache_enabled == 0)
		return 0;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	STARPU_MPI_ASSERT_MSG(dest < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", dest, _starpu_cache_comm_size);
	already_sent = mpi_data->cache_sent[dest];
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
	return already_sent;
}

static void _starpu_mpi_cache_flush_nolock(starpu_data_handle_t data_handle)
{
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;
	int i, nb_nodes;

	if (_starpu_cache_enabled == 0)
		return;

	starpu_mpi_comm_size(mpi_data->node_tag.node.comm, &nb_nodes);
	for(i=0 ; i<nb_nodes ; i++)
	{
		if (mpi_data->cache_sent[i] == 1)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data_handle);
			mpi_data->cache_sent[i] = 0;
			_starpu_mpi_cache_stats_dec(i, data_handle);
		}
	}

	if (mpi_data->cache_received == 1)
	{
		int mpi_rank = starpu_mpi_data_get_rank(data_handle);
		_STARPU_MPI_DEBUG(2, "Clearing received cache for data %p\n", data_handle);
		mpi_data->cache_received = 0;
		_starpu_mpi_cache_stats_dec(mpi_rank, data_handle);
	}
}

void _starpu_mpi_cache_flush(starpu_data_handle_t data_handle)
{
	if (_starpu_cache_enabled == 0)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	_starpu_mpi_cache_flush_nolock(data_handle);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
}

static void _starpu_mpi_cache_flush_and_invalidate_nolock(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int my_rank, mpi_rank;

	_starpu_mpi_cache_flush_nolock(data_handle);

	starpu_mpi_comm_rank(comm, &my_rank);
	mpi_rank = starpu_mpi_data_get_rank(data_handle);
	if (mpi_rank != my_rank && mpi_rank != -1)
		starpu_data_invalidate_submit(data_handle);
}

void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	_starpu_mpi_data_flush(data_handle);

	if (_starpu_cache_enabled == 0)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	_starpu_mpi_cache_flush_and_invalidate_nolock(comm, data_handle);
	_starpu_mpi_cache_data_remove_nolock(data_handle);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
}

void starpu_mpi_cache_flush_all_data(MPI_Comm comm)
{
	struct _starpu_data_entry *entry=NULL, *tmp=NULL;

	if (_starpu_cache_enabled == 0)
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_mutex);
	HASH_ITER(hh, _cache_data, entry, tmp)
	{
		_starpu_mpi_cache_flush_and_invalidate_nolock(comm, entry->data_handle);
		HASH_DEL(_cache_data, entry);
		free(entry);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_mutex);
}
