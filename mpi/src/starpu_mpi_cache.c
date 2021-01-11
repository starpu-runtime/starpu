/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi_cache.h>
#include <starpu_mpi_cache_stats.h>
#include <starpu_mpi_private.h>

/* Whether we are allowed to keep copies of remote data. */
struct _starpu_data_entry
{
	UT_hash_handle hh;
	starpu_data_handle_t data;
};

static starpu_pthread_mutex_t *_cache_sent_mutex;
static starpu_pthread_mutex_t *_cache_received_mutex;
static struct _starpu_data_entry **_cache_sent_data = NULL;
static struct _starpu_data_entry **_cache_received_data = NULL;
int _starpu_cache_enabled=1;
MPI_Comm _starpu_cache_comm;
int _starpu_cache_comm_size;

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
			_starpu_mpi_cache_free(_starpu_cache_comm_size);
		}
		_starpu_cache_enabled = 0;
	}
	return 0;
}

void _starpu_mpi_cache_init(MPI_Comm comm)
{
	int i;

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
	_STARPU_MPI_DEBUG(2, "Initialising htable for cache\n");

	_STARPU_MPI_MALLOC(_cache_sent_data, _starpu_cache_comm_size * sizeof(struct _starpu_data_entry *));
	_STARPU_MPI_MALLOC(_cache_received_data, _starpu_cache_comm_size * sizeof(struct _starpu_data_entry *));
	_STARPU_MPI_MALLOC(_cache_sent_mutex, _starpu_cache_comm_size * sizeof(starpu_pthread_mutex_t));
	_STARPU_MPI_MALLOC(_cache_received_mutex, _starpu_cache_comm_size * sizeof(starpu_pthread_mutex_t));

	for(i=0 ; i<_starpu_cache_comm_size ; i++)
	{
		_cache_sent_data[i] = NULL;
		_cache_received_data[i] = NULL;
		STARPU_PTHREAD_MUTEX_INIT(&_cache_sent_mutex[i], NULL);
		STARPU_PTHREAD_MUTEX_INIT(&_cache_received_mutex[i], NULL);
	}
	_starpu_mpi_cache_stats_init(comm);
}

static
void _starpu_mpi_cache_empty_tables(int world_size)
{
	int i;
	if (_starpu_cache_enabled == 0) return;

	_STARPU_MPI_DEBUG(2, "Clearing htable for cache\n");

	for(i=0 ; i<world_size ; i++)
	{
		struct _starpu_data_entry *entry, *tmp;

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[i]);
		HASH_ITER(hh, _cache_sent_data[i], entry, tmp)
		{
			HASH_DEL(_cache_sent_data[i], entry);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[i]);

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[i]);
		HASH_ITER(hh, _cache_received_data[i], entry, tmp)
		{
			HASH_DEL(_cache_received_data[i], entry);
			_starpu_mpi_cache_stats_dec(i, entry->data);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[i]);
	}
}

void _starpu_mpi_cache_free()
{
	int i;

	if (_starpu_cache_enabled == 0)
		return;

	_starpu_mpi_cache_empty_tables(_starpu_cache_comm_size);
	free(_cache_sent_data);
	free(_cache_received_data);

	for(i=0 ; i<_starpu_cache_comm_size ; i++)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&_cache_sent_mutex[i]);
		STARPU_PTHREAD_MUTEX_DESTROY(&_cache_received_mutex[i]);
	}
	free(_cache_sent_mutex);
	free(_cache_received_mutex);

	_starpu_mpi_cache_stats_free();
}

void _starpu_mpi_cache_sent_data_clear(MPI_Comm comm, starpu_data_handle_t data)
{
	int n, size;
	starpu_mpi_comm_size(comm, &size);

	for(n=0 ; n<size ; n++)
	{
		struct _starpu_data_entry *already_sent;

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[n]);
		HASH_FIND_PTR(_cache_sent_data[n], &data, already_sent);
		if (already_sent)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data);
			HASH_DEL(_cache_sent_data[n], already_sent);
			free(already_sent);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[n]);
	}
}

void _starpu_mpi_cache_received_data_clear(starpu_data_handle_t data)
{
	int mpi_rank = starpu_mpi_data_get_rank(data);
	struct _starpu_data_entry *already_received;
	if (mpi_rank == STARPU_MPI_PER_NODE)
		return;

	STARPU_MPI_ASSERT_MSG(mpi_rank < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", mpi_rank, _starpu_cache_comm_size);

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[mpi_rank]);
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	if (already_received)
	{
#ifdef STARPU_DEVEL
#  warning TODO: Somebody else will write to the data, so discard our cached copy if any. starpu_mpi could just remember itself.
#endif
		_STARPU_MPI_DEBUG(2, "Clearing receive cache for data %p\n", data);
		HASH_DEL(_cache_received_data[mpi_rank], already_received);
		_starpu_mpi_cache_stats_dec(mpi_rank, data);
		free(already_received);
		starpu_data_invalidate_submit(data);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[mpi_rank]);
}

void starpu_mpi_cache_flush_all_data(MPI_Comm comm)
{
	int nb_nodes, i;
	int mpi_rank, my_rank;

	if (_starpu_cache_enabled == 0)
		return;

	starpu_mpi_comm_size(comm, &nb_nodes);
	starpu_mpi_comm_rank(comm, &my_rank);

	for(i=0 ; i<nb_nodes ; i++)
	{
		struct _starpu_data_entry *entry, *tmp;

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[i]);
		HASH_ITER(hh, _cache_sent_data[i], entry, tmp)
		{
			mpi_rank = starpu_mpi_data_get_rank(entry->data);
			if (mpi_rank != my_rank && mpi_rank != -1)
				starpu_data_invalidate_submit(entry->data);
			HASH_DEL(_cache_sent_data[i], entry);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[i]);

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[i]);
		HASH_ITER(hh, _cache_received_data[i], entry, tmp)
		{
			mpi_rank = starpu_mpi_data_get_rank(entry->data);
			if (mpi_rank != my_rank && mpi_rank != -1)
				starpu_data_invalidate_submit(entry->data);
			HASH_DEL(_cache_received_data[i], entry);
			_starpu_mpi_cache_stats_dec(i, entry->data);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[i]);
	}
}

void _starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	struct _starpu_data_entry *avail;
	int i, nb_nodes;

	if (_starpu_cache_enabled == 0)
		return;

	starpu_mpi_comm_size(comm, &nb_nodes);
	for(i=0 ; i<nb_nodes ; i++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[i]);
		HASH_FIND_PTR(_cache_sent_data[i], &data_handle, avail);
		if (avail)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data_handle);
			HASH_DEL(_cache_sent_data[i], avail);
			free(avail);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[i]);

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[i]);
		HASH_FIND_PTR(_cache_received_data[i], &data_handle, avail);
		if (avail)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data_handle);
			HASH_DEL(_cache_received_data[i], avail);
			_starpu_mpi_cache_stats_dec(i, data_handle);
			free(avail);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[i]);
	}
}

void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int my_rank, mpi_rank;

	_starpu_mpi_cache_flush(comm, data_handle);

	starpu_mpi_comm_rank(comm, &my_rank);
	mpi_rank = starpu_mpi_data_get_rank(data_handle);
	if (mpi_rank != my_rank && mpi_rank != -1)
		starpu_data_invalidate_submit(data_handle);
}

void *_starpu_mpi_cache_received_data_set(starpu_data_handle_t data, int mpi_rank)
{
	struct _starpu_data_entry *already_received;

	if (_starpu_cache_enabled == 0) return NULL;

	STARPU_MPI_ASSERT_MSG(mpi_rank < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", mpi_rank, _starpu_cache_comm_size);

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[mpi_rank]);
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	if (already_received == NULL)
	{
		struct _starpu_data_entry *entry;
		_STARPU_MPI_MALLOC(entry, sizeof(*entry));
		entry->data = data;
		HASH_ADD_PTR(_cache_received_data[mpi_rank], data, entry);
		_STARPU_MPI_DEBUG(2, "Noting that data %p has already been received by %d\n", data, mpi_rank);
		_starpu_mpi_cache_stats_inc(mpi_rank, data);
	}
	else
	{
		_STARPU_MPI_DEBUG(2, "Do not receive data %p from node %d as it is already available\n", data, mpi_rank);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[mpi_rank]);
	return already_received;
}

void *_starpu_mpi_cache_received_data_get(starpu_data_handle_t data, int mpi_rank)
{
	struct _starpu_data_entry *already_received;

	if (_starpu_cache_enabled == 0)
		return NULL;

	if (mpi_rank == STARPU_MPI_PER_NODE)
		return NULL;
	STARPU_MPI_ASSERT_MSG(mpi_rank < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", mpi_rank, _starpu_cache_comm_size);

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[mpi_rank]);
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[mpi_rank]);
	return already_received;
}

int starpu_mpi_cached_receive(starpu_data_handle_t data_handle)
{
	int owner = starpu_mpi_data_get_rank(data_handle);
	void *already_received = _starpu_mpi_cache_received_data_get(data_handle, owner);
	return already_received != NULL;
}

void *_starpu_mpi_cache_sent_data_set(starpu_data_handle_t data, int dest)
{
	struct _starpu_data_entry *already_sent;

	if (_starpu_cache_enabled == 0)
		return NULL;

	STARPU_MPI_ASSERT_MSG(dest < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", dest, _starpu_cache_comm_size);

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[dest]);
	HASH_FIND_PTR(_cache_sent_data[dest], &data, already_sent);
	if (already_sent == NULL)
	{
		struct _starpu_data_entry *entry;
		_STARPU_MPI_MALLOC(entry, sizeof(*entry));
		entry->data = data;
		HASH_ADD_PTR(_cache_sent_data[dest], data, entry);
		_STARPU_MPI_DEBUG(2, "Noting that data %p has already been sent to %d\n", data, dest);
	}
	else
	{
		_STARPU_MPI_DEBUG(2, "Do not send data %p to node %d as it has already been sent\n", data, dest);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[dest]);
	return already_sent;
}

void *_starpu_mpi_cache_sent_data_get(starpu_data_handle_t data, int dest)
{
	struct _starpu_data_entry *already_sent;

	if (_starpu_cache_enabled == 0) return NULL;

	STARPU_MPI_ASSERT_MSG(dest < _starpu_cache_comm_size, "Node %d invalid. Max node is %d\n", dest, _starpu_cache_comm_size);

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[dest]);
	HASH_FIND_PTR(_cache_sent_data[dest], &data, already_sent);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[dest]);
	return already_sent;
}

int starpu_mpi_cached_send(starpu_data_handle_t data_handle, int dest)
{
	void *already_sent = _starpu_mpi_cache_sent_data_get(data_handle, dest);
	return already_sent != NULL;
}

