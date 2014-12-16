/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2011-2014  Université de Bordeaux
 * Copyright (C) 2014 INRIA
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

void _starpu_mpi_cache_init(MPI_Comm comm)
{
	int nb_nodes;
	int i;

	_starpu_cache_enabled = starpu_get_env_number("STARPU_MPI_CACHE");
	if (_starpu_cache_enabled == -1)
	{
		_starpu_cache_enabled = 1;
	}

	if (_starpu_cache_enabled == 0)
	{
		if (!getenv("STARPU_SILENT")) fprintf(stderr,"Warning: StarPU MPI Communication cache is disabled\n");
		return;
	}

	MPI_Comm_size(comm, &nb_nodes);
	_STARPU_MPI_DEBUG(2, "Initialising htable for cache\n");

	_cache_sent_data = malloc(nb_nodes * sizeof(struct _starpu_data_entry *));
	_cache_received_data = malloc(nb_nodes * sizeof(struct _starpu_data_entry *));
	_cache_sent_mutex = malloc(nb_nodes * sizeof(starpu_pthread_mutex_t));
	_cache_received_mutex = malloc(nb_nodes * sizeof(starpu_pthread_mutex_t));

	for(i=0 ; i<nb_nodes ; i++)
	{
		_cache_sent_data[i] = NULL;
		_cache_received_data[i] = NULL;
		STARPU_PTHREAD_MUTEX_INIT(&_cache_sent_mutex[i], NULL);
		STARPU_PTHREAD_MUTEX_INIT(&_cache_received_mutex[i], NULL);
	}
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
			_starpu_mpi_cache_stats_dec(-1, i, entry->data);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[i]);
	}
}

void _starpu_mpi_cache_free(int world_size)
{
	int i;

	if (_starpu_cache_enabled == 0) return;

	_starpu_mpi_cache_empty_tables(world_size);
	free(_cache_sent_data);
	free(_cache_received_data);

	for(i=0 ; i<world_size ; i++)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&_cache_sent_mutex[i]);
		STARPU_PTHREAD_MUTEX_DESTROY(&_cache_received_mutex[i]);
	}
	free(_cache_sent_mutex);
	free(_cache_received_mutex);

	_starpu_mpi_cache_stats_free();
}

void _starpu_mpi_cache_flush_sent(MPI_Comm comm, starpu_data_handle_t data)
{
	int n, size;
	MPI_Comm_size(comm, &size);

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

void _starpu_mpi_cache_flush_recv(starpu_data_handle_t data, int me)
{
	int mpi_rank = starpu_data_get_rank(data);
	struct _starpu_data_entry *already_received;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[mpi_rank]);
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	if (already_received)
	{
#ifdef STARPU_DEVEL
#  warning TODO: Somebody else will write to the data, so discard our cached copy if any. starpu_mpi could just remember itself.
#endif
		_STARPU_MPI_DEBUG(2, "Clearing receive cache for data %p\n", data);
		HASH_DEL(_cache_received_data[mpi_rank], already_received);
		_starpu_mpi_cache_stats_dec(me, mpi_rank, data);
		free(already_received);
		starpu_data_invalidate_submit(data);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[mpi_rank]);
}

void starpu_mpi_cache_flush_all_data(MPI_Comm comm)
{
	int nb_nodes, i;
	int mpi_rank, my_rank;

	if (_starpu_cache_enabled == 0) return;

	MPI_Comm_size(comm, &nb_nodes);
	MPI_Comm_rank(comm, &my_rank);

	for(i=0 ; i<nb_nodes ; i++)
	{
		struct _starpu_data_entry *entry, *tmp;

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[i]);
		HASH_ITER(hh, _cache_sent_data[i], entry, tmp)
		{
			mpi_rank = starpu_data_get_rank(entry->data);
			if (mpi_rank != my_rank && mpi_rank != -1)
				starpu_data_invalidate_submit(entry->data);
			HASH_DEL(_cache_sent_data[i], entry);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_sent_mutex[i]);

		STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[i]);
		HASH_ITER(hh, _cache_received_data[i], entry, tmp)
		{
			mpi_rank = starpu_data_get_rank(entry->data);
			if (mpi_rank != my_rank && mpi_rank != -1)
				starpu_data_invalidate_submit(entry->data);
			HASH_DEL(_cache_received_data[i], entry);
			_starpu_mpi_cache_stats_dec(my_rank, i, entry->data);
			free(entry);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[i]);
	}
}

void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	struct _starpu_data_entry *avail;
	int i, my_rank, nb_nodes;
	int mpi_rank;

	if (_starpu_cache_enabled == 0) return;

	MPI_Comm_size(comm, &nb_nodes);
	MPI_Comm_rank(comm, &my_rank);
	mpi_rank = starpu_data_get_rank(data_handle);

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
			_starpu_mpi_cache_stats_dec(my_rank, i, data_handle);
			free(avail);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[i]);
	}

	if (mpi_rank != my_rank && mpi_rank != -1)
		starpu_data_invalidate_submit(data_handle);
}

void *_starpu_mpi_already_received(int src, starpu_data_handle_t data, int mpi_rank)
{
	struct _starpu_data_entry *already_received;

	if (_starpu_cache_enabled == 0) return NULL;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_received_mutex[mpi_rank]);
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	if (already_received == NULL)
	{
		struct _starpu_data_entry *entry = (struct _starpu_data_entry *)malloc(sizeof(*entry));
		entry->data = data;
		HASH_ADD_PTR(_cache_received_data[mpi_rank], data, entry);
		_starpu_mpi_cache_stats_inc(src, mpi_rank, data);
	}
	else
	{
		_STARPU_MPI_DEBUG(2, "Do not receive data %p from node %d as it is already available\n", data, mpi_rank);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_cache_received_mutex[mpi_rank]);
	return already_received;
}

void *_starpu_mpi_already_sent(starpu_data_handle_t data, int dest)
{
	struct _starpu_data_entry *already_sent;

	if (_starpu_cache_enabled == 0) return NULL;

	STARPU_PTHREAD_MUTEX_LOCK(&_cache_sent_mutex[dest]);
	HASH_FIND_PTR(_cache_sent_data[dest], &data, already_sent);
	if (already_sent == NULL)
	{
		struct _starpu_data_entry *entry = (struct _starpu_data_entry *)malloc(sizeof(*entry));
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

