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

#include <starpu_mpi_cache.h>
#include <starpu_mpi_cache_stats.h>
#include <starpu_mpi_private.h>

/* Whether we are allowed to keep copies of remote data. */
struct _starpu_data_entry
{
	UT_hash_handle hh;
	void *data;
};

static struct _starpu_data_entry **_cache_sent_data = NULL;
static struct _starpu_data_entry **_cache_received_data = NULL;
int _starpu_cache_enabled=1;

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
			int world_size;
			starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
			MPI_Comm_size(MPI_COMM_WORLD, &world_size);
			_starpu_mpi_cache_free(world_size);
		}
		_starpu_cache_enabled = 0;
	}
	return 0;
}

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
		if (!_starpu_silent) fprintf(stderr,"Warning: StarPU MPI Communication cache is disabled\n");
		return;
	}

	MPI_Comm_size(comm, &nb_nodes);
	_STARPU_MPI_DEBUG(2, "Initialising htable for cache\n");
	_cache_sent_data = malloc(nb_nodes * sizeof(struct _starpu_data_entry *));
	for(i=0 ; i<nb_nodes ; i++) _cache_sent_data[i] = NULL;
	_cache_received_data = malloc(nb_nodes * sizeof(struct _starpu_data_entry *));
	for(i=0 ; i<nb_nodes ; i++) _cache_received_data[i] = NULL;
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
		HASH_ITER(hh, _cache_sent_data[i], entry, tmp)
		{
			HASH_DEL(_cache_sent_data[i], entry);
			free(entry);
		}
		HASH_ITER(hh, _cache_received_data[i], entry, tmp)
		{
			HASH_DEL(_cache_received_data[i], entry);
			_starpu_mpi_cache_stats_dec(i, (starpu_data_handle_t) entry->data);
			free(entry);
		}
	}
}

void _starpu_mpi_cache_free(int world_size)
{
	if (_starpu_cache_enabled == 0) return;

	_starpu_mpi_cache_empty_tables(world_size);
	free(_cache_sent_data);
	free(_cache_received_data);
	_starpu_mpi_cache_stats_free();
}

void _starpu_mpi_cache_sent_data_clear(MPI_Comm comm, starpu_data_handle_t data)
{
	int n, size;
	MPI_Comm_size(comm, &size);

	for(n=0 ; n<size ; n++)
	{
		struct _starpu_data_entry *already_sent;
		HASH_FIND_PTR(_cache_sent_data[n], &data, already_sent);
		if (already_sent)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data);
			HASH_DEL(_cache_sent_data[n], already_sent);
			free(already_sent);
		}
	}
}

void _starpu_mpi_cache_received_data_clear(starpu_data_handle_t data)
{
	int mpi_rank = starpu_mpi_data_get_rank(data);
	struct _starpu_data_entry *already_received;

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
		HASH_ITER(hh, _cache_sent_data[i], entry, tmp)
		{
			mpi_rank = starpu_mpi_data_get_rank((starpu_data_handle_t) entry->data);
			if (mpi_rank != my_rank && mpi_rank != -1)
				starpu_data_invalidate_submit((starpu_data_handle_t) entry->data);
			HASH_DEL(_cache_sent_data[i], entry);
			free(entry);
		}
		HASH_ITER(hh, _cache_received_data[i], entry, tmp)
		{
			mpi_rank = starpu_mpi_data_get_rank((starpu_data_handle_t) entry->data);
			if (mpi_rank != my_rank && mpi_rank != -1)
				starpu_data_invalidate_submit((starpu_data_handle_t) entry->data);
			HASH_DEL(_cache_received_data[i], entry);
			_starpu_mpi_cache_stats_dec(i, (starpu_data_handle_t) entry->data);
			free(entry);
		}
	}
}

void _starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	struct _starpu_data_entry *avail;
	int i, my_rank, nb_nodes;
	int mpi_rank;

	if (_starpu_cache_enabled == 0) return;

	MPI_Comm_size(comm, &nb_nodes);
	MPI_Comm_rank(comm, &my_rank);
	mpi_rank = starpu_mpi_data_get_rank(data_handle);

	for(i=0 ; i<nb_nodes ; i++)
	{
		HASH_FIND_PTR(_cache_sent_data[i], &data_handle, avail);
		if (avail)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data_handle);
			HASH_DEL(_cache_sent_data[i], avail);
			free(avail);
		}
		HASH_FIND_PTR(_cache_received_data[i], &data_handle, avail);
		if (avail)
		{
			_STARPU_MPI_DEBUG(2, "Clearing send cache for data %p\n", data_handle);
			HASH_DEL(_cache_received_data[i], avail);
			_starpu_mpi_cache_stats_dec(i, data_handle);
			free(avail);
		}
	}
}

void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int my_rank, mpi_rank;

	_starpu_mpi_cache_flush(comm, data_handle);

	MPI_Comm_rank(comm, &my_rank);
	mpi_rank = starpu_mpi_data_get_rank(data_handle);
	if (mpi_rank != my_rank && mpi_rank != -1)
		starpu_data_invalidate_submit(data_handle);
}

void *_starpu_mpi_cache_received_data_set(starpu_data_handle_t data, int mpi_rank)
{
	if (_starpu_cache_enabled == 0) return NULL;

	struct _starpu_data_entry *already_received;
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	if (already_received == NULL)
	{
		struct _starpu_data_entry *entry = (struct _starpu_data_entry *)malloc(sizeof(*entry));
		entry->data = data;
		HASH_ADD_PTR(_cache_received_data[mpi_rank], data, entry);
		_starpu_mpi_cache_stats_inc(mpi_rank, data);
	}
	else
	{
		_STARPU_MPI_DEBUG(2, "Do not receive data %p from node %d as it is already available\n", data, mpi_rank);
	}
	return already_received;
}

void *_starpu_mpi_cache_received_data_get(starpu_data_handle_t data, int mpi_rank)
{
	struct _starpu_data_entry *already_received;

	if (_starpu_cache_enabled == 0) return NULL;
	HASH_FIND_PTR(_cache_received_data[mpi_rank], &data, already_received);
	return already_received;
}

void *_starpu_mpi_cache_sent_data_set(starpu_data_handle_t data, int dest)
{
	if (_starpu_cache_enabled == 0) return NULL;

	struct _starpu_data_entry *already_sent;
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
	return already_sent;
}

