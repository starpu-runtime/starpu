/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi.h>
#include <mpi_failure_tolerance/ulfm/starpu_mpi_ulfm_comm.h>
#include <common/list.h>
#include <common/uthash.h>
//for testing purposes
#include <mpi/starpu_mpi_comm.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_service_comms.h>

struct _starpu_mpi_ulfm_comm_hashtable
{
	UT_hash_handle hh;
	MPI_Comm key_comm;
	MPI_Comm comm;
};

struct _starpu_mpi_ulfm_comm_hashtable *_starpu_mpi_ulfm_key_to_comm_lut;
struct _starpu_mpi_ulfm_comm_hashtable *_starpu_mpi_ulfm_comm_to_key_lut;

static starpu_pthread_rwlock_t mpi_ulfm_comms_mutex;

void _starpu_mpi_ulfm_comm_init()
{
	STARPU_PTHREAD_RWLOCK_INIT(&mpi_ulfm_comms_mutex, NULL);
}

void _starpu_mpi_ulfm_comm_shutdown()
{
	struct _starpu_mpi_ulfm_comm_hashtable *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, _starpu_mpi_ulfm_key_to_comm_lut, entry, tmp)
	{
		HASH_DEL(_starpu_mpi_ulfm_key_to_comm_lut, entry);
		free(entry);
	}
	HASH_ITER(hh, _starpu_mpi_ulfm_comm_to_key_lut, entry, tmp)
	{
		HASH_DEL(_starpu_mpi_ulfm_comm_to_key_lut, entry);
		free(entry);
	}

	STARPU_PTHREAD_RWLOCK_DESTROY(&mpi_ulfm_comms_mutex);
}

void _starpu_mpi_ulfm_comm_register(MPI_Comm key_comm)
{
	struct _starpu_mpi_ulfm_comm_hashtable *found;
	STARPU_PTHREAD_RWLOCK_RDLOCK(&mpi_ulfm_comms_mutex);
	HASH_FIND(hh, _starpu_mpi_ulfm_key_to_comm_lut, &key_comm, sizeof(MPI_Comm), found);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&mpi_ulfm_comms_mutex);
	if (found)
	{
		_STARPU_MPI_DEBUG(10, "[FT_ULFM] comm %ld (%ld) already registered in lookup tables\n", (long int)key_comm, (long int)MPI_COMM_WORLD);
	}

	STARPU_PTHREAD_RWLOCK_WRLOCK(&mpi_ulfm_comms_mutex);
	HASH_FIND(hh, _starpu_mpi_ulfm_key_to_comm_lut, &key_comm, sizeof(MPI_Comm), found);
	if (found)
	{
		_STARPU_MPI_DEBUG(10, "[FT_ULFM] comm %ld (%ld) already registered in lookup tables in between\n", (long int)key_comm, (long int)MPI_COMM_WORLD);
	}
	else
	{
		struct _starpu_mpi_ulfm_comm_hashtable *key_to_comm_entry;
		struct _starpu_mpi_ulfm_comm_hashtable *comm_to_key_entry;
		_STARPU_MPI_MALLOC(key_to_comm_entry, sizeof(*key_to_comm_entry));
		_STARPU_MPI_MALLOC(comm_to_key_entry, sizeof(*comm_to_key_entry));
		key_to_comm_entry->comm = key_comm;
		key_to_comm_entry->key_comm = key_comm;
		comm_to_key_entry->comm     = key_comm;
		comm_to_key_entry->key_comm = key_comm;
		HASH_ADD(hh, _starpu_mpi_ulfm_key_to_comm_lut, key_comm, sizeof(key_to_comm_entry->key_comm), key_to_comm_entry);
		HASH_ADD(hh, _starpu_mpi_ulfm_comm_to_key_lut, comm, sizeof(comm_to_key_entry->comm), comm_to_key_entry);
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&mpi_ulfm_comms_mutex);
}

void _starpu_mpi_ulfm_comm_update(MPI_Comm key_comm, MPI_Comm comm)
{
	struct _starpu_mpi_ulfm_comm_hashtable *found;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&mpi_ulfm_comms_mutex);
	// update the existing key_to_comm entry
	HASH_FIND(hh, _starpu_mpi_ulfm_key_to_comm_lut, &key_comm, sizeof(MPI_Comm), found);
	STARPU_MPI_ASSERT_MSG(found, "[FT_ULFM] Could not find key communicator %ld in _starpu_mpi_ulfm_key_to_comm_lut", (long int)key_comm);
	found->comm = comm;

	// add a new comm_to_key_entry
	struct _starpu_mpi_ulfm_comm_hashtable *comm_to_key_entry;
	_STARPU_MPI_MALLOC(comm_to_key_entry, sizeof(*comm_to_key_entry));
	comm_to_key_entry->comm = comm;
	comm_to_key_entry->key_comm = key_comm;
	HASH_ADD(hh, _starpu_mpi_ulfm_comm_to_key_lut, comm, sizeof(comm_to_key_entry->comm), comm_to_key_entry);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&mpi_ulfm_comms_mutex);
}

void _starpu_mpi_ulfm_comm_delete(MPI_Comm comm)
{
	struct _starpu_mpi_ulfm_comm_hashtable *entry_to_delete;
	HASH_FIND(hh, _starpu_mpi_ulfm_comm_to_key_lut, &comm, sizeof(MPI_Comm), entry_to_delete);
	// Check if the to be deleted comm is not a key comm
	struct _starpu_mpi_ulfm_comm_hashtable *found;
	HASH_FIND(hh, _starpu_mpi_ulfm_key_to_comm_lut, &entry_to_delete->key_comm, sizeof(MPI_Comm), found);
	STARPU_MPI_ASSERT_MSG(found->comm != comm, "[FT_ULFM] Trying to delete communicator pointed by a key communicator.");
	HASH_DEL(_starpu_mpi_ulfm_comm_to_key_lut, entry_to_delete);
}

void _starpu_mpi_ulfm_comm_test_update(MPI_Comm key_comm, MPI_Comm comm)
{
	_starpu_mpi_ulfm_comm_update(key_comm, comm);
	_starpu_mpi_comm_register(key_comm);
	starpu_mpi_ft_service_comm_update(comm);
}

MPI_Comm _starpu_mpi_ulfm_get_key_comm(starpu_mpi_comm comm)
{
	struct _starpu_mpi_ulfm_comm_hashtable *found;
	HASH_FIND(hh, _starpu_mpi_ulfm_comm_to_key_lut, &comm, sizeof(MPI_Comm), found);
	STARPU_MPI_ASSERT_MSG(found, "[FT_ULFM] Couldn't find the key comm for the MPI comm %ld.", (long int)comm);
	return found->key_comm;
}

starpu_mpi_comm _starpu_mpi_ulfm_get_mpi_comm_from_key(MPI_Comm key_comm)
{
	struct _starpu_mpi_ulfm_comm_hashtable *found;
	HASH_FIND(hh, _starpu_mpi_ulfm_key_to_comm_lut, &key_comm, sizeof(MPI_Comm), found);
	STARPU_MPI_ASSERT_MSG(found, "[FT_ULFM] Couldn't find the MPI comm from the key comm %ld.", (long int)key_comm);
	return found->comm;
}
