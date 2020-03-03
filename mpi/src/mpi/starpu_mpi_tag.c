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
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <common/uthash.h>
#include <common/starpu_spinlock.h>
#include <datawizard/coherency.h>

#ifdef STARPU_USE_MPI_MPI

/* Entry in the `registered_tag_handles' hash table.  */
struct handle_tag_entry
{
	UT_hash_handle hh;
	starpu_mpi_tag_t data_tag;
	starpu_data_handle_t handle;
};

/* Hash table mapping host tags to data handles.  */
static struct handle_tag_entry *registered_tag_handles;
static struct _starpu_spinlock    registered_tag_handles_lock;

void _starpu_mpi_tag_init(void)
{
	_starpu_spin_init(&registered_tag_handles_lock);
}

void _starpu_mpi_tag_shutdown(void)
{
     	struct handle_tag_entry *tag_entry=NULL, *tag_tmp=NULL;

	_starpu_spin_destroy(&registered_tag_handles_lock);

	HASH_ITER(hh, registered_tag_handles, tag_entry, tag_tmp)
	{
		HASH_DEL(registered_tag_handles, tag_entry);
		free(tag_entry);
	}

	registered_tag_handles = NULL;
}

starpu_data_handle_t _starpu_mpi_tag_get_data_handle_from_tag(starpu_mpi_tag_t data_tag)
{
	struct handle_tag_entry *ret;

	_starpu_spin_lock(&registered_tag_handles_lock);
	HASH_FIND(hh, registered_tag_handles, &data_tag, sizeof(ret->data_tag), ret);
	_starpu_spin_unlock(&registered_tag_handles_lock);

	if (ret)
	{
		return ret->handle;
	}
	else
	{
		return NULL;
	}
}

void _starpu_mpi_tag_data_register(starpu_data_handle_t handle, starpu_mpi_tag_t data_tag)
{
	if (data_tag == -1)
	{
		/* No tag for this data, probably a temporary data not to be communicated */
		return;
	}

	struct handle_tag_entry *entry;
	_STARPU_MPI_MALLOC(entry, sizeof(*entry));

	STARPU_ASSERT_MSG(!(_starpu_mpi_tag_get_data_handle_from_tag(data_tag)),
			  "There is already a data handle %p registered with the tag %ld\n", _starpu_mpi_tag_get_data_handle_from_tag(data_tag), data_tag);

	_STARPU_MPI_DEBUG(42, "Adding handle %p with tag %"PRIi64" in hashtable\n", handle, data_tag);

	entry->handle = handle;
	entry->data_tag = data_tag;

	_starpu_spin_lock(&registered_tag_handles_lock);
#ifndef STARPU_NO_ASSERT
	struct handle_tag_entry *old;
	HASH_FIND(hh, registered_tag_handles, &data_tag, sizeof(entry->data_tag), old);
	STARPU_ASSERT_MSG(!old, "tag %"PRIi64" being registered for data %p, but is already used by data %p!\n", data_tag, handle, old?old->handle:NULL);
#endif
	HASH_ADD(hh, registered_tag_handles, data_tag, sizeof(entry->data_tag), entry);
	_starpu_spin_unlock(&registered_tag_handles_lock);
}

int _starpu_mpi_tag_data_release(starpu_data_handle_t handle)
{
	starpu_mpi_tag_t data_tag = starpu_mpi_data_get_tag(handle);

	_STARPU_MPI_DEBUG(42, "Removing handle %p with tag %"PRIi64" from hashtable\n", handle, data_tag);

	if (data_tag != -1)
	{
		struct handle_tag_entry *tag_entry;

		_starpu_spin_lock(&registered_tag_handles_lock);
		HASH_FIND(hh, registered_tag_handles, &(((struct _starpu_mpi_data *)(handle->mpi_data))->node_tag.data_tag), sizeof(tag_entry->data_tag), tag_entry);
		STARPU_ASSERT_MSG((tag_entry != NULL),"Data handle %p with tag %"PRIi64" isn't in the hashmap !", handle, data_tag);

		HASH_DEL(registered_tag_handles, tag_entry);

		_starpu_spin_unlock(&registered_tag_handles_lock);

		free(tag_entry);
	}
	return 0;
}

#endif // STARPU_USE_MPI_MPI
