/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013, 2014, 2015  CNRS
 * Copyright (C) 2011-2015  Université de Bordeaux
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
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <common/uthash.h>
#include <common/starpu_spinlock.h>
#include <datawizard/coherency.h>

/* Entry in the `registered_tag_handles' hash table.  */
struct handle_tag_entry
{
	UT_hash_handle hh;
	int tag;
	starpu_data_handle_t handle;
};

/* Hash table mapping host tags to data handles.  */
static struct handle_tag_entry *registered_tag_handles;
static struct _starpu_spinlock    registered_tag_handles_lock;

void _starpu_mpi_tag_init(void)
{
	_starpu_spin_init(&registered_tag_handles_lock);
}

void _starpu_mpi_tag_free(void)
{
     	struct handle_tag_entry *tag_entry, *tag_tmp;

	_starpu_spin_destroy(&registered_tag_handles_lock);

	HASH_ITER(hh, registered_tag_handles, tag_entry, tag_tmp)
	{
		HASH_DEL(registered_tag_handles, tag_entry);
		free(tag_entry);
	}

	registered_tag_handles = NULL;
}

starpu_data_handle_t _starpu_mpi_data_get_data_handle_from_tag(int tag)
{
	struct handle_tag_entry *ret;

	_starpu_spin_lock(&registered_tag_handles_lock);
	HASH_FIND_INT(registered_tag_handles, &tag, ret);
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

void _starpu_mpi_data_register_tag(starpu_data_handle_t handle, int tag)
{
	struct handle_tag_entry *entry;
	entry = (struct handle_tag_entry *) malloc(sizeof(*entry));
	STARPU_ASSERT(entry != NULL);

	STARPU_ASSERT_MSG(!(_starpu_mpi_data_get_data_handle_from_tag(tag)),
			  "There is already a data handle %p registered with the tag %d\n", _starpu_mpi_data_get_data_handle_from_tag(tag), tag);

	_STARPU_MPI_DEBUG(42, "Adding handle %p with tag %d in hashtable\n", handle, tag);

	entry->handle = handle;
	entry->tag = tag;

	_starpu_spin_lock(&registered_tag_handles_lock);
	HASH_ADD_INT(registered_tag_handles, tag, entry);
	_starpu_spin_unlock(&registered_tag_handles_lock);
}

int _starpu_mpi_data_release_tag(starpu_data_handle_t handle)
{
	struct handle_tag_entry *tag_entry;
	int tag = starpu_mpi_data_get_tag(handle);

	_STARPU_MPI_DEBUG(42, "Removing handle %p with tag %d from hashtable\n", handle, tag);

	if (tag != -1)
	{
		_starpu_spin_lock(&registered_tag_handles_lock);
		HASH_FIND_INT(registered_tag_handles, &(((struct _starpu_mpi_node_tag *)(handle->mpi_data))->data_tag), tag_entry);
		STARPU_ASSERT_MSG((tag_entry != NULL),"Data handle %p with tag %d isn't in the hashmap !",handle,tag);

		HASH_DEL(registered_tag_handles, tag_entry);

		_starpu_spin_unlock(&registered_tag_handles_lock);

		free(tag_entry);
	}
	return 0;
}
