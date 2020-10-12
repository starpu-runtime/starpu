/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <starpu_mpi_private.h>
#include <common/config.h>

#include "data_movements_interface.h"

#if defined(STARPU_USE_MPI_MPI)

starpu_mpi_tag_t **data_movements_get_ref_tags_table(starpu_data_handle_t handle)
{
	struct data_movements_interface *dm_interface =
		(struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	if (dm_interface->tags)
		return &dm_interface->tags;
	else
		return NULL;
}

int **data_movements_get_ref_ranks_table(starpu_data_handle_t handle)
{
	struct data_movements_interface *dm_interface =
		(struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	if (dm_interface->ranks)
		return &dm_interface->ranks;
	else
		return NULL;
}

starpu_mpi_tag_t *data_movements_get_tags_table(starpu_data_handle_t handle)
{
	struct data_movements_interface *dm_interface =
		(struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return dm_interface->tags;
}

int *data_movements_get_ranks_table(starpu_data_handle_t handle)
{
	struct data_movements_interface *dm_interface =
		(struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return dm_interface->ranks;
}

int data_movements_get_size_tables(starpu_data_handle_t handle)
{
	struct data_movements_interface *dm_interface =
		(struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return dm_interface->size;
}

int data_movements_reallocate_tables(starpu_data_handle_t handle, int size)
{
	struct data_movements_interface *dm_interface =
		(struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	if (dm_interface->size)
	{
		STARPU_ASSERT(dm_interface->tags);
		free(dm_interface->tags);
		dm_interface->tags = NULL;

		STARPU_ASSERT(dm_interface->ranks);
		free(dm_interface->ranks);
		dm_interface->ranks = NULL;
	}
	else
	{
		STARPU_ASSERT(!dm_interface->tags);
		STARPU_ASSERT(!dm_interface->ranks);
	}

	dm_interface->size = size;

	if (dm_interface->size)
	{
		_STARPU_MPI_MALLOC(dm_interface->tags, size*sizeof(*dm_interface->tags));
		_STARPU_MPI_MALLOC(dm_interface->ranks, size*sizeof(*dm_interface->ranks));
	}

	return 0 ;
}

static void data_movements_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct data_movements_interface *dm_interface = (struct data_movements_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct data_movements_interface *local_interface = (struct data_movements_interface *)
			starpu_data_get_interface_on_node(handle, node);

		local_interface->size = dm_interface->size;
		if (node == home_node)
		{
			local_interface->tags = dm_interface->tags;
			local_interface->ranks = dm_interface->ranks;
		}
		else
		{
			local_interface->tags = NULL;
			local_interface->ranks = NULL;
		}
	}
}

static starpu_ssize_t data_movements_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct data_movements_interface *dm_interface = (struct data_movements_interface *) data_interface;

	starpu_mpi_tag_t *addr_tags;
	int *addr_ranks;
	starpu_ssize_t requested_memory = dm_interface->size * sizeof(int);

	addr_tags = (int*) starpu_malloc_on_node(node, requested_memory);
	if (!addr_tags)
		goto fail_tags;
	addr_ranks = (int*) starpu_malloc_on_node(node, requested_memory);
	if (!addr_ranks)
		goto fail_ranks;

	/* update the data properly in consequence */
	dm_interface->tags = addr_tags;
	dm_interface->ranks = addr_ranks;

	return 2*requested_memory;

fail_ranks:
	starpu_free_on_node(node, (uintptr_t) addr_tags, requested_memory);
fail_tags:
	return -ENOMEM;
}

static void data_movements_free_data_on_node(void *data_interface, unsigned node)
{
	struct data_movements_interface *dm_interface = (struct data_movements_interface *) data_interface;
	starpu_ssize_t requested_memory_tags = dm_interface->size * sizeof(starpu_mpi_tag_t);
	starpu_ssize_t requested_memory_ranks = dm_interface->size * sizeof(int);

	starpu_free_on_node(node, (uintptr_t) dm_interface->tags, requested_memory_tags);
	starpu_free_on_node(node, (uintptr_t) dm_interface->ranks, requested_memory_ranks);
}

static size_t data_movements_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct data_movements_interface *dm_interface = (struct data_movements_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	size = (dm_interface->size * sizeof(starpu_mpi_tag_t)) + (dm_interface->size * sizeof(int)) + sizeof(int);
	return size;
}

static uint32_t data_movements_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(data_movements_get_size(handle), 0);
}

static int data_movements_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct data_movements_interface *dm_interface = (struct data_movements_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = data_movements_get_size(handle);
	if (ptr != NULL)
	{
		char *data;
		starpu_malloc_flags((void**) &data, *count, 0);
		assert(data);
		*ptr = data;
		memcpy(data, &dm_interface->size, sizeof(int));
		if (dm_interface->size)
		{
			memcpy(data+sizeof(int), dm_interface->tags, (dm_interface->size*sizeof(starpu_mpi_tag_t)));
			memcpy(data+sizeof(int)+(dm_interface->size*sizeof(starpu_mpi_tag_t)), dm_interface->ranks, dm_interface->size*sizeof(int));
		}
	}

	return 0;
}

static int data_movements_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	char *data = ptr;
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct data_movements_interface *dm_interface = (struct data_movements_interface *)
		starpu_data_get_interface_on_node(handle, node);

	int size = 0;
	memcpy(&size, data, sizeof(int));
	STARPU_ASSERT(count == (2 * size * sizeof(int)) + sizeof(int));

	data_movements_reallocate_tables(handle, size);

	if (dm_interface->size)
	{
		memcpy(dm_interface->tags, data+sizeof(int), dm_interface->size*sizeof(starpu_mpi_tag_t));
		memcpy(dm_interface->ranks, data+sizeof(int)+(dm_interface->size*sizeof(starpu_mpi_tag_t)), dm_interface->size*sizeof(int));
	}

    return 0;
}

static int copy_any_to_any(void *src_interface, unsigned src_node,
			   void *dst_interface, unsigned dst_node,
			   void *async_data)
{
	struct data_movements_interface *src_data_movements = src_interface;
	struct data_movements_interface *dst_data_movements = dst_interface;
	int ret = 0;

	if (starpu_interface_copy((uintptr_t) src_data_movements->tags, 0, src_node,
				    (uintptr_t) dst_data_movements->tags, 0, dst_node,
				     src_data_movements->size*sizeof(starpu_mpi_tag_t),
				     async_data))
		ret = -EAGAIN;
	if (starpu_interface_copy((uintptr_t) src_data_movements->ranks, 0, src_node,
				    (uintptr_t) dst_data_movements->ranks, 0, dst_node,
				     src_data_movements->size*sizeof(int),
				     async_data))
		ret = -EAGAIN;
	return ret;
}

static const struct starpu_data_copy_methods data_movements_copy_methods =
{
	.any_to_any = copy_any_to_any
};

static struct starpu_data_interface_ops interface_data_movements_ops =
{
	.register_data_handle = data_movements_register_data_handle,
	.allocate_data_on_node = data_movements_allocate_data_on_node,
	.free_data_on_node = data_movements_free_data_on_node,
	.copy_methods = &data_movements_copy_methods,
	.get_size = data_movements_get_size,
	.footprint = data_movements_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct data_movements_interface),
	.to_pointer = NULL,
	.pack_data = data_movements_pack_data,
	.unpack_data = data_movements_unpack_data,
	.describe = NULL
};

void data_movements_data_register(starpu_data_handle_t *handleptr, unsigned home_node, int *ranks, starpu_mpi_tag_t *tags, int size)
{
	struct data_movements_interface data_movements =
	{
		.tags = tags,
		.ranks = ranks,
		.size = size
	};

	if (interface_data_movements_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_data_movements_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &data_movements, &interface_data_movements_ops);
}

#endif
