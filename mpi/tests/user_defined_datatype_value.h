/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef _USER_DEFINED_DATATYPE_VALUE_H
#define _USER_DEFINED_DATATYPE_VALUE_H

struct starpu_value_interface
{
	int *value;
};
#define STARPU_VALUE_GET(interface)	(((struct starpu_value_interface *)(interface))->value)

int *starpu_value_get(starpu_data_handle_t handle)
{
	struct starpu_value_interface *value_interface =
		(struct starpu_value_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return value_interface->value;
}

static void value_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_value_interface *value_interface = (struct starpu_value_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_value_interface *local_interface = (struct starpu_value_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
			local_interface->value = value_interface->value;
		else
			local_interface->value = 0;
	}
}

static starpu_ssize_t value_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_value_interface *value_interface = (struct starpu_value_interface *) data_interface;
	int *addr = 0;

	addr = (int *) starpu_malloc_on_node(node, sizeof(int));
	if (!addr)
		return -ENOMEM;

	/* update the data properly in consequence */
	value_interface->value = addr;

	return sizeof(int);
}

static void value_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_value_interface *value_interface = (struct starpu_value_interface *) data_interface;

	starpu_free_on_node(node, (uintptr_t) value_interface->value, sizeof(int));
}

static size_t value_get_size(starpu_data_handle_t handle)
{
	(void)handle;
	return sizeof(int);
}

static uint32_t value_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(value_get_size(handle), 0);
}

static void *value_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_value_interface *value_interface = data_interface;

	return (void*) value_interface->value;
}

static int value_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_value_interface *value_interface = (struct starpu_value_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = sizeof(int);
	if (ptr != NULL)
	{
		*ptr = (void*) starpu_malloc_on_node_flags(node, *count, 0);
		memcpy(*ptr, value_interface->value, sizeof(int));
	}

	return 0;
}

static int value_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	(void)count;
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_value_interface *value_interface = (struct starpu_value_interface *)
		starpu_data_get_interface_on_node(handle, node);

	value_interface->value[0] = ((int *)ptr)[0];

	assert(value_interface->value[0] == 36);

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

static int copy_any_to_any(void *src_interface, unsigned src_node,
			   void *dst_interface, unsigned dst_node,
			   void *async_data)
{
	struct starpu_value_interface *src_value = src_interface;
	struct starpu_value_interface *dst_value = dst_interface;

	return starpu_interface_copy((uintptr_t) src_value->value, 0, src_node,
				     (uintptr_t) dst_value->value, 0, dst_node,
				     sizeof(int),
				     async_data);
}

static const struct starpu_data_copy_methods value_copy_methods =
{
	.any_to_any = copy_any_to_any
};

static struct starpu_data_interface_ops interface_value_ops =
{
	.register_data_handle = value_register_data_handle,
	.allocate_data_on_node = value_allocate_data_on_node,
	.free_data_on_node = value_free_data_on_node,
	.copy_methods = &value_copy_methods,
	.get_size = value_get_size,
	.footprint = value_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_value_interface),
	.to_pointer = value_to_pointer,
	.pack_data = value_pack_data,
	.unpack_data = value_unpack_data
};

void starpu_value_data_register(starpu_data_handle_t *handleptr, unsigned home_node, int *value)
{
	struct starpu_value_interface value_int =
	{
		.value = value
	};

	if (interface_value_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_value_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &value_int, &interface_value_ops);
}

#endif /* _USER_DEFINED_DATATYPE_VALUE_H */
