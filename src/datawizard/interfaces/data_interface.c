/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <datawizard/datawizard.h>

unsigned starpu_get_handle_interface_id(starpu_data_handle handle)
{
	return handle->ops->interfaceid;
}

void *starpu_data_get_interface_on_node(starpu_data_handle handle, unsigned memory_node)
{
	return handle->interface[memory_node];
}

void register_data_handle(starpu_data_handle *handleptr, uint32_t home_node,
				void *interface,
				struct starpu_data_interface_ops_t *ops)
{
	starpu_data_handle handle =
		starpu_data_state_create(ops);

	STARPU_ASSERT(handleptr);
	*handleptr = handle;

	/* fill the interface fields with the appropriate method */
	ops->register_data_handle(handle, home_node, interface);

	register_new_data(handle, home_node, 0);
}
/* register data interface ? (do we need to register ?) descr =  type enum, required to get an id !  */
