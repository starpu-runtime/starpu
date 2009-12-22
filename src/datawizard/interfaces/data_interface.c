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
	return &handle->interface[memory_node];
}

/* register data interface ? (do we need to register ?) descr =  type enum, required to get an id !  */
