/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <starpu.h>
#include <common/config.h>

/* Some data interfaces or filters use this interface internally */
extern struct starpu_data_interface_ops starpu_interface_matrix_ops;
void _starpu_data_free_interfaces(starpu_data_handle_t handle)
	STARPU_ATTRIBUTE_INTERNAL;

extern
int _starpu_data_handle_init(starpu_data_handle_t handle, struct starpu_data_interface_ops *interface_ops, unsigned int mf_node);

extern void _starpu_data_interface_init(void) STARPU_ATTRIBUTE_INTERNAL;
extern int _starpu_data_check_not_busy(starpu_data_handle_t handle) STARPU_ATTRIBUTE_INTERNAL;
extern void _starpu_data_interface_shutdown(void) STARPU_ATTRIBUTE_INTERNAL;

extern void _starpu_data_register_ram_pointer(starpu_data_handle_t handle,
						void *ptr)
	STARPU_ATTRIBUTE_INTERNAL;

extern void _starpu_data_unregister_ram_pointer(starpu_data_handle_t handle)
	STARPU_ATTRIBUTE_INTERNAL;

#define _starpu_data_is_multiformat_handle(handle) handle->ops->is_multiformat

#endif // __DATA_INTERFACE_H__
