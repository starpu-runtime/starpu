/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __COMPLEX_INTERFACE_H
#define __COMPLEX_INTERFACE_H

/* interface for complex numbers */
struct starpu_complex_interface
{
	uintptr_t ptr;
	uintptr_t dev_handle;
	size_t offset;
	int nx;
};

void starpu_complex_data_register(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, int nx);
void starpu_complex_ptr_register(starpu_data_handle_t handle, int node, uintptr_t ptr, uintptr_t dev_handle, size_t offset);
void starpu_complex_data_register_ops();

uintptr_t starpu_complex_get_ptr(starpu_data_handle_t handle);
uintptr_t starpu_complex_dev_handle_get_dev_handle(starpu_data_handle_t handle);
int starpu_complex_get_nx(starpu_data_handle_t handle);

#define STARPU_COMPLEX_GET_PTR(interface)      (((struct starpu_complex_interface *)(interface))->ptr)
#define STARPU_COMPLEX_GET_DEV_REAL(interface) (((struct starpu_complex_interface *)(interface))->dev_handle)
#define STARPU_COMPLEX_GET_NX(interface)       (((struct starpu_complex_interface *)(interface))->nx)

/* Split complex vector into smaller complex vectors */
void starpu_complex_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nchunks);

#endif /* __COMPLEX_INTERFACE_H */
