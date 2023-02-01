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

#ifndef __COMPLEX_DEV_HANDLE_INTERFACE_H
#define __COMPLEX_DEV_HANDLE_INTERFACE_H

/* interface for complex numbers supporting opencl*/
struct starpu_complex_dev_handle_interface
{
	int nx;
    uintptr_t ptr_real;
    uintptr_t dev_handle_real;
    size_t offset_real;
    uintptr_t ptr_imaginary;
    uintptr_t dev_handle_imaginary;
    size_t offset_imaginary;
};

void starpu_complex_dev_handle_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr_real, uintptr_t ptr_imaginary, int nx);

void starpu_complex_dev_handle_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr_real, uintptr_t ptr_imaginary, uintptr_t dev_handle_real, uintptr_t dev_handle_imaginary, size_t offset_real, size_t offset_imaginary);

int starpu_complex_dev_handle_get_nx(starpu_data_handle_t handle);
uintptr_t starpu_complex_dev_handle_get_ptr_real(starpu_data_handle_t handle);
uintptr_t starpu_complex_dev_handle_get_dev_handle_real(starpu_data_handle_t handle);
size_t starpu_complex_dev_handle_get_offset_real(starpu_data_handle_t handle);
uintptr_t starpu_complex_dev_handle_get_ptr_imaginary(starpu_data_handle_t handle);
uintptr_t starpu_complex_dev_handle_get_dev_handle_imaginary(starpu_data_handle_t handle);
size_t starpu_complex_dev_handle_get_offset_imaginary(starpu_data_handle_t handle);

#define STARPU_COMPLEX_DEV_HANDLE_GET_NX(interface)	(((struct starpu_complex_dev_handle_interface *)(interface))->nx)
#define STARPU_COMPLEX_DEV_HANDLE_GET_PTR_REAL(interface) (((struct starpu_complex_dev_handle_interface *)(interface))->ptr_real)
#define STARPU_COMPLEX_DEV_HANDLE_GET_DEV_HANDLE_REAL(interface) (((struct starpu_complex_dev_handle_interface *)(interface))->dev_handle_real)
#define STARPU_COMPLEX_DEV_HANDLE_GET_OFFSET_REAL(interface) (((struct starpu_complex_dev_handle_interface *)(interface))->offset_real)
#define STARPU_COMPLEX_DEV_HANDLE_GET_PTR_IMAGINARY(interface) (((struct starpu_complex_dev_handle_interface *)(interface))->ptr_imaginary)
#define STARPU_COMPLEX_DEV_HANDLE_GET_DEV_HANDLE_IMAGINARY(interface) (((struct starpu_complex_dev_handle_interface *)(interface))->dev_handle_imaginary)
#define STARPU_COMPLEX_DEV_HANDLE_GET_OFFSET_IMAGINARY(interface) (((struct starpu_complex_dev_handle_interface *)(interface))->offset_imaginary)

/* Split complex vector into smaller complex vectors */
void starpu_complex_dev_handle_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nchunks);

/* Split complex into two simple vectors */
void starpu_complex_dev_handle_filter_canonical(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nchunks);
struct starpu_data_interface_ops *starpu_complex_dev_handle_filter_canonical_child_ops(struct starpu_data_filter *f, unsigned child);

#endif /* __COMPLEX_DEV_HANDLE_INTERFACE_H */
