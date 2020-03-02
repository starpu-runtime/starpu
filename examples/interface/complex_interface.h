/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	double *real;
	double *imaginary;
	int nx;
};

void starpu_complex_data_register(starpu_data_handle_t *handle, unsigned home_node, double *real, double *imaginary, int nx);

double *starpu_complex_get_real(starpu_data_handle_t handle);
double *starpu_complex_get_imaginary(starpu_data_handle_t handle);
int starpu_complex_get_nx(starpu_data_handle_t handle);

#define STARPU_COMPLEX_GET_REAL(interface)	(((struct starpu_complex_interface *)(interface))->real)
#define STARPU_COMPLEX_GET_IMAGINARY(interface)	(((struct starpu_complex_interface *)(interface))->imaginary)
#define STARPU_COMPLEX_GET_NX(interface)	(((struct starpu_complex_interface *)(interface))->nx)

/* Split complex vector into smaller complex vectors */
void starpu_complex_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nchunks);

/* Split complex into two simple vectors */
void starpu_complex_filter_canonical(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nchunks);
struct starpu_data_interface_ops *starpu_complex_filter_canonical_child_ops(struct starpu_data_filter *f, unsigned child);

#endif /* __COMPLEX_INTERFACE_H */
