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

#ifndef __ALLOC_H__
#define __ALLOC_H__

/** @file */

void _starpu_malloc_init(unsigned dst_node);
void _starpu_malloc_shutdown(unsigned dst_node);

void _starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size);

int _starpu_malloc_flags_on_node(unsigned dst_node, void **A, size_t dim, int flags);
int _starpu_free_flags_on_node(unsigned dst_node, void *A, size_t dim, int flags);
#endif
