/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_STDLIB_H__
#define __STARPU_STDLIB_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

void starpu_malloc_set_align(size_t align);
int starpu_malloc(void **A, size_t dim);
int starpu_free(void *A);

/* Allocate SIZE bytes on node NODE */
uintptr_t starpu_malloc_on_node(unsigned dst_node, size_t size);
/* Free ADDR on node NODE */
void starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_STDLIB_H__ */
