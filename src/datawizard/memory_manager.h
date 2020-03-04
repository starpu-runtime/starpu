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

#ifndef __MEMORY_MANAGER_H__
#define __MEMORY_MANAGER_H__

/** @file */

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Initialises the memory manager
 */
int _starpu_memory_manager_init();

/**
 * Initialises the global memory size for the given node
 *
 */
void _starpu_memory_manager_set_global_memory_size(unsigned node, size_t size);

/**
 * Gets the global memory size for the given node
 *
 */
size_t _starpu_memory_manager_get_global_memory_size(unsigned node);

int _starpu_memory_manager_test_allocate_size(unsigned node, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* __MEMORY_MANAGER_H__ */
