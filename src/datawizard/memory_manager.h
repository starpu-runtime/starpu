/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2013  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include <common/config.h>
#include <core/workers.h>

/**
 * Initialises the memory manager
 */
int _starpu_memory_manager_init();

/**
 * Initialises the global memory for the given node
 *
 */
void _starpu_memory_manager_init_global_memory(unsigned node, enum starpu_archtype type, int devid, struct _starpu_machine_config *config);

/**
 * Indicates if memory can be allocated on the given node
 *
 * @param size amount of memory to allocate
 * @param node node where the memory is to be allocated
 * @return 1 if the given amount of memory can be allocated on the given node
 */
int _starpu_memory_manager_can_allocate_size(size_t size, unsigned node) STARPU_WARN_UNUSED_RESULT;

/**
 * Indicates the given amount of memory is going to be deallocated from the given node
 *
 * @param size amount of memory to be deallocated
 * @param node node where the memory is going to be deallocated
 */
void _starpu_memory_manager_deallocate_size(size_t size, unsigned node);

#endif /* __MEMORY_MANAGER_H__ */
