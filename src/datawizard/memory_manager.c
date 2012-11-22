/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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
#include <common/config.h>
#include <datawizard/memory_manager.h>

static size_t global_size;
static size_t used_size;

int _starpu_memory_manager_init()
{
#ifdef STARPU_DEVEL
#  warning use hwloc to get global size
#endif
     global_size = 0;
     used_size = 0;
     return 0;
}

int _starpu_memory_manager_add_size(size_t size)
{
     used_size += size;
     return 0;
}
