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

#include <starpu.h>
#include <common/config.h>
#include <datawizard/memory_manager.h>

static size_t global_size[STARPU_MAXNODES];
static size_t used_size[STARPU_MAXNODES];

int _starpu_memory_manager_init()
{
#ifdef STARPU_DEVEL
#  warning use hwloc to get global size
#endif
	int i;

	for(i=0 ; i<STARPU_MAXNODES ; i++)
	{
		global_size[i] = 0;
		used_size[i] = 0;
	}
	return 0;
}

int _starpu_memory_manager_can_allocate_size(size_t size, unsigned node)
{
	used_size[node] += size;
	return 1;
}

void _starpu_memory_manager_deallocate_size(size_t size, unsigned node)
{
	used_size[node] -= size;
}
