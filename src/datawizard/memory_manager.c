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
#include <starpu_cuda.h>
#include <starpu_opencl.h>

static size_t global_size[STARPU_MAXNODES];
static size_t used_size[STARPU_MAXNODES];

int _starpu_memory_manager_init()
{
	int i;

	for(i=0 ; i<STARPU_MAXNODES ; i++)
	{
		global_size[i] = 0;
		used_size[i] = 0;
	}
	return 0;
}

void _starpu_memory_manager_init_global_memory(unsigned node, enum starpu_archtype type, int devid, struct _starpu_machine_config *config)
{
	switch (type)
	{
#ifdef STARPU_USE_CPU
	case STARPU_CPU_WORKER:
	{
		/* FIXME: when we have NUMA support, properly turn node number into NUMA node number */
		global_size[node] = _starpu_cpu_get_global_mem_size(node, config);
		break;
	}
#endif

#ifdef STARPU_USE_CUDA
	case STARPU_CUDA_WORKER:
	{
		global_size[node] = starpu_cuda_get_global_mem_size(devid);
		break;
	}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
	case STARPU_OPENCL_WORKER:
	{
		global_size[node] = starpu_opencl_get_global_mem_size(devid);
		break;
	}
#endif /* STARPU_USE_OPENCL */

	default:
		STARPU_ABORT();
	}

	_STARPU_DEBUG("Global size for node %d (%d) is %ld\n", node, type, (long)global_size[node]);
}

int _starpu_memory_manager_can_allocate_size(size_t size, unsigned node)
{
	if (global_size[node] == 0)
	{
		// We do not have information on the available size, let's suppose it is going to fit
		used_size[node] += size;
		return 1;
	}
	else if (used_size[node] + size < global_size[node])
	{
		used_size[node] += size;
		return 1;
	}
	else
	{
		return 0;
	}
}

void _starpu_memory_manager_deallocate_size(size_t size, unsigned node)
{
	used_size[node] -= size;
}
