/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013 Corentin Salingue
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

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <common/config.h>
#include <core/workers.h>
#include <core/debug.h>
#include <core/disk.h>
#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>
#include <profiling/profiling.h>
#include <common/uthash.h>

struct disk_register {
	unsigned node;
	char * src;
	struct disk_ops * functions;
};

static void add_disk_in_list(unsigned node, char * src, struct disk_ops * func);

static struct disk_register ** disk_register_list = NULL;
static int disk_number = -1;
static int size_register_list = 2;

unsigned
starpu_disk_register(char * src, struct disk_ops * func)
{

	unsigned memory_node = _starpu_memory_node_register(STARPU_DISK_RAM, 0);

	_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
	_starpu_register_bus(memory_node, STARPU_MAIN_RAM);

	add_disk_in_list(memory_node,src,func);

	return memory_node;
}

void
starpu_disk_free(unsigned node)
{

	bool find = false;
	int i;
	for (i = 0; i < disk_number; ++i)
	{
		if (find)
			disk_register_list[i-1] = disk_register_list[i];
		if (disk_register_list[i]->node == node)
		{
			free(disk_register_list[i]);
			find = true; 
		}
	}

	disk_number--;

	if (disk_register_list != NULL && disk_number == -1)
	{
		free(disk_register_list);
		disk_register_list = NULL;
	}
}

static void add_disk_in_list(unsigned node, char * src, struct disk_ops * func)
{
	/* initialization */
	if(disk_register_list == NULL)
	{
		disk_register_list = malloc(size_register_list*sizeof(struct disk_register *));
		STARPU_ASSERT(disk_register_list != NULL);
	}
	/* small size -> new size  */
	if((disk_number+1) > size_register_list)
	{
		struct disk_register ** ptr_realloc = realloc(disk_register_list, 2*size_register_list*sizeof(struct disk_register *));
 
		if (ptr_realloc != NULL)
		{
			size_register_list *= 2;
			disk_register_list = ptr_realloc;
		}
		else
		{
			STARPU_ASSERT(ptr_realloc != NULL);
		}
	}
	struct disk_register * dr = malloc(sizeof(struct disk_register));
	STARPU_ASSERT(dr != NULL);
	dr->node = node;
	dr->src = src;
	dr->functions = func;
	disk_register_list[disk_number++] = dr;
}


