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

#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>
#include <profiling/profiling.h>
#include <common/uthash.h>

struct disk_register {
	unsigned node;
	void * base;
	struct disk_ops * functions;
};

static void add_disk_in_list(unsigned node, struct disk_ops * func, void * base);

static struct disk_register ** disk_register_list = NULL;
static int disk_number = -1;
static int size_register_list = 2;

unsigned
starpu_disk_register(struct disk_ops * func, void *parameter)
{

	unsigned memory_node = _starpu_memory_node_register(STARPU_DISK_RAM, 0);

	_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
	_starpu_register_bus(memory_node, STARPU_MAIN_RAM);

	void * base = func->plug(parameter);

	add_disk_in_list(memory_node,func,base);

	return memory_node;
}

void
starpu_disk_free(unsigned node)
{
	bool find = false;
	int i;
	for (i = 0; i <= disk_number; ++i)
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

static void 
add_disk_in_list(unsigned node,  struct disk_ops * func, void * base)
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
	dr->base = base;
	dr->functions = func;
	disk_register_list[++disk_number] = dr;
}

static int
get_location_with_node(unsigned node)
{
	int i;
	for(i = 0; i <= disk_number; ++i)
		if (disk_register_list[i]->node == node)
			return i;
}


/* */




void *  starpu_posix_alloc  (void *base, size_t size) { char * p; return (void *) p;  } /* nom de fichier: mkstemp, et retourne obj */
	 void    starpu_posix_free   (void *base, void *obj, size_t size) { } /* supprime et libère l'obj */
	 void *  starpu_posix_open   (void *base, void *pos, size_t size) { char * p; return (void *) p; } /* open dans le répertoire  un fichier existant, retourne l'obj */
void   starpu_posix_close  (void *base, void *obj, size_t size) { } /* libère l'obj */
ssize_t  starpu_posix_read   (void *base, void *obj, void *buf, off_t offset, size_t size) { return 0;} /* ~= pread */
ssize_t  starpu_posix_write  (void *base, void *obj, const void *buf, off_t offset, size_t size) { return 0; }
	/* readv, writev, read2d, write2d, etc. */
void *  starpu_posix_plug   (void *parameter) { char * p; return (void *) p; } /* en posix, directory, retourne base */
	 void    starpu_posix_unplug (void *base) { } /* libère la base */

struct disk_ops write_on_file = {
	.alloc = starpu_posix_alloc,
	.free = starpu_posix_free,
	.open = starpu_posix_open,
	.close = starpu_posix_close,
	.read = starpu_posix_read,
	.write = starpu_posix_write,
	.plug = starpu_posix_plug,
	.unplug = starpu_posix_unplug
};
