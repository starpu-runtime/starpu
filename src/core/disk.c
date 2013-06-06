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
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <common/config.h>
#include <core/workers.h>
#include <core/debug.h>
#include <core/perfmodel/perfmodel.h>
#include <datawizard/memory_manager.h>

#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>
#include <profiling/profiling.h>
#include <common/uthash.h>

#define SIZE	(1024*1024)
#define NITER	64


struct disk_register {
	unsigned node;
	void * base;
	struct disk_ops * functions;
};

static void add_disk_in_list(unsigned node, struct disk_ops * func, void * base);
static int get_location_with_node(unsigned node);

static struct disk_register ** disk_register_list = NULL;
static int disk_number = -1;
static int size_register_list = 2;


unsigned
starpu_disk_register(struct disk_ops * func, void *parameter, size_t size)
{

	STARPU_ASSERT_MSG(size >= SIZE,"Minimum disk size is %u Bytes ! (Here %u)", SIZE, size);
	/* register disk */
	unsigned memory_node = _starpu_memory_node_register(STARPU_DISK_RAM, 0);

	_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
	_starpu_register_bus(memory_node, STARPU_MAIN_RAM);

	/* connect disk */
	void * base = func->plug(parameter);

	/* remember it */
	add_disk_in_list(memory_node,func,base);

	func->bandwidth(base,memory_node);
	_starpu_memory_manager_set_global_memory_size(memory_node, size);
	return memory_node;
}


void
starpu_disk_unregister(unsigned node)
{
	bool find = false;
	int i;
	
	/* search disk and delete it */
	for (i = 0; i <= disk_number; ++i)
	{
		if (find)
			disk_register_list[i-1] = disk_register_list[i];
		if (disk_register_list[i]->node == node)
		{
			/* don't forget to unplug */
			disk_register_list[i]->functions->unplug(disk_register_list[i]->base);
			free(disk_register_list[i]);
			find = true; 
		}
	}
	
	/* no disk in the list -> delete the list */
	STARPU_ASSERT_MSG(find, "Disk node not found !(%u) ", node);
	disk_number--;

	if (disk_register_list != NULL && disk_number == -1)
	{
		free(disk_register_list);
		disk_register_list = NULL;
	}
}


/* interface between user and disk memory */

void *  
starpu_disk_alloc(unsigned node, size_t size)
{
	int pos = get_location_with_node(node);
	return disk_register_list[pos]->functions->alloc(disk_register_list[pos]->base, size);
}

void
starpu_disk_free(unsigned node, void *obj, size_t size)
{
	int pos = get_location_with_node(node);
	disk_register_list[pos]->functions->free(disk_register_list[pos]->base, obj, size);
}

ssize_t 
starpu_disk_read(unsigned node, void *obj, void *buf, off_t offset, size_t size)
{
	int pos = get_location_with_node(node);
	return disk_register_list[pos]->functions->read(disk_register_list[pos]->base, obj, buf, offset, size);
}


ssize_t 
starpu_disk_write(unsigned node, void *obj, const void *buf, off_t offset, size_t size)
{
	int pos = get_location_with_node(node);
	return disk_register_list[pos]->functions->write(disk_register_list[pos]->base, obj, buf, offset, size);
}

int
starpu_disk_copy(unsigned node_src, void* obj_src, off_t offset_src, unsigned node_dst, void* obj_dst, off_t offset_dst, size_t size)
{
	int pos_src = get_location_with_node(node_src);
	int pos_dst = get_location_with_node(node_dst);
	/* both nodes have same copy function */
	return disk_register_list[pos_src]->functions->copy(disk_register_list[pos_src]->base, obj_src, offset_src, 
						        disk_register_list[pos_dst]->base, obj_dst, offset_dst,
							size);

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
	STARPU_ASSERT_MSG(false, "Disk node not found !(%u) ", node);
	return -1;
}

int 
starpu_is_same_kind_disk(unsigned node1, unsigned node2)
{
	
	if(starpu_node_get_kind(node1) == STARPU_DISK_RAM && starpu_node_get_kind(node2) == STARPU_DISK_RAM)
	{
		int pos1 = get_location_with_node(node1);
		int pos2 = get_location_with_node(node2);
		if(disk_register_list[pos1]->functions == disk_register_list[pos2]->functions)
			/* they must have a copy function */
			if(disk_register_list[pos1]->functions->copy != NULL)
				return 1;
	}
	return 0;
}

/* ------------------- use STDIO to write on disk -------------------  */

struct starpu_stdio_obj {
	int descriptor;
	FILE * file;
	char * path;
	double size;
};


/* allocation memory on disk */
static void * 
starpu_stdio_alloc (void *base, size_t size)
{
	struct starpu_stdio_obj * obj = malloc(sizeof(struct starpu_stdio_obj));
	STARPU_ASSERT(obj != NULL);
	int id = -1;

	/* create template for mkstemp */
	unsigned int sizeBase = 16;
	while(sizeBase < (strlen(base)+7))
		sizeBase *= 2;

	char * baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);
	char * tmp = "XXXXXX";

	strcpy(baseCpy, (char *) base);
	strcat(baseCpy,tmp);

	id = mkstemp(baseCpy);
	STARPU_ASSERT_MSG(id >= 0, "Stdio allocation failed");

	FILE * f = fdopen(id, "rb+");
	STARPU_ASSERT_MSG(f != NULL, "Stdio allocation failed");

	int val = ftruncate(id,size);
	STARPU_ASSERT_MSG(val >= 0, "Stdio allocation failed");

	obj->descriptor = id;
	obj->file = f;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
}


/* free memory on disk */
static void
starpu_stdio_free (void *base, void *obj, size_t size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	unlink(tmp->path);
	fclose(tmp->file);
	close(tmp->descriptor);

	free(tmp->path);
	free(tmp);
}


/* open an existing memory on disk */
static void * 
starpu_unistd_open (void *base, void *pos, size_t size)
{
	struct starpu_stdio_obj * obj = malloc(sizeof(struct starpu_stdio_obj));
	STARPU_ASSERT(obj != NULL);

	/* create template for mkstemp */
	unsigned int sizeBase = 16;
	while(sizeBase < (strlen(base)+strlen(pos)+1))
		sizeBase *= 2;
	
	char * baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);
	strcpy(baseCpy,(char *) base);
	strcat(baseCpy,(char *) pos);

	int id = open(baseCpy, O_RDONLY);
	STARPU_ASSERT_MSG(id >= 0, "Unistd open failed");

	FILE * f = fdopen(id,"rb+");
	STARPU_ASSERT_MSG(f != NULL, "Unistd open failed");

	obj->descriptor = id;
	obj->file = f;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
	
}


/* free memory without delete it */
static void 
starpu_unistd_close (void *base, void *obj, size_t size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	fclose(tmp->file);
	close(tmp->descriptor);
	free(tmp->path);
	free(tmp);	
}


/* read the memory disk */
static ssize_t 
starpu_stdio_read (void *base, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;
	
	int res = fseek(tmp->file, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res == 0, "Stdio read failed");

	ssize_t nb = fread (buf, 1, size, tmp->file);
	return nb;
}


/* write on the memory disk */
static ssize_t 
starpu_stdio_write (void *base, void *obj, const void *buf, off_t offset, size_t size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	int res = fseek(tmp->file, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res == 0, "Stdio write failed");

	ssize_t nb = fwrite (buf, 1, size, tmp->file);

	return nb;
}


/* create a new copy of parameter == base */
static void * 
starpu_stdio_plug (void *parameter)
{
	char * tmp = malloc(sizeof(char)*(strlen(parameter)+1));
	STARPU_ASSERT(tmp != NULL);
	strcpy(tmp,(char *) parameter);
	return (void *) tmp;	
}


/* free memory allocated for the base */
static void
starpu_stdio_unplug (void *base)
{
	free(base);
}


static void
get_stdio_bandwidth_between_disk_and_main_ram(void * base, unsigned node)
{

	unsigned iter;
	double timing_slowness, timing_latency;
	struct timeval start;
	struct timeval end;
	
	srand (time (NULL)); 
	int pos = get_location_with_node(node);
	char * buf = malloc(SIZE*sizeof(char));
	STARPU_ASSERT(buf != NULL);
	
	/* allocate memory */
	void * mem = disk_register_list[pos]->functions->alloc(base, SIZE);
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) mem;

	/* Measure upload slowness */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		disk_register_list[pos]->functions->write(base, mem, buf, 0, SIZE);
		/* clean cache memory */
		int res = fflush (tmp->file);
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed");

		res = fsync(tmp->descriptor);
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed");
	}
	gettimeofday(&end, NULL);
	timing_slowness = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));


	/* free memory */
	disk_register_list[pos]->functions->free(base, mem, SIZE);
	free(buf);

	mem = disk_register_list[pos]->functions->alloc(base, 2*SIZE);
	tmp = (struct starpu_stdio_obj *) mem;
	buf = malloc(sizeof(char));
	STARPU_ASSERT(buf != NULL);

	/* Measure latency */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		disk_register_list[pos]->functions->write(base, mem, buf, rand() % ((2*SIZE)-1) +1 , 1);

		int res = fflush (tmp->file);
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");

		res = fsync(tmp->descriptor);
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");
	}
	gettimeofday(&end, NULL);
	timing_latency = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	disk_register_list[pos]->functions->free(base, mem, SIZE);
	free(buf);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*1000000, (NITER/timing_slowness)*1000000,
					       timing_latency/NITER, timing_latency/NITER, node);
}



struct disk_ops write_on_file = {
	.alloc = starpu_stdio_alloc,
	.free = starpu_stdio_free,
	.open = starpu_unistd_open,
	.close = starpu_unistd_close,
	.read = starpu_stdio_read,
	.write = starpu_stdio_write,
	.plug = starpu_stdio_plug,
	.unplug = starpu_stdio_unplug,
	.copy = NULL,
	.bandwidth = get_stdio_bandwidth_between_disk_and_main_ram
};
