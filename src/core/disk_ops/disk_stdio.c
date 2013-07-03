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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>

#define NITER	64

/* ------------------- use STDIO to write on disk -------------------  */

struct starpu_stdio_obj {
	int descriptor;
	FILE * file;
	char * path;
	double size;
};


/* allocation memory on disk */
static void * 
starpu_stdio_alloc (void *base, size_t size STARPU_ATTRIBUTE_UNUSED)
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

	char * tmp = "STARPU_XXXXXX";

	strcpy(baseCpy, (char *) base);
	strcat(baseCpy,tmp);

	id = mkstemp(baseCpy);
	/* fail */
	if (id < 0)
	{
		free(obj);
		free(baseCpy);
		return NULL;
	}

	FILE * f = fdopen(id, "rb+");
	/* fail */
	if (f == NULL)
	{
		/* delete fic */
		free(obj);
		free(baseCpy);
		unlink(baseCpy);
		return NULL;
	}

	int val = ftruncate(id,size);
	/* fail */
	if (val < 0)
	{
		free(obj);
		free(baseCpy);
		unlink(baseCpy);
		return NULL;
	}

	obj->descriptor = id;
	obj->file = f;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
}


/* free memory on disk */
static void
starpu_stdio_free (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
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
starpu_stdio_open (void *base, void *pos, size_t size)
{
	struct starpu_stdio_obj * obj = malloc(sizeof(struct starpu_stdio_obj));
	STARPU_ASSERT(obj != NULL);

	/* create template */
	unsigned int sizeBase = 16;
	while(sizeBase < (strlen(base)+strlen(pos)+1))
		sizeBase *= 2;
	
	char * baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);
	strcpy(baseCpy,(char *) base);
	strcat(baseCpy,(char *) pos);

	int id = open(baseCpy, O_RDWR);
	if (id < 0)
	{
		free(obj);
		free(baseCpy);
		return NULL;
	}

	FILE * f = fdopen(id,"rb+");
	if (f == NULL)
	{
		free(obj);
		free(baseCpy);
		return NULL;
	}

	obj->descriptor = id;
	obj->file = f;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
	
}


/* free memory without delete it */
static void 
starpu_stdio_close (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	fclose(tmp->file);
	close(tmp->descriptor);
	free(tmp->path);
	free(tmp);	
}


/* read the memory disk */
static ssize_t 
starpu_stdio_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	int res = fseek(tmp->file, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res == 0, "Stdio read failed");

	ssize_t nb = fread (buf, 1, size, tmp->file);

	return nb;
}


/* write on the memory disk */
static ssize_t 
starpu_stdio_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size)
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


static int
get_stdio_bandwidth_between_disk_and_main_ram(unsigned node)
{

	unsigned iter;
	double timing_slowness, timing_latency;
	struct timeval start;
	struct timeval end;
	
	srand (time (NULL)); 
	char * buf = malloc(SIZE_DISK_MIN*sizeof(char));
	STARPU_ASSERT(buf != NULL);
	
	/* allocate memory */
	void * mem = _starpu_disk_alloc(node, SIZE_DISK_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) mem;

	/* Measure upload slowness */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(node, mem, buf, 0, SIZE_DISK_MIN);
		/* clean cache memory */
		int res = fflush (tmp->file);
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed \n");

		res = fsync(tmp->descriptor);
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed \n");
	}
	gettimeofday(&end, NULL);
	timing_slowness = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));


	/* free memory */
	free(buf);

	buf = malloc(sizeof(char));
	STARPU_ASSERT(buf != NULL);

	/* Measure latency */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(node, mem, buf, rand() % (SIZE_DISK_MIN -1) , 1);

		int res = fflush (tmp->file);
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");

		res = fsync(tmp->descriptor);
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");
	}
	gettimeofday(&end, NULL);
	timing_latency = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	_starpu_disk_free(node, mem, SIZE_DISK_MIN);
	free(buf);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*1000000, (NITER/timing_slowness)*1000000,
					       timing_latency/NITER, timing_latency/NITER, node);
	return 1;
}



struct starpu_disk_ops starpu_disk_stdio_ops = {
	.alloc = starpu_stdio_alloc,
	.free = starpu_stdio_free,
	.open = starpu_stdio_open,
	.close = starpu_stdio_close,
	.read = starpu_stdio_read,
	.write = starpu_stdio_write,
	.plug = starpu_stdio_plug,
	.unplug = starpu_stdio_unplug,
	.copy = NULL,
	.bandwidth = get_stdio_bandwidth_between_disk_and_main_ram
};
