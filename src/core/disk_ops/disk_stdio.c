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
#include <errno.h>
#include <time.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memory_manager.h>

#ifdef STARPU_HAVE_WINDOWS
        #include <io.h>
#endif

#define NITER	64

/* ------------------- use STDIO to write on disk -------------------  */

struct starpu_stdio_obj {
	int descriptor;
	FILE * file;
	char * path;
	double size;
	starpu_pthread_mutex_t mutex;
};


/* allocation memory on disk */
static void * 
starpu_stdio_alloc (void *base, size_t size)
{
	
	struct starpu_stdio_obj * obj = malloc(sizeof(struct starpu_stdio_obj));
	STARPU_ASSERT(obj != NULL);
	int id = -1;

	/* create template for mkstemp */
	char * tmp = "STARPU_XXXXXX";
	char * baseCpy = malloc(strlen(base)+1+strlen(tmp)+1);
	STARPU_ASSERT(baseCpy != NULL);

	strcpy(baseCpy, (char *) base);
	strcat(baseCpy,"/");
	strcat(baseCpy,tmp);

#ifdef STARPU_HAVE_WINDOWS
        _mktemp(baseCpy);
        id = open(baseCpy, O_RDWR | O_BINARY);
#else
	id = mkstemp(baseCpy);

#endif
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

#ifdef STARPU_HAVE_WINDOWS
	int val = _chsize(id, size);
#else
	int val = ftruncate(id,size);
#endif
	/* fail */
	if (val < 0)
	{
		free(obj);
		free(baseCpy);
		unlink(baseCpy);
		return NULL;
	}

	STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

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

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

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
	char * baseCpy = malloc(strlen(base)+1+strlen(pos)+1);
	STARPU_ASSERT(baseCpy != NULL);
	strcpy(baseCpy,(char *) base);
	strcat(baseCpy,(char *) "/");
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

	STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

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

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	fclose(tmp->file);
	close(tmp->descriptor);
	free(tmp->path);
	free(tmp);	
}


/* read the memory disk */
static int 
starpu_stdio_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, void * async_channel STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;
		
	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	int res = fseek(tmp->file, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res == 0, "Stdio read failed");

	ssize_t nb = fread (buf, 1, size, tmp->file);
	STARPU_ASSERT_MSG(nb >= 0, "Stdio read failed");

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return 0;
}

static int
starpu_stdio_full_read(unsigned node, void *base STARPU_ATTRIBUTE_UNUSED, void * obj, void ** ptr, size_t * size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	*size = tmp->size;
	*ptr = malloc(*size);
	return _starpu_disk_read(node, STARPU_MAIN_RAM, obj, *ptr, 0, *size, NULL);
}

/* write on the memory disk */
static int 
starpu_stdio_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size, void * async_channel STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;

	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	int res = fseek(tmp->file, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res == 0, "Stdio write failed");

	fwrite (buf, 1, size, tmp->file);

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return 0;
}

static int
starpu_stdio_full_write (unsigned node, void * base STARPU_ATTRIBUTE_UNUSED, void * obj, void * ptr, size_t size)
{
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) obj;
	
	/* update file size to realise the next good full_read */
	if(size != tmp->size)
	{
		_starpu_memory_manager_deallocate_size(tmp->size, node);
		if (_starpu_memory_manager_can_allocate_size(size, node))
		{
#ifdef STARPU_HAVE_WINDOWS
			int val = _chsize(tmp->descriptor, size);
#else
			int val = ftruncate(tmp->descriptor,size);
#endif

			STARPU_ASSERT_MSG(val >= 0,"StarPU Error to truncate file in STDIO full_write function");
			tmp->size = size;
		}
		else
		{
			STARPU_ASSERT_MSG(0, "Can't allocate size %u on the disk !", (int) size); 
		}
	}	
	return _starpu_disk_write(STARPU_MAIN_RAM, node, obj, ptr, 0, tmp->size, NULL);
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
	char * buf = malloc(SIZE_DISK_MIN);
	STARPU_ASSERT(buf != NULL);
	
	/* allocate memory */
	void * mem = _starpu_disk_alloc(node, SIZE_DISK_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;
	struct starpu_stdio_obj * tmp = (struct starpu_stdio_obj *) mem;

	memset(buf, 0, SIZE_DISK_MIN);

	/* Measure upload slowness */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, SIZE_DISK_MIN, NULL);
		/* clean cache memory */
		int res = fflush (tmp->file);
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed \n");

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(tmp->descriptor);
#else
		res = fsync(tmp->descriptor);
#endif
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
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, rand() % (SIZE_DISK_MIN -1) , 1, NULL);

		int res = fflush (tmp->file);
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(tmp->descriptor);
#else
		res = fsync(tmp->descriptor);
#endif
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
	.bandwidth = get_stdio_bandwidth_between_disk_and_main_ram,
	.full_read = starpu_stdio_full_read,
	.full_write = starpu_stdio_full_write
};
