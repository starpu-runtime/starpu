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
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdint.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <core/disk_ops/unistd/disk_unistd_global.h>

#if STARPU_HAVE_WINDOWS
        #include <io.h>
#endif

#define NITER	64
#define SIZE_BENCH (4*getpagesize())

/* ------------------- use UNISTD to write on disk -------------------  */

/* allocation memory on disk */
 void * 
starpu_unistd_global_alloc (struct starpu_unistd_global_obj * obj, void *base, size_t size)
{
	int id = -1;
	const char *template = "STARPU_XXXXXX";

	/* create template for mkstemp */
	unsigned int sizeBase = strlen(base) + strlen(template)+1;

	char * baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);

	strcpy(baseCpy, (char *) base);
	strcat(baseCpy,template);
#ifdef STARPU_LINUX_SYS
	id = mkostemp(baseCpy, obj->flags);
#elif STARPU_HAVE_WINDOWS
	/* size in windows is a multiple of char */
	_mktemp_s(baseCpy, size/sizeof(char));
	id = open(baseCpy, obj->flags);
#else
	STARPU_ASSERT(obj->flags == O_RDWR);
	id = mkstemp(baseCpy);
#endif

	/* fail */
	if (id < 0)
	{
		free(obj);
		free(baseCpy);
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
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
}


/* free memory on disk */
 void
starpu_unistd_global_free (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	unlink(tmp->path);
	close(tmp->descriptor);

	free(tmp->path);
	free(tmp);
}


/* open an existing memory on disk */
 void * 
starpu_unistd_global_open (struct starpu_unistd_global_obj * obj, void *base, void *pos, size_t size)
{
	/* create template */
	unsigned int sizeBase = 16;
	while(sizeBase < (strlen(base)+strlen(pos)+1))
		sizeBase *= 2;
	
	char * baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);
	strcpy(baseCpy,(char *) base);
	strcat(baseCpy,(char *) pos);

	int id = open(baseCpy, obj->flags);
	if (id < 0)
	{
		free(obj);
		free(baseCpy);
		return NULL;
	}

	obj->descriptor = id;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
	
}


/* free memory without delete it */
 void 
starpu_unistd_global_close (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	close(tmp->descriptor);
	free(tmp->path);
	free(tmp);	
}


/* read the memory disk */
 ssize_t 
starpu_unistd_global_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	int res = lseek(tmp->descriptor, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd read failed");

	ssize_t nb = read(tmp->descriptor, buf, size);
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd read failed");
	
	return nb;
}


/* write on the memory disk */
 ssize_t 
starpu_unistd_global_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	int res = lseek(tmp->descriptor, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd write failed");

	ssize_t nb = write (tmp->descriptor, buf, size);
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd write failed");

	return nb;
}


/* create a new copy of parameter == base */
 void * 
starpu_unistd_global_plug (void *parameter)
{
	char * tmp = malloc(sizeof(char)*(strlen(parameter)+1));
	STARPU_ASSERT(tmp != NULL);
	strcpy(tmp,(char *) parameter);

	return (void *) tmp;	
}


/* free memory allocated for the base */
 void
starpu_unistd_global_unplug (void *base)
{
	free(base);
}


 int
get_unistd_global_bandwidth_between_disk_and_main_ram(unsigned node)
{

	unsigned iter;
	double timing_slowness, timing_latency;
	struct timeval start;
	struct timeval end;

	srand (time (NULL)); 
	char * buf;
	starpu_malloc((void *) &buf, SIZE_BENCH*sizeof(char));
	STARPU_ASSERT(buf != NULL);
	
	/* allocate memory */
	void * mem = _starpu_disk_alloc(node, SIZE_BENCH);
	/* fail to alloc */
	if (mem == NULL)
		return 0;

	/* Measure upload slowness */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(node, mem, buf, 0, SIZE_BENCH);
	}
	gettimeofday(&end, NULL);
	timing_slowness = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));


	/* free memory */
	starpu_free(buf);

	
	starpu_malloc((void *) &buf, getpagesize()*sizeof(char));
	STARPU_ASSERT(buf != NULL);

	/* Measure latency */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(node, mem, buf, rand() % (SIZE_BENCH -1) , getpagesize());
	}
	gettimeofday(&end, NULL);
	timing_latency = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	_starpu_disk_free(node, mem, SIZE_BENCH);
	starpu_free(buf);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*1000000, (NITER/timing_slowness)*1000000,
					       timing_latency/NITER, timing_latency/NITER, node);
	return 1;
}
