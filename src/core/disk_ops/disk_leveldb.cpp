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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <leveldb/db.h>
#include <leveldb/options.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memory_manager.h>

#define NITER	64

/* ------------------- use leveldb to write on disk -------------------  */

struct starpu_leveldb_obj {
	char * key;
	double size;
	starpu_pthread_mutex_t mutex;
};

struct starpu_leveldb_base {
	leveldb::DB* db;
	/* if StarPU creates the leveldb */
	bool created;
};


/* allocation memory on disk */
static void * 
starpu_leveldb_alloc (void *base, size_t size)
{
	struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;
	struct starpu_leveldb_obj * obj = (struct starpu_leveldb_obj *) malloc(sizeof(struct starpu_leveldb_obj));
	STARPU_ASSERT(obj != NULL);

        STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	char * key = (char *) malloc(256*sizeof(char));
	strcpy(key, "STARPU");
	strcat(key,(char *) obj);

	/* create and add a key with a small memory */
	leveldb::Status s = base_tmp->db->Put(leveldb::WriteOptions(), key, "a");
	STARPU_ASSERT(s.ok());

	/* obj->size is the real size in the disk */
	obj->key = key;
	obj->size = sizeof(char);

	return (void *) obj;
}


/* free memory on disk */
static void
starpu_leveldb_free (void *base , void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) obj;
	struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;

	base_tmp->db->Delete(leveldb::WriteOptions(), tmp->key);

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	free(tmp->key);
	free(tmp);
}


/* open an existing memory on disk */
static void * 
starpu_leveldb_open (void *base, void *pos, size_t size)
{
	struct starpu_leveldb_obj * obj = (struct starpu_leveldb_obj *) malloc(sizeof(struct starpu_leveldb_obj));
	STARPU_ASSERT(obj != NULL);

        STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	char * key = (char *) malloc((strlen((char *) pos)+1)*sizeof(char));
	strcpy(key, (char *) pos);

	obj->key = key;	
	obj->size = size;

	return (void *) obj;
	
}


/* free memory without delete it */
static void 
starpu_leveldb_close (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) obj;

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	free(tmp->key);
	free(tmp);	
}


/* in the leveldb, we are obliged to read and to write the entire data 
 * so, we have to use buffers to have offset and size options */
static int 
starpu_leveldb_read (void *base, void *obj, void *buf, off_t offset, size_t size, void * async_channel STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) obj;
	struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;	
	
	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	/* leveldb need a string to store datas */
	std::string value;
	leveldb::Status s = base_tmp->db->Get(leveldb::ReadOptions(), tmp->key, &value);
	uintptr_t value_read = (uintptr_t)(value.c_str());

	/* use buffer */
	if(s.ok())
		memcpy(buf, (void *) (value_read+offset), size);
	else
		STARPU_ASSERT(s.ok());

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return 0;
}

static int
starpu_leveldb_full_read(unsigned node, void *base, void * obj, void ** ptr, size_t * size)
{
        struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) obj;
        struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;

	*size = tmp->size;
	*ptr = (size_t *)malloc(*size);
	return _starpu_disk_read(node, STARPU_MAIN_RAM, obj, *ptr, 0, *size, NULL);
}

/* write on the memory disk */
static int 
starpu_leveldb_write (void *base, void *obj, const void *buf, off_t offset, size_t size, void * async_channel)
{
        struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) obj;
        struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;

	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	uintptr_t buf_tmp = (uintptr_t) buf;
	void * buffer = (void *) malloc((tmp->size > size) ? tmp->size : size);

	/* we read the data */
        std::string value;

        leveldb::Status s = base_tmp->db->Get(leveldb::ReadOptions(), tmp->key, &value);
        uintptr_t value_read = (uintptr_t)(value.c_str());

        if(s.ok())
                memcpy(buffer, (void *) value_read, tmp->size);
        else
                STARPU_ASSERT(s.ok());

	/* put the new data on their new place */
	memcpy(buffer, (void *) (buf_tmp+offset), size); 

	/* and write them */
	s = base_tmp->db->Put(leveldb::WriteOptions(), tmp->key, (char *)buffer);
	STARPU_ASSERT(s.ok());	

	/* if the new size is higher than the old, we update it - first write after the alloc */
	tmp->size = (tmp->size > size) ? tmp->size : size;
	free(buffer);
	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return 0;
}

static int
starpu_leveldb_full_write (unsigned node, void * base, void * obj, void * ptr, size_t size)
{
	struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) obj;
	struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;
	
	/* update file size to realise the next good full_read */
	if(size != tmp->size)
	{
		_starpu_memory_manager_deallocate_size(tmp->size, node);
		if (_starpu_memory_manager_can_allocate_size(size, node))
			tmp->size = size;
		else
			STARPU_ASSERT_MSG(0, "Can't allocate size %u on the disk !", (int) size); 
	}	
	leveldb::WriteOptions write_options;
	write_options.sync = true;

        leveldb::Status s = base_tmp->db->Put(write_options, tmp->key, (char *)ptr);
	STARPU_ASSERT(s.ok());
	return 0;
}


/* create a new copy of parameter == base */
static void * 
starpu_leveldb_plug (void *parameter)
{
	struct starpu_leveldb_base * tmp = (struct starpu_leveldb_base *) malloc(sizeof(struct starpu_leveldb_base));
	STARPU_ASSERT(tmp != NULL);

	leveldb::Status status;
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	
	/* try to create the database */
	options.error_if_exists = true;
	status = leveldb::DB::Open(options, (char *) parameter, &db);
	tmp->created = true;
	
	/* if it has already been created  before */
	if (!status.ok())
	{
		options.error_if_exists = false;
		status = leveldb::DB::Open(options, (char *) parameter, &db);
                STARPU_ASSERT_MSG(status.ok(), "StarPU leveldb plug failed !");
		tmp->created = false;
	}

	tmp->db = db;
	STARPU_ASSERT(status.ok());
	return (void *) tmp;	
}


/* free memory allocated for the base */
static void
starpu_leveldb_unplug (void *base)
{
	struct starpu_leveldb_base * base_tmp = (struct starpu_leveldb_base *) base;
	if(base_tmp->created)
		delete base_tmp->db;
	free(base);
}


static int
get_leveldb_bandwidth_between_disk_and_main_ram(unsigned node)
{

	unsigned iter;
	double timing_slowness, timing_latency;
	struct timeval start;
	struct timeval end;
	
	srand (time (NULL)); 
	char * buf = (char *) malloc(SIZE_DISK_MIN*sizeof(char));
	STARPU_ASSERT(buf != NULL);
	
	/* allocate memory */
	void * mem = _starpu_disk_alloc(node, SIZE_DISK_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;
	struct starpu_leveldb_obj * tmp = (struct starpu_leveldb_obj *) mem;

	/* Measure upload slowness */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, SIZE_DISK_MIN, NULL);
	}
	gettimeofday(&end, NULL);
	timing_slowness = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));


	/* free memory */
	free(buf);

	buf = (char *) malloc(sizeof(char));
	STARPU_ASSERT(buf != NULL);

	/* Measure latency */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, rand() % (SIZE_DISK_MIN -1) , 1, NULL);
	}
	gettimeofday(&end, NULL);
	timing_latency = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	_starpu_disk_free(node, mem, SIZE_DISK_MIN);
	free(buf);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*1000000, (NITER/timing_slowness)*1000000,
					       timing_latency/NITER, timing_latency/NITER, node);
	return 1;
}

#if __cplusplus >= 201103L
struct starpu_disk_ops starpu_disk_leveldb_ops = {
	.alloc = starpu_leveldb_alloc,
	.free = starpu_leveldb_free,
	.open = starpu_leveldb_open,
	.close = starpu_leveldb_close,
	.read = starpu_leveldb_read,
	.write = starpu_leveldb_write,
	.async_write = NULL,
	.async_read = NULL,
	.plug = starpu_leveldb_plug,
	.unplug = starpu_leveldb_unplug,
	.copy = NULL,
	.bandwidth = get_leveldb_bandwidth_between_disk_and_main_ram,
	.wait_request = NULL,
	.test_request = NULL,
	.full_read = starpu_leveldb_full_read,
	.full_write = starpu_leveldb_full_write
};
#else
struct starpu_disk_ops starpu_disk_leveldb_ops = {
	starpu_leveldb_alloc,
	starpu_leveldb_free,
	starpu_leveldb_open,
	starpu_leveldb_close,
	starpu_leveldb_read,
	starpu_leveldb_write,
	NULL,
	NULL,
	starpu_leveldb_plug,
	starpu_leveldb_unplug,
	NULL,
	get_leveldb_bandwidth_between_disk_and_main_ram,
	NULL,
	NULL,
	starpu_leveldb_full_read,
	starpu_leveldb_full_write
};
#endif
