/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Corentin Salingue
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
#include <common/config.h>
#include <leveldb/db.h>
#include <leveldb/options.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memory_manager.h>
#include <starpu_parameters.h>

#define NITER	_starpu_calibration_minimum

/* ------------------- use leveldb to write on disk -------------------  */

struct starpu_leveldb_obj
{
	char * key;
	size_t size;
	starpu_pthread_mutex_t mutex;
};

struct starpu_leveldb_base
{
	char *path;
	leveldb::DB* db;
	/* if StarPU creates the leveldb */
	bool created;
};

/* allocation memory on disk */
static void *starpu_leveldb_alloc(void *base, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;
	struct starpu_leveldb_obj *obj = (struct starpu_leveldb_obj *)malloc(sizeof(struct starpu_leveldb_obj));
	STARPU_ASSERT(obj);

        STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	size_t len = 6 + 1 + 2+sizeof(void*)*2 + 1;
	char *key = (char *)malloc(len*sizeof(char));
	STARPU_ASSERT(key);
	snprintf(key, len, "STARPU-%p", obj);

	/* create and add a key with a small memory */
	leveldb::Status s = base_tmp->db->Put(leveldb::WriteOptions(), key, "a");
	STARPU_ASSERT(s.ok());

	/* obj->size is the real size in the disk */
	obj->key = key;
	obj->size = sizeof(char);

	return (void *) obj;
}

/* free memory on disk */
static void starpu_leveldb_free(void *base , void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_obj *tmp = (struct starpu_leveldb_obj *) obj;
	struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;

	base_tmp->db->Delete(leveldb::WriteOptions(), tmp->key);

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	free(tmp->key);
	free(tmp);
}

/* open an existing memory on disk */
static void *starpu_leveldb_open(void *base STARPU_ATTRIBUTE_UNUSED, void *pos, size_t size)
{
	struct starpu_leveldb_obj *obj = (struct starpu_leveldb_obj *)malloc(sizeof(struct starpu_leveldb_obj));
	STARPU_ASSERT(obj);

        STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	obj->key = strdup((char*) pos);
	obj->size = size;

	return (void *) obj;
}

/* free memory without delete it */
static void starpu_leveldb_close(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_obj *tmp = (struct starpu_leveldb_obj *) obj;

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	free(tmp->key);
	free(tmp);
}

/* in the leveldb, we are obliged to read and to write the entire data
 * so, we have to use buffers to have offset and size options */
static int starpu_leveldb_read(void *base, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_leveldb_obj *tmp = (struct starpu_leveldb_obj *) obj;
	struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;

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

static int starpu_leveldb_full_read(void *base, void *obj, void **ptr, size_t *size, unsigned dst_node)
{
        struct starpu_leveldb_obj *tmp = (struct starpu_leveldb_obj *) obj;
        struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;

	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	/* leveldb need a string to store datas */
	std::string value;
	leveldb::Status s = base_tmp->db->Get(leveldb::ReadOptions(), tmp->key, &value);

	STARPU_ASSERT(s.ok());

	*size = value.length();
	_starpu_malloc_flags_on_node(dst_node, ptr, *size, 0);
	STARPU_ASSERT(*ptr);

	/* use buffer */
	memcpy(*ptr, value.c_str(), *size);

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return 0;
}

/* write on the memory disk */
static int starpu_leveldb_write(void *base, void *obj, const void *buf, off_t offset, size_t size)
{
        struct starpu_leveldb_obj *tmp = (struct starpu_leveldb_obj *) obj;
        struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;
	void *buffer;
	leveldb::Status s;

	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	if (offset == 0 && size >= tmp->size)
	{
		/* We overwrite everything, no need to get the old value */
		buffer = (void*) buf;
	}
	else
	{
		uintptr_t buf_tmp = (uintptr_t) buf;
		buffer = malloc((tmp->size > (offset + size)) ? tmp->size : (offset + size));
		STARPU_ASSERT(buffer);

		/* we read the data */
		std::string value;

		s = base_tmp->db->Get(leveldb::ReadOptions(), tmp->key, &value);
		uintptr_t value_read = (uintptr_t)(value.c_str());

		STARPU_ASSERT(s.ok());
		memcpy(buffer, (void *) value_read, tmp->size);

		/* put the new data on their new place */
		memcpy((void *) ((uintptr_t) buffer + offset), (void *) buf_tmp, size);
	}

	/* and write them */
	s = base_tmp->db->Put(leveldb::WriteOptions(), tmp->key, (char *)buffer);
	STARPU_ASSERT(s.ok());

	/* if the new size is higher than the old, we update it - first write after the alloc */
	tmp->size = (tmp->size > size) ? tmp->size : size;
	if (buffer != buf)
		free(buffer);
	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return 0;
}

static int starpu_leveldb_full_write(void *base, void *obj, void *ptr, size_t size)
{
	struct starpu_leveldb_obj *tmp = (struct starpu_leveldb_obj *) obj;
	struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;

	/* update file size to achieve correct writes */
	tmp->size = size;

	leveldb::WriteOptions write_options;
	write_options.sync = true;

        leveldb::Status s = base_tmp->db->Put(write_options, tmp->key, (char *)ptr);
	STARPU_ASSERT(s.ok());
	return 0;
}

/* create a new copy of parameter == base */
static void *starpu_leveldb_plug(void *parameter, starpu_ssize_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_leveldb_base *tmp = (struct starpu_leveldb_base *)malloc(sizeof(struct starpu_leveldb_base));
	STARPU_ASSERT(tmp);

	leveldb::Status status;
	leveldb::DB *db;
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
	tmp->path = strdup((const char*) parameter);
	STARPU_ASSERT(status.ok());
	return (void *) tmp;
}

/* free memory allocated for the base */
static void starpu_leveldb_unplug(void *base)
{
	struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;
	if(base_tmp->created)
		delete base_tmp->db;
	free(base_tmp->path);
	free(base);
}

static int get_leveldb_bandwidth_between_disk_and_main_ram(unsigned node, void *base)
{
	unsigned iter;
	double timing_slowness, timing_latency;
	double start;
	double end;
        struct starpu_leveldb_base *base_tmp = (struct starpu_leveldb_base *) base;

	srand(time (NULL));
	char *buf = (char *)malloc(STARPU_DISK_SIZE_MIN*sizeof(char));
	STARPU_ASSERT(buf);

	/* allocate memory */
	void *mem = _starpu_disk_alloc(node, STARPU_DISK_SIZE_MIN);
	/* fail to alloc */
	if (mem == NULL)
	{
		free(buf);
		return 0;
	}

	/* Measure upload slowness */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, STARPU_DISK_SIZE_MIN, NULL);
	}
	end = starpu_timing_now();
	timing_slowness = end - start;


	/* free memory */
	free(buf);

	buf = (char *)malloc(sizeof(char));
	STARPU_ASSERT(buf);

	/* Measure latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, rand() % (STARPU_DISK_SIZE_MIN -1) , 1, NULL);
	}
	end = starpu_timing_now();
	timing_latency = end - start;

	_starpu_disk_free(node, mem, STARPU_DISK_SIZE_MIN);
	free(buf);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*STARPU_DISK_SIZE_MIN, (NITER/timing_slowness)*STARPU_DISK_SIZE_MIN,
			timing_latency/NITER, timing_latency/NITER, node, base_tmp->path);
	return 1;
}

#if __cplusplus >= 201103L
struct starpu_disk_ops starpu_disk_leveldb_ops =
{
	.plug = starpu_leveldb_plug,
	.unplug = starpu_leveldb_unplug,
	.bandwidth = get_leveldb_bandwidth_between_disk_and_main_ram,
	.alloc = starpu_leveldb_alloc,
	.free = starpu_leveldb_free,
	.open = starpu_leveldb_open,
	.close = starpu_leveldb_close,
	.read = starpu_leveldb_read,
	.write = starpu_leveldb_write,
	.full_read = starpu_leveldb_full_read,
	.full_write = starpu_leveldb_full_write,
	.async_write = NULL,
	.async_read = NULL,
	.async_full_read = NULL,
	.async_full_write = NULL,
	.copy = NULL,
	.wait_request = NULL,
	.test_request = NULL,
	.free_request = NULL
};
#else
struct starpu_disk_ops starpu_disk_leveldb_ops =
{
	starpu_leveldb_plug,
	starpu_leveldb_unplug,
	get_leveldb_bandwidth_between_disk_and_main_ram,
	starpu_leveldb_alloc,
	starpu_leveldb_free,
	starpu_leveldb_open,
	starpu_leveldb_close,
	starpu_leveldb_read,
	starpu_leveldb_write,
	starpu_leveldb_full_read,
	starpu_leveldb_full_write,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL
};
#endif
