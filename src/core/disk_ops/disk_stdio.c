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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <starpu_parameters.h>

#ifdef STARPU_HAVE_WINDOWS
#  include <io.h>
#endif

#define NITER	_starpu_calibration_minimum

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define MAX_OPEN_FILES 64
#define TEMP_HIERARCHY_DEPTH 2

/* ------------------- use STDIO to write on disk -------------------  */
static unsigned starpu_stdio_opened_files;

struct starpu_stdio_obj
{
	int descriptor;
	FILE * file;
	char * path;
	size_t size;
	starpu_pthread_mutex_t mutex;
};

struct starpu_stdio_base
{
	char * path;
	int created;
};

static struct starpu_stdio_obj *_starpu_stdio_init(int descriptor, char *path, size_t size)
{
	struct starpu_stdio_obj *obj;
	_STARPU_MALLOC(obj, sizeof(struct starpu_stdio_obj));

	FILE *f = fdopen(descriptor,"rb+");
	if (f == NULL)
	{
		free(obj);
		return NULL;
	}

	STARPU_HG_DISABLE_CHECKING(starpu_stdio_opened_files);
	if (starpu_stdio_opened_files >= MAX_OPEN_FILES)
	{
		/* Too many opened files, avoid keeping this one opened */
		fclose(f);
		f = NULL;
		descriptor = -1;
	}
	else
		(void) STARPU_ATOMIC_ADD(&starpu_stdio_opened_files, 1);

	STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	obj->descriptor = descriptor;
	obj->file = f;
	obj->path = path;
	obj->size = size;

	return (void *) obj;
}

static FILE *_starpu_stdio_reopen(struct starpu_stdio_obj *obj)
{
	int id = open(obj->path, O_RDWR);
	STARPU_ASSERT(id >= 0);

	FILE *f = fdopen(id,"rb+");
	STARPU_ASSERT(f);

	return f;
}

static void _starpu_stdio_reclose(FILE *f)
{
	fclose(f);
}

static void _starpu_stdio_close(struct starpu_stdio_obj *obj)
{
	if (obj->descriptor < 0)
		return;

	if (starpu_stdio_opened_files < MAX_OPEN_FILES)
		(void) STARPU_ATOMIC_ADD(&starpu_stdio_opened_files, -1);

	fclose(obj->file);
}

static void _starpu_stdio_fini(struct starpu_stdio_obj *obj)
{
	STARPU_PTHREAD_MUTEX_DESTROY(&obj->mutex);

	free(obj->path);
	free(obj);
}

/* allocation memory on disk */
static void *starpu_stdio_alloc(void *base, size_t size)
{
	struct starpu_stdio_obj *obj;
	struct starpu_stdio_base * fileBase = (struct starpu_stdio_base *) base;

	int id;
	char *baseCpy = _starpu_mktemp_many(fileBase->path, TEMP_HIERARCHY_DEPTH, O_RDWR | O_BINARY, &id);

	/* fail */
	if (!baseCpy)
		return NULL;

	int val = _starpu_ftruncate(id,size);
	/* fail */
	if (val < 0)
	{
		_STARPU_DISP("Could not truncate file, ftruncate failed with error '%s'\n", strerror(errno));
		close(id);
		unlink(baseCpy);
		free(baseCpy);
		return NULL;
	}

	obj = _starpu_stdio_init(id, baseCpy, size);
	if (!obj)
	{
		close(id);
		unlink(baseCpy);
		free(baseCpy);
	}

	return obj;
}

/* free memory on disk */
static void starpu_stdio_free(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) obj;

	_starpu_stdio_close(tmp);
	unlink(tmp->path);
	_starpu_rmtemp_many(tmp->path, TEMP_HIERARCHY_DEPTH);
	_starpu_stdio_fini(tmp);
}

/* open an existing memory on disk */
static void *starpu_stdio_open(void *base, void *pos, size_t size)
{
	struct starpu_stdio_base * fileBase = (struct starpu_stdio_base *) base;
	struct starpu_stdio_obj *obj;
	/* create template */
	char *baseCpy;
	_STARPU_MALLOC(baseCpy, strlen(fileBase->path)+1+strlen(pos)+1);

	snprintf(baseCpy, strlen(fileBase->path)+1+strlen(pos)+1, "%s/%s", fileBase->path, (char *)pos);

	int id = open(baseCpy, O_RDWR);
	if (id < 0)
	{
		free(baseCpy);
		return NULL;
	}

	obj = _starpu_stdio_init(id, baseCpy, size);
	if (!obj)
		free(baseCpy);
	return obj;
}

/* free memory without delete it */
static void starpu_stdio_close(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) obj;

	_starpu_stdio_close(tmp);
	_starpu_stdio_fini(tmp);
}

/* read the memory disk */
static int starpu_stdio_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) obj;
	FILE *f = tmp->file;

	if (f)
		STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);
	else
		f = _starpu_stdio_reopen(obj);

	int res = fseek(f, offset, SEEK_SET);
	STARPU_ASSERT_MSG(res == 0, "Stdio read failed");

	starpu_ssize_t nb = fread(buf, 1, size, f);
	STARPU_ASSERT_MSG(nb >= 0, "Stdio read failed");

	if (tmp->file)
		STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
	else
		_starpu_stdio_reclose(f);

	return 0;
}

static int starpu_stdio_full_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void **ptr, size_t *size, unsigned dst_node)
{
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) obj;
	FILE *f = tmp->file;
	starpu_ssize_t ssize;

	if (f)
		STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);
	else
		f = _starpu_stdio_reopen(obj);

	int res = fseek(f, 0, SEEK_END);
	STARPU_ASSERT_MSG(res == 0, "Stdio write failed");
	ssize = ftell(f);
	STARPU_ASSERT_MSG(ssize >= 0, "Stdio write failed");
	*size = ssize;

	if (tmp->file)
		STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
	/* Alloc aligned buffer */
	_starpu_malloc_flags_on_node(dst_node, ptr, *size, 0);
	if (tmp->file)
		STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	res = fseek(f, 0, SEEK_SET);
	STARPU_ASSERT_MSG(res == 0, "Stdio read failed");

	starpu_ssize_t nb = fread(*ptr, 1, *size, f);
	STARPU_ASSERT_MSG(nb >= 0, "Stdio read failed");

	if (tmp->file)
		STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
	else
		_starpu_stdio_reclose(f);

	return 0;
}

/* write on the memory disk */
static int starpu_stdio_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size)
{
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) obj;
	FILE *f = tmp->file;

	if (f)
		STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);
	else
		f = _starpu_stdio_reopen(obj);

	int res = fseek(f, offset, SEEK_SET);
	STARPU_ASSERT_MSG(res == 0, "Stdio write failed");

	fwrite(buf, 1, size, f);

	if (tmp->file)
		STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
	else
		_starpu_stdio_reclose(f);

	return 0;
}

static int starpu_stdio_full_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *ptr, size_t size)
{
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) obj;
	FILE *f = tmp->file;

	if (!f)
		f = _starpu_stdio_reopen(obj);

	/* update file size to realise the next good full_read */
	if(size != tmp->size)
	{
		int val = _starpu_fftruncate(f,size);
		STARPU_ASSERT(val == 0);

		tmp->size = size;
	}

	int res = fseek(f, 0, SEEK_SET);
	STARPU_ASSERT_MSG(res == 0, "Stdio write failed");

	fwrite(ptr, 1, size, f);

	if (!tmp->file)
		_starpu_stdio_reclose(f);

	return 0;
}

static void *starpu_stdio_plug(void *parameter, starpu_ssize_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_stdio_base * base;
	struct stat buf;

	_STARPU_MALLOC(base, sizeof(*base));
	base->created = 0;
	base->path = strdup((char *) parameter);
	STARPU_ASSERT(base->path);

	if (!(stat(base->path, &buf) == 0 && S_ISDIR(buf.st_mode)))
	{
		_starpu_mkpath(base->path, S_IRWXU);
		base->created = 1;
	}

	return (void *) base;
}

/* free memory allocated for the base */
static void starpu_stdio_unplug(void *base)
{
	struct starpu_stdio_base * fileBase = (struct starpu_stdio_base *) base;
	if (fileBase->created)
		rmdir(fileBase->path);
	free(fileBase->path);
	free(fileBase);
}

static int get_stdio_bandwidth_between_disk_and_main_ram(unsigned node, void *base)
{
	unsigned iter;
	double timing_slowness, timing_latency;
	double start;
	double end;
	char *buf;
	struct starpu_stdio_base * fileBase = (struct starpu_stdio_base *) base;

	srand(time(NULL));
	starpu_malloc_flags((void **) &buf, STARPU_DISK_SIZE_MIN, 0);
	STARPU_ASSERT(buf != NULL);

	/* allocate memory */
	void *mem = _starpu_disk_alloc(node, STARPU_DISK_SIZE_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;
	struct starpu_stdio_obj *tmp = (struct starpu_stdio_obj *) mem;

	memset(buf, 0, STARPU_DISK_SIZE_MIN);

	/* Measure upload slowness */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		FILE *f = tmp->file;

		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, STARPU_DISK_SIZE_MIN, NULL);

		if (!f)
			f = _starpu_stdio_reopen(tmp);

		/* clean cache memory */
		int res = fflush(f);
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed \n");

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(fileno(f));
#else
		res = fsync(fileno(f));
#endif
		STARPU_ASSERT_MSG(res == 0, "Slowness computation failed \n");

		if (!tmp->file)
			_starpu_stdio_reclose(f);
	}
	end = starpu_timing_now();
	timing_slowness = end - start;

	/* free memory */
	starpu_free_flags(buf, STARPU_DISK_SIZE_MIN, 0);

	starpu_malloc_flags((void**) &buf, sizeof(char), 0);
	STARPU_ASSERT(buf != NULL);

	*buf = 0;

	/* Measure latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		FILE *f = tmp->file;

		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, rand() % (STARPU_DISK_SIZE_MIN -1) , 1, NULL);

		if (!f)
			f = _starpu_stdio_reopen(tmp);

		int res = fflush(f);
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(fileno(f));
#else
		res = fsync(fileno(f));
#endif
		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");

		if (!tmp->file)
			_starpu_stdio_reclose(f);
	}
	end = starpu_timing_now();
	timing_latency = end - start;

	_starpu_disk_free(node, mem, STARPU_DISK_SIZE_MIN);
	starpu_free_flags(buf, sizeof(char), 0);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*STARPU_DISK_SIZE_MIN, (NITER/timing_slowness)*STARPU_DISK_SIZE_MIN,
			timing_latency/NITER, timing_latency/NITER, node, fileBase->path);
	return 1;
}

struct starpu_disk_ops starpu_disk_stdio_ops =
{
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
