/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <hdfs.h>
#include "starpujni-hdfs.h"

#define TMPPREFIX "yeah"
#define HDFS_NAMENODE "default"
#define HDFS_PORT 0

#define S(fmt_, ...) STARPUJNI_TRACE("START " fmt_,  ## __VA_ARGS__)

#define E(fmt_, ...)  STARPUJNI_TRACE("END " fmt_,  ## __VA_ARGS__)

#define HDFS_ERRMSG(fmt_, ...) ERROR_MSG("hdfs error: " fmt_, ## __VA_ARGS__)
#define HDFS_ERRMSG_GOTO(label_, fmt_, ...) ERROR_MSG_GOTO(label_, "hdfs error: " fmt_, ##  __VA_ARGS__)

struct _starpu_async_channel;

unsigned _starpu_calibration_minimum;
extern void _starpu_save_bandwidth_and_latency_disk(double bandwidth_write,
						    double bandwidth_read,
						    double latency_write,
						    double latency_read, unsigned node,
						    const char *name);
extern void _starpu_disk_free(int devid, void *obj, size_t size);
extern int _starpu_disk_write(int src_dev, int dst_dev, void *obj, void *buf,
			      off_t offset, size_t size,
			      struct _starpu_async_channel *async_channel);
extern void *_starpu_disk_alloc(int devid, size_t size);

#define NITER _starpu_calibration_minimum

struct hdfs_file
{
	char *path;
	size_t path_size;
	size_t file_size;
	starpu_pthread_mutex_t mutex;
};

struct hdfs_base
{
	char *path;
	size_t path_size;
	hdfsFS fs;
	int created;
};

static void s_hdfs_destroy(struct hdfs_base *hdfs);

static struct hdfs_file *s_hdfs_open_or_create_file(struct hdfs_base *hdfs, const char *filename, size_t size);

static struct hdfs_file *s_hdfs_mktemp(struct hdfs_base *hdfs, size_t size);

static void s_hdfs_file_destroy(struct hdfs_base *hdfs, struct hdfs_file *file);

static hdfsFile s_hdfs_file_get_input_stream(struct hdfs_base *hdfs, struct hdfs_file *file, ssize_t offset);

static starpu_ssize_t s_hdfs_file_get_size(struct hdfs_base *hdfs, const char *path);

static int s_hdfs_write(struct hdfs_base *hdfs, const char *path, hdfsFile output, const void *buf, size_t size);

/*
 * TODO: implement a better handling of errors ibstead of assertions.
 */
static void *s_starpujni_disk_plug(void *parameter, starpu_ssize_t size)
{
	struct hdfs_base *hdfs;

	_starpu_calibration_minimum = starpu_getenv_number_default("STARPU_CALIBRATE_MINIMUM", 10);

	S("%s", (char *) parameter);

	starpu_malloc((void **) &hdfs, sizeof(*hdfs));
	if (hdfs == NULL)
		HDFS_ERRMSG_GOTO(err, "can't allocate HDFS disk.");

	hdfs->path = strdup(parameter);
	if (hdfs->path == NULL)
		HDFS_ERRMSG_GOTO(err, "can't allocate HDFS disk.");

	hdfs->fs = hdfsConnect(HDFS_NAMENODE, HDFS_PORT);
	if (hdfs->fs == NULL)
		HDFS_ERRMSG_GOTO(err, "can't connect to hdfs namenode %s:%d.",
				 HDFS_NAMENODE, HDFS_PORT);
	hdfs->created =(hdfsCreateDirectory(hdfs->fs, hdfs->path) == 0);
	if (!hdfs->created && hdfsExists(hdfs->fs, hdfs->path))
	{
		hdfsFileInfo *fileInfo = hdfsGetPathInfo(hdfs->fs, hdfs->path);
		if (fileInfo != NULL)
		{
			hdfs->created =(fileInfo->mKind == kObjectKindDirectory);
			fprintf(stderr, "hdfs warning: disk directory %s already exists.",
				hdfs->path);
			hdfsFreeFileInfo(fileInfo, 1);
		}
	}
	if (!hdfs->created)
		HDFS_ERRMSG_GOTO(err, "can't create directory %s.", hdfs->path);
	E("%s", (char *) parameter);
	return hdfs;
err:
	s_hdfs_destroy(hdfs);
	return NULL;
}

static void s_starpujni_disk_unplug(void *base)
{
	struct hdfs_base *hdfs = base;
	S("%s", hdfs->path);
	s_hdfs_destroy(hdfs);
	E("");
}

/*
 * Copy/paste from StarPU source code: src/core/disk_op/disk_hdf5.c
 */
static int s_starpujni_disk_bandwidth(unsigned node, void *base)
{
	S("");

	unsigned iter;
	double timing_slowness, timing_latency;
	double start;
	double end;
	char *buf;
	struct hdfs_base *fileBase = (struct hdfs_base *) base;

	srand(time(NULL));
	starpu_malloc_flags((void **) &buf, STARPU_DISK_SIZE_MIN, 0);
	STARPU_ASSERT(buf != NULL);

	/* allocate memory */
	int devid = starpu_memory_node_get_devid(node);
	void *mem = _starpu_disk_alloc(devid, STARPU_DISK_SIZE_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;

	memset(buf, 0, STARPU_DISK_SIZE_MIN);

	S("writing NITER=%d", NITER);
	/* Measure upload slowness */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(0, devid, mem, buf, 0, STARPU_DISK_SIZE_MIN, NULL);
	}
	end = starpu_timing_now();
	timing_slowness = end - start;
	E("writing");
	/* free memory */
	starpu_free_flags(buf, STARPU_DISK_SIZE_MIN, 0);

	starpu_malloc_flags((void **) &buf, sizeof(char), 0);
	STARPU_ASSERT(buf != NULL);

	*buf = 0;

	/* Measure latency */
	S("latency measuring NITER=%d", NITER);
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(0, devid, mem, buf,
				    rand() %(STARPU_DISK_SIZE_MIN - 1), 1, NULL);
	}
	end = starpu_timing_now();
	E("latency measure");
	timing_latency = end - start;

	_starpu_disk_free(devid, mem, STARPU_DISK_SIZE_MIN);
	starpu_free_flags(buf, sizeof(char), 0);

	_starpu_save_bandwidth_and_latency_disk((NITER / timing_slowness) * STARPU_DISK_SIZE_MIN,
						(NITER / timing_slowness) * STARPU_DISK_SIZE_MIN,
						 timing_latency / NITER, timing_latency / NITER, node, fileBase->path);
	E("");
	return 1;
}

static void *s_starpujni_disk_alloc(void *base, size_t size)
{
	S("");
	struct hdfs_base *hdfs = base;
	struct hdfs_file *obj = s_hdfs_mktemp(hdfs, size);
	E("%s", obj->path);

	return obj;
}

static void s_starpujni_disk_free(void *base, void *obj, size_t size)
{
	S("");
	struct hdfs_base *hdfs = base;
	struct hdfs_file *file = obj;
	s_hdfs_file_destroy(hdfs, file);
	E("");
}

static void *s_starpujni_disk_open(void *base, void *pos, size_t size)
{
	S("%s ", (char *) pos);
	struct hdfs_base *hdfs = base;
	struct hdfs_file *obj = s_hdfs_open_or_create_file(hdfs, pos, size);
	E("%s ", (char *) pos);
	return obj;
}

static void s_starpujni_disk_close(void *base, void *obj, size_t size)
{
	S("%s ",((struct hdfs_file *) obj)->path);
	struct hdfs_base *hdfs = base;
	s_hdfs_file_destroy(hdfs, obj);
	E("");
}

static int s_starpujni_disk_read(void *base, void *obj, void *buf, off_t offset, size_t size)
{
	starpu_ssize_t nb;
	struct hdfs_base *hdfs = base;
	struct hdfs_file *file = obj;
	hdfsFile is;

	S("%s offset=%ld size=%ld", file->path, (long) offset, size);

	STARPU_PTHREAD_MUTEX_LOCK(&file->mutex);
	is = s_hdfs_file_get_input_stream(hdfs, file, offset);
	if (is != NULL)
	{
		nb = hdfsRead(hdfs->fs, is, buf, size);
		if (nb < 0)
			HDFS_ERRMSG("while reading %lu bytes from offset %ld in %s.",
				     size, (long) offset, file->path);
		hdfsCloseFile(hdfs->fs, is);
	}
	else
		nb = -1;
	STARPU_PTHREAD_MUTEX_UNLOCK(&file->mutex);
	E("%s offset=%ld size=%ld", file->path, (long) offset, size);

	return nb;
}

static int s_copy(struct hdfs_base *base, const char *dstPath, hdfsFile dst, hdfsFile src, off_t offset, size_t size)
{
	void *buffer = NULL;
	int res = 0;
	size_t remain = size;
	S("offset=%ld size=%lu", (long) offset, size);

	starpu_malloc_flags(&buffer, size, 0);
	if (buffer == NULL)
		HDFS_ERRMSG_GOTO(err, "can't allocate copy buffer of size %lu.", size);
	if (hdfsSeek(base->fs, src, offset) < 0)
		HDFS_ERRMSG_GOTO(err, "can't seek at %ld in input file.", (long) offset);
	while (remain > 0)
	{
		ssize_t len = hdfsRead(base->fs, src, buffer, remain);
		if (len < 0)
			HDFS_ERRMSG_GOTO(err, "while reading %ld bytes.", remain);
		STARPU_ASSERT((ssize_t) len <= remain);
		remain -= len;
	}

	if (s_hdfs_write(base, dstPath, dst, buffer, size) != size)
		HDFS_ERRMSG_GOTO(err, "while writing %ld bytes.", size);
	res = 1;
	E("offset=%ld size=%lu", (long) offset, size);

err:
	if (buffer != NULL)
		starpu_free_flags(buffer, size, 0);

	return res;
}

static int s_starpujni_disk_write(void *base, void *obj, const void *buf, off_t offset, size_t size)
{
	struct hdfs_base *hdfs = base;
	struct hdfs_file *file = obj;
	S("%s offset=%ld size=%lu", file->path, (long) offset, size);
	struct hdfs_file *tmp = s_hdfs_mktemp(hdfs, STARPU_MAX(offset + size, file->file_size));
	hdfsFile tmpos;
	hdfsFile is = NULL;
	size_t prefixlen = (size_t) offset;
	size_t suffixlen = file->file_size - offset - size;

	STARPU_ASSERT(offset >= 0);
	STARPU_ASSERT(hdfsExists(hdfs->fs, file->path) >= 0);
	STARPU_ASSERT(s_hdfs_file_get_size(hdfs, file->path) == file->file_size);

	if (tmp == NULL)
	{
		HDFS_ERRMSG("can't create tmp file for writing in %s.", file->path);
		return 1;
	}
	tmpos = hdfsOpenFile(hdfs->fs, tmp->path, O_WRONLY, 0, 0, 0);
	if (tmpos == NULL)
		HDFS_ERRMSG_GOTO(err, "can't open tmp file for writing in %s.", tmp->path);

	STARPU_PTHREAD_MUTEX_LOCK(&file->mutex);
	is = hdfsOpenFile(hdfs->fs, file->path, O_RDONLY, 0, 0, 0);
	if (is == NULL)
		HDFS_ERRMSG_GOTO(err, "can't open file '%s' for reading.", file->path);

	if (prefixlen > 0 && !s_copy(hdfs, tmp->path, tmpos, is, 0, prefixlen))
		HDFS_ERRMSG_GOTO(err, "while copying prefix of size %ld from %s into %s.",
				 prefixlen, tmp->path, file->path);

	if (s_hdfs_write(hdfs, tmp->path, tmpos, buf, size) != size)
		HDFS_ERRMSG_GOTO(err, "error while writing input buffer into %s.",
				 file->path);

	if (suffixlen > 0 && !s_copy(hdfs, tmp->path, tmpos, is, offset + size, suffixlen))
		HDFS_ERRMSG_GOTO(err, "while copying suffix of size %ld from %s into %s.",
				  suffixlen, file->path, tmp->path);
	hdfsFlush(hdfs->fs, tmpos);
	hdfsCloseFile(hdfs->fs, tmpos);
	tmpos = NULL;
	hdfsCloseFile(hdfs->fs, is);
	is = NULL;
	if (s_hdfs_file_get_size(hdfs, tmp->path) != file->file_size)
		HDFS_ERRMSG_GOTO(err, "inconsistent size of files %s and %s before "
				  "copy.", tmp->path, file->path);

	if (hdfsCopy(hdfs->fs, tmp->path, hdfs->fs, file->path) < 0)
		HDFS_ERRMSG_GOTO(err, "error when copying tmp file %s onto %s.",
				  tmp->path, file->path);
	hdfsDelete(hdfs->fs, tmp->path, 0);
	s_hdfs_file_destroy(hdfs, tmp);

	STARPU_PTHREAD_MUTEX_UNLOCK(&file->mutex);
	E("%s", file->path);

	STARPU_ASSERT(file->file_size == s_hdfs_file_get_size(hdfs, file->path));

	return 0;

err:
	hdfsDelete(hdfs->fs, tmp->path, 0);
	s_hdfs_file_destroy(hdfs, tmp);
	if (tmpos != NULL)
		hdfsCloseFile(hdfs->fs, tmpos);
	if (is != NULL)
		hdfsCloseFile(hdfs->fs, is);
	STARPU_PTHREAD_MUTEX_UNLOCK(&file->mutex);
	E("%s", file->path);

	return 1;
}

static int s_starpujni_disk_full_read(void *base, void *obj, void **ptr, size_t *size, unsigned dst_node)
{
	struct hdfs_base *hdfs = base;
	struct hdfs_file *file = obj;
	starpu_ssize_t ssize = s_hdfs_file_get_size(hdfs, file->path);

	if (ssize < 0)
		return -1;
	if (ssize != file->file_size)
	{
		HDFS_ERRMSG("inconsistent size %lu (expected size = %lu) for file %s",
			     ssize, file->file_size, file->path);
		return -1;
	}

	*ptr = (void *) starpu_malloc_on_node_flags(dst_node, *size, 0);
	if (*ptr == NULL)
	{
		HDFS_ERRMSG("can't allocate buffer on node %d to read %s.", dst_node,
			    file->path);
		return -1;
	}
	*size = ssize;
	if ((ssize = s_starpujni_disk_read(base, obj, *ptr, 0, *size)) < 0)
	{
		starpu_free_on_node_flags(dst_node, (uintptr_t) *ptr, ssize, 0);
		*ptr = NULL;
		*size = 0;
		HDFS_ERRMSG("can't read %lu bytes from %s.", ssize, file->path);
	}
	return ssize;
}

static int s_starpujni_disk_full_write(void *base, void *obj, void *ptr, size_t size)
{
	int res = -1;
	hdfsFile output;
	struct hdfs_base *hdfs = base;
	struct hdfs_file *file = obj;

	STARPU_PTHREAD_MUTEX_LOCK(&file->mutex);
	output = hdfsOpenFile(hdfs->fs, file->path, O_WRONLY, 0, 0, 0);
	if (output == NULL)
		HDFS_ERRMSG("can't open file '%s' for writing.", file->path);
	else
	{
		if (s_hdfs_write(hdfs, file->path, output, ptr, size) == size)
			res = 0;
		hdfsCloseFile(hdfs->fs, output);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&file->mutex);

	return res;
}

#define s_starpujni_disk_async_write NULL
#define s_starpujni_disk_async_read NULL
#define s_starpujni_disk_async_full_read NULL
#define s_starpujni_disk_async_full_write NULL
#define s_starpujni_disk_copy NULL
#define s_starpujni_disk_wait_request NULL
#define s_starpujni_disk_test_request NULL
#define s_starpujni_disk_free_request NULL

#define STARPUJNI_DISK_OP(_op_) ._op_ = s_starpujni_disk_ ## _op_

struct starpu_disk_ops starpujni_disk_hdfs_ops =
{
	STARPUJNI_DISK_OP(plug),
	STARPUJNI_DISK_OP(unplug),
	STARPUJNI_DISK_OP(bandwidth),
	STARPUJNI_DISK_OP(alloc),
	STARPUJNI_DISK_OP(free),
	STARPUJNI_DISK_OP(open),
	STARPUJNI_DISK_OP(close),
	STARPUJNI_DISK_OP(read),
	STARPUJNI_DISK_OP(write),
	STARPUJNI_DISK_OP(full_read),
	STARPUJNI_DISK_OP(full_write),
	STARPUJNI_DISK_OP(async_write),
	STARPUJNI_DISK_OP(async_read),
	STARPUJNI_DISK_OP(async_full_read),
	STARPUJNI_DISK_OP(async_full_write),
	STARPUJNI_DISK_OP(copy),
	STARPUJNI_DISK_OP(wait_request),
	STARPUJNI_DISK_OP(test_request),
	STARPUJNI_DISK_OP(free_request)
};

static void s_hdfs_destroy(struct hdfs_base *hdfs)
{
	if (hdfs == NULL)
		return;
	if (hdfs->path != NULL)
		starpu_free_noflag(hdfs->path, hdfs->path_size);
	if (hdfs->fs != NULL)
	{
		//    if (hdfs->created)
		//        hdfsDelete(hdfs->fs, hdfs->path, 1);
		hdfsDisconnect(hdfs->fs);
	}
	starpu_free_noflag(hdfs, sizeof(*hdfs));
}

static struct hdfs_file *s_hdfs_allocate_file(struct hdfs_base *hdfs, char *path, size_t size)
{
	struct hdfs_file *result;

	starpu_malloc((void **) &result, sizeof(*result));
	if (result != NULL)
	{
		result->path = path;
		result->file_size = size;
		STARPU_PTHREAD_MUTEX_INIT(&result->mutex, NULL);
	}
	else
	{
		hdfsDelete(hdfs->fs, path, 0);
		starpu_free_noflag(path, hdfs->path_size);
	}

	return result;
}

static char *s_hdfs_create_or_resize_file(struct hdfs_base *hdfs, const char *filename, size_t size)
{
	char *path;
	void *buf = NULL;
	hdfsFile writefile = NULL;

	hdfs->path_size = strlen(hdfs->path) + sizeof('/') + strlen(filename) + sizeof('\0');
	starpu_malloc((void **) &path, hdfs->path_size);
	if (path == NULL)
		return NULL;

	sprintf(path, "%s/%s", hdfs->path, filename);

	if (hdfsExists(hdfs->fs, path) == 0 &&
	    s_hdfs_file_get_size(hdfs, path) == size)
		goto end;

	writefile = hdfsOpenFile(hdfs->fs, path, O_WRONLY, 0, 0, 0);
	if (writefile == NULL)
		HDFS_ERRMSG_GOTO(err, "can't open file %s for writing.", path);

	if (size > 0)
	{
		starpu_malloc(&buf, size);
		if (buf == NULL)
			HDFS_ERRMSG_GOTO(err, "can't allocate buffer to create %s.", path);
		memset(buf, 0, size);
		if (s_hdfs_write(hdfs, filename, writefile, buf, size) != size)
			HDFS_ERRMSG_GOTO(err, "can't resize file %s to %ld bytes.", path,
					  size);
		starpu_free_noflag(buf, size);
	}
	hdfsCloseFile(hdfs->fs, writefile);

end:
	return path;

err:
	starpu_free_noflag(path, hdfs->path_size);
	if (buf != NULL)
		starpu_free_noflag(buf, size);
	if (writefile != NULL)
	{
		hdfsCloseFile(hdfs->fs, writefile);
		hdfsDelete(hdfs->fs, path, 0);
	}
	return NULL;
}

static struct hdfs_file *s_hdfs_open_or_create_file(struct hdfs_base *hdfs, const char *filename, size_t size)
{
	char *path = s_hdfs_create_or_resize_file(hdfs, filename, size);

	if (path == NULL)
		return NULL;

	STARPU_ASSERT(s_hdfs_file_get_size(hdfs, path) == size);

	return s_hdfs_allocate_file(hdfs, path, size);
}

static struct hdfs_file *s_hdfs_mktemp(struct hdfs_base *hdfs, size_t size)
{
	S("");
	char *path;
	long suffix;
	struct hdfs_file *result;

	hdfs->path_size = strlen(hdfs->path) + sizeof('/') + sizeof(long) * 2 + strlen(TMPPREFIX) + sizeof('\0');

	starpu_malloc((void **) &path, hdfs->path_size);
	if (path == NULL)
		return NULL;

	do
	{
		suffix = lrand48();
		sprintf(path, "%s/%s%lx", hdfs->path, TMPPREFIX, suffix);
	}
	while (hdfsExists(hdfs->fs, path) == 0);

	sprintf(path, "%s%lx", TMPPREFIX, suffix);
	result = s_hdfs_open_or_create_file(hdfs, path, size);
	starpu_free_noflag(path, hdfs->path_size);
	E("");

	return result;
}

static void s_hdfs_file_destroy(struct hdfs_base *hdfs, struct hdfs_file *file)
{
	STARPU_PTHREAD_MUTEX_DESTROY(&file->mutex);
	starpu_free_noflag(file->path, hdfs->path_size);
	starpu_free_noflag(file, sizeof(*file));
}

static hdfsFile s_hdfs_file_get_input_stream(struct hdfs_base *hdfs, struct hdfs_file *file, ssize_t offset)
{
	STARPU_ASSERT(offset >= 0);
	STARPU_ASSERT(hdfsExists(hdfs->fs, file->path));
	hdfsFile is = hdfsOpenFile(hdfs->fs, file->path, O_RDONLY, 0, 0, 0);

	if (is == NULL)
	{
		HDFS_ERRMSG("can't read file '%s'.", file->path);
		return NULL;
	}
	if (hdfsSeek(hdfs->fs, is, offset) == 0)
		return is;

	hdfsCloseFile(hdfs->fs, is);
	HDFS_ERRMSG("can't seek file %s to position %ld.", file->path, offset);

	return NULL;
}

static starpu_ssize_t s_hdfs_file_get_size(struct hdfs_base *hdfs, const char *path)
{
	starpu_ssize_t result;
	hdfsFileInfo *fileInfo = hdfsGetPathInfo(hdfs->fs, path);

	if (fileInfo == NULL)
	{
		HDFS_ERRMSG("hdfs error: can't get size of file %s.", path);
		result = -1;
	}
	else
	{
		result = fileInfo->mSize;
		hdfsFreeFileInfo(fileInfo, 1);
	}
	return result;
}

static int s_hdfs_write(struct hdfs_base *hdfs, const char *path, hdfsFile output, const void *buf, size_t size)
{
	S("%s size=%ld", path, size);
	int result = hdfsWrite(hdfs->fs, output, buf, size);

	if (result != size)
	{
		HDFS_ERRMSG("while writing %lu bytes in %s.", size, path);
		result = -1;
	}
	if (hdfsHSync(hdfs->fs, output) < 0)
		result = -1;
	if (hdfsHFlush(hdfs->fs, output) < 0)
		result = -1;
	if (hdfsFlush(hdfs->fs, output) < 0)
		result = -1;
	E("%s size=%ld", path, size);

	return result;
}
