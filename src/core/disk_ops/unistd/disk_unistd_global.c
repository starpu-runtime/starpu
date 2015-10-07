/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013 Corentin Salingue
 * Copyright (C) 2015 CNRS
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
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>
#ifdef HAVE_AIO_H
#include <aio.h>
#endif
#include <errno.h>

#include <common/config.h>
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif
#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <core/disk_ops/unistd/disk_unistd_global.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memory_manager.h>
#include <starpu_parameters.h>

#ifdef STARPU_HAVE_WINDOWS
#  include <io.h>
#endif

#define NITER	_starpu_calibration_minimum

#ifdef O_DIRECT
#  define MEM_SIZE getpagesize()
#else
#  define MEM_SIZE 1
#endif

/* TODO: on Linux, use io_submit */

/* ------------------- use UNISTD to write on disk -------------------  */

/* allocation memory on disk */
void *starpu_unistd_global_alloc(struct starpu_unistd_global_obj *obj, void *base, size_t size)
{
	int id = -1;
	const char *template = "STARPU_XXXXXX";

	/* create template for mkstemp */
	unsigned int sizeBase = strlen(base) + 1 + strlen(template)+1;

	char *baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);

	strcpy(baseCpy, (char *) base);
	strcat(baseCpy,"/");
	strcat(baseCpy,template);
#if defined(STARPU_HAVE_WINDOWS)
	/* size in windows is a multiple of char */
	_mktemp(baseCpy);
	id = open(baseCpy, obj->flags);
#elif defined (HAVE_MKOSTEMP)
	id = mkostemp(baseCpy, obj->flags);
#else
	STARPU_ASSERT(obj->flags == (O_RDWR | O_BINARY));
	id = mkstemp(baseCpy);
#endif

	/* fail */
	if (id < 0)
	{
		_STARPU_DISP("Could not create temporary file in directory '%s', mskostemp failed with error '%s'\n", (char*)base, strerror(errno));
		free(baseCpy);
		free(obj);
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
#ifdef STARPU_HAVE_WINDOWS
		_STARPU_DISP("Could not truncate file, _chsize failed with error '%s'\n", strerror(errno));
#else
		_STARPU_DISP("Could not truncate file, ftruncate failed with error '%s'\n", strerror(errno));
#endif
		close(id);
		unlink(baseCpy);
		free(baseCpy);
		free(obj);
		return NULL;
	}

	STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	obj->descriptor = id;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
}

/* free memory on disk */
void starpu_unistd_global_free(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	close(tmp->descriptor);
	unlink(tmp->path);

	free(tmp->path);
	free(tmp);
}

/* open an existing memory on disk */
void *starpu_unistd_global_open(struct starpu_unistd_global_obj *obj, void *base, void *pos, size_t size)
{
	/* create template */
	char *baseCpy = malloc(strlen(base)+1+strlen(pos)+1);
	STARPU_ASSERT(baseCpy != NULL);
	strcpy(baseCpy,(char *) base);
	strcat(baseCpy,(char *) "/");
	strcat(baseCpy,(char *) pos);

	int id = open(baseCpy, obj->flags);
	if (id < 0)
	{
		free(obj);
		free(baseCpy);
		return NULL;
	}

	STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	obj->descriptor = id;
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
}

/* free memory without delete it */
void starpu_unistd_global_close(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	close(tmp->descriptor);
	free(tmp->path);
	free(tmp);
}

/* read the memory disk */
int starpu_unistd_global_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;
	starpu_ssize_t nb;

#ifdef HAVE_PREAD
	nb = pread(tmp->descriptor, buf, size, offset);
#else
	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	int res = lseek(tmp->descriptor, offset, SEEK_SET);
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd lseek for read failed: offset %lu got errno %d", (unsigned long) offset, errno);

	nb = read(tmp->descriptor, buf, size);

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
#endif

	STARPU_ASSERT_MSG(nb >= 0, "Starpu Disk unistd read failed: size %lu got errno %d", (unsigned long) size, errno);
	return nb;
}

#ifdef HAVE_AIO_H
void *starpu_unistd_global_async_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_unistd_global_obj *tmp = obj;

        struct aiocb *aiocb = calloc(1,sizeof(*aiocb));

        aiocb->aio_fildes = tmp->descriptor;
        aiocb->aio_offset = offset;
        aiocb->aio_nbytes = size;
        aiocb->aio_buf = buf;
        aiocb->aio_reqprio = 0;
        aiocb->aio_lio_opcode = LIO_NOP;

        if (aio_read(aiocb) < 0)
        {
                free(aiocb);
                aiocb = NULL;
        }

        return aiocb;
}
#endif

int starpu_unistd_global_full_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void **ptr, size_t *size)
{
        struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

#ifdef STARPU_HAVE_WINDOWS
	*size = _filelength(tmp->descriptor);
#else
	struct stat st;
	fstat(tmp->descriptor, &st);

	*size = st.st_size;
#endif

	/* Allocated aligned buffer */
	starpu_malloc_flags(ptr, *size, 0);
	return starpu_unistd_global_read(base, obj, *ptr, 0, *size);
}

/* write on the memory disk */
int starpu_unistd_global_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;
	int res;

#ifdef HAVE_PWRITE
	res = pwrite(tmp->descriptor, buf, size, offset);
#else
	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	res = lseek(tmp->descriptor, offset, SEEK_SET);
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd lseek for write failed: offset %lu got errno %d", (unsigned long) offset, errno);

	res = write(tmp->descriptor, buf, size);

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
#endif

	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd write failed: size %lu got errno %d", (unsigned long) size, errno);
	return 0;
}

#ifdef HAVE_AIO_H
void *starpu_unistd_global_async_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_unistd_global_obj *tmp = obj;
        struct aiocb *aiocb = calloc(1,sizeof(*aiocb));

        aiocb->aio_fildes = tmp->descriptor;
        aiocb->aio_offset = offset;
        aiocb->aio_nbytes = size;
        aiocb->aio_buf = buf;
        aiocb->aio_reqprio = 0;
        aiocb->aio_lio_opcode = LIO_NOP;

        if (aio_write(aiocb) < 0)
        {
                free(aiocb);
                aiocb = NULL;
        }

        return aiocb;
}
#endif

int starpu_unistd_global_full_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *ptr, size_t size)
{
        struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

        /* update file size to realise the next good full_read */
        if(size != tmp->size)
        {
#ifdef STARPU_HAVE_WINDOWS
		int val = _chsize(tmp->descriptor, size);
#else
		int val = ftruncate(tmp->descriptor,size);
#endif
		STARPU_ASSERT(val == 0);
		tmp->size = size;
        }

	return starpu_unistd_global_write(base, obj, ptr, 0, size);
}

/* create a new copy of parameter == base */
void *starpu_unistd_global_plug(void *parameter, starpu_ssize_t size STARPU_ATTRIBUTE_UNUSED)
{
	char *tmp = malloc(sizeof(char)*(strlen(parameter)+1));
	STARPU_ASSERT(tmp != NULL);
	strcpy(tmp,(char *) parameter);

	{
		struct stat buf;
		if (!(stat(tmp, &buf) == 0 && S_ISDIR(buf.st_mode)))
		{
			_STARPU_ERROR("Directory '%s' does not exist\n", tmp);
		}
	}

	return (void *) tmp;
}

/* free memory allocated for the base */
void starpu_unistd_global_unplug(void *base)
{
	free(base);
}

int get_unistd_global_bandwidth_between_disk_and_main_ram(unsigned node)
{
	int res;
	unsigned iter;
	double timing_slowness, timing_latency;
	double start;
	double end;

	srand(time(NULL));
	char *buf;
	starpu_malloc_flags((void *) &buf, SIZE_DISK_MIN, 0);
	STARPU_ASSERT(buf != NULL);
	memset(buf, 0, SIZE_DISK_MIN);

	/* allocate memory */
	void *mem = _starpu_disk_alloc(node, SIZE_DISK_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;

	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) mem;

	/* Measure upload slowness */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, SIZE_DISK_MIN, NULL);

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(tmp->descriptor);
#else
		res = fsync(tmp->descriptor);
#endif

		STARPU_ASSERT_MSG(res == 0, "bandwidth computation failed");
	}
	end = starpu_timing_now();
	timing_slowness = end - start;

	/* free memory */
	starpu_free_flags(buf, SIZE_DISK_MIN, 0);

	starpu_malloc_flags((void *) &buf, MEM_SIZE, 0);
	STARPU_ASSERT(buf != NULL);

	memset(buf, 0, MEM_SIZE);

	/* Measure latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, (rand() % (SIZE_DISK_MIN/MEM_SIZE)) * MEM_SIZE, MEM_SIZE, NULL);

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(tmp->descriptor);
#else
		res = fsync(tmp->descriptor);
#endif

		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");
	}
	end = starpu_timing_now();
	timing_latency = end - start;

	_starpu_disk_free(node, mem, SIZE_DISK_MIN);
	starpu_free_flags(buf, MEM_SIZE, 0);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*1000000, (NITER/timing_slowness)*1000000,
					       timing_latency/NITER, timing_latency/NITER, node);
	return 1;
}

#ifdef HAVE_AIO_H
void starpu_unistd_global_wait_request(void *async_channel)
{
        const struct aiocb *aiocb = async_channel;
        int values = -1;
        int ret, myerrno = EAGAIN;
        while(values < 0 && (myerrno == EAGAIN || myerrno == EINTR))
        {
                /* Wait the answer of the request TIMESTAMP IS NULL */
                values = aio_suspend(&aiocb, 1, NULL);
                myerrno = errno;
        }
        ret = aio_error(aiocb);
        STARPU_ASSERT_MSG(!ret, "aio_error returned %d", ret);
}

int starpu_unistd_global_test_request(void *async_channel)
{
        const struct aiocb *aiocb = async_channel;
        int ret;

#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 22))
        /* glibc's aio_error was not threadsafe before glibc 2.22 */
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 0 };
        ret = aio_suspend(&aiocb, 1, &ts);
        if (ret < 0 && (errno == EAGAIN || errno == EINTR))
                return 0;
        STARPU_ASSERT_MSG(!ret, "aio_suspend returned %d %d\n", ret, errno);
#endif
        /* Test the answer of the request */
        ret = aio_error(aiocb);
        if (ret == 0)
                /* request is finished */
                return 1;
        if (ret == EINPROGRESS || ret == EAGAIN)
                return 0;
        /* an error occured */
        STARPU_ABORT_MSG("aio_error returned %d", ret);
}

void starpu_unistd_global_free_request(void *async_channel)
{
        struct aiocb *aiocb = async_channel;
        aio_return(aiocb);
        free(aiocb);
}
#endif
