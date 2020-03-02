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
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>
#include <errno.h>

#include <common/config.h>
#if defined(HAVE_LIBAIO_H)
#include <libaio.h>
#elif defined(HAVE_AIO_H)
#include <aio.h>
#endif
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif
#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <core/disk_ops/unistd/disk_unistd_global.h>
#include <datawizard/copy_driver.h>
#include <datawizard/data_request.h>
#include <datawizard/memory_manager.h>
#include <starpu_parameters.h>
#include <common/uthash.h>

#ifdef STARPU_HAVE_WINDOWS
#  include <io.h>
#endif

#define NITER	_starpu_calibration_minimum

#ifdef O_DIRECT
#  define MEM_SIZE getpagesize()
#else
#  define MEM_SIZE 1
#endif

#define MAX_OPEN_FILES 64
#define TEMP_HIERARCHY_DEPTH 2

#if !defined(HAVE_COPY_FILE_RANGE) && defined( __NR_copy_file_range)
static starpu_ssize_t copy_file_range(int fd_in, loff_t *off_in, int fd_out,
				      loff_t *off_out, size_t len, unsigned int flags)
{
	return syscall(__NR_copy_file_range, fd_in, off_in, fd_out,
			off_out, len, flags);
}
#endif

static unsigned starpu_unistd_opened_files;

#ifdef STARPU_UNISTD_USE_COPY
LIST_TYPE(starpu_unistd_work_copy,
	int fd_src;
	int fd_dst;
	loff_t off_src;
	loff_t off_dst;
	struct starpu_unistd_global_obj * obj_src;
	struct starpu_unistd_global_obj * obj_dst;
	size_t len;
	unsigned flags;
	starpu_sem_t finished;
);

struct starpu_unistd_copy_thread
{
	int run;
	starpu_pthread_t thread;
	starpu_pthread_cond_t cond;
	starpu_pthread_mutex_t mutex;
	struct starpu_unistd_work_copy_list list;
};

struct starpu_unistd_copy_thread copy_thread[STARPU_MAXNODES][STARPU_MAXNODES];
static unsigned starpu_unistd_nb_disk_opened = 0;
/* copy_file_range syscall can return ENOSYS. Use global var to catch
 * and prevent StarPU using direct disk to disk copy */
static int starpu_unistd_copy_works = 1;
#endif

struct starpu_unistd_base
{
	char * path;
	int created;
	/* To know which thread handles the copy function */
#ifdef STARPU_UNISTD_USE_COPY
	unsigned disk_index;
#endif
#if defined(HAVE_LIBAIO_H)
	io_context_t ctx;
        struct starpu_unistd_aiocb_link * hashtable;
        starpu_pthread_mutex_t mutex;
#endif
};

#if defined(HAVE_LIBAIO_H)
struct starpu_unistd_aiocb_link
{
        UT_hash_handle hh;
        void * starpu_aiocb;
        void * aiocb;
};
struct starpu_unistd_aiocb
{
        int finished;
	struct iocb iocb;
	struct starpu_unistd_global_obj *obj;
        struct starpu_unistd_base *base;
	size_t len;
};
#elif defined(HAVE_AIO_H)
struct starpu_unistd_aiocb
{
	struct aiocb aiocb;
	struct starpu_unistd_global_obj *obj;
};
#endif

enum starpu_unistd_wait_type { STARPU_UNISTD_AIOCB, STARPU_UNISTD_COPY };

union starpu_unistd_wait_event
{
	struct starpu_unistd_work_copy * event_copy;
#if defined(HAVE_LIBAIO_H) || defined(HAVE_AIO_H)
	struct starpu_unistd_aiocb event_aiocb;
#endif
};

struct starpu_unistd_wait
{
	enum starpu_unistd_wait_type type;
	union starpu_unistd_wait_event event;
};

/* ------------------- use UNISTD to write on disk -------------------  */

static void _starpu_unistd_init(struct starpu_unistd_global_obj *obj, int descriptor, char *path, size_t size)
{
	STARPU_HG_DISABLE_CHECKING(starpu_unistd_opened_files);
#ifdef STARPU_UNISTD_USE_COPY
	STARPU_HG_DISABLE_CHECKING(starpu_unistd_copy_works);
#endif
	if (starpu_unistd_opened_files >= MAX_OPEN_FILES)
	{
		/* Too many opened files, avoid keeping this one opened */
		close(descriptor);
		descriptor = -1;
	}
	else
		(void) STARPU_ATOMIC_ADD(&starpu_unistd_opened_files, 1);

	STARPU_PTHREAD_MUTEX_INIT(&obj->mutex, NULL);

	obj->descriptor = descriptor;
	obj->path = path;
	obj->size = size;
}

static int _starpu_unistd_reopen(struct starpu_unistd_global_obj *obj)
{
	int id = open(obj->path, obj->flags);
	STARPU_ASSERT_MSG(id >= 0, "Reopening file %s failed: errno %d", obj->path, errno);
	return id;
}

static void _starpu_unistd_reclose(int id)
{
	close(id);
}

static void _starpu_unistd_close(struct starpu_unistd_global_obj *obj)
{
	if (obj->descriptor < 0)
		return;

	if (starpu_unistd_opened_files < MAX_OPEN_FILES)
		(void) STARPU_ATOMIC_ADD(&starpu_unistd_opened_files, -1);

	close(obj->descriptor);
}

static void _starpu_unistd_fini(struct starpu_unistd_global_obj *obj)
{
	STARPU_PTHREAD_MUTEX_DESTROY(&obj->mutex);

	free(obj->path);
	obj->path = NULL;
	free(obj);
}

/* allocation memory on disk */
void *starpu_unistd_global_alloc(struct starpu_unistd_global_obj *obj, void *base, size_t size)
{
	int id;
	struct starpu_unistd_base * fileBase = (struct starpu_unistd_base *) base;
	char *baseCpy = _starpu_mktemp_many(fileBase->path, TEMP_HIERARCHY_DEPTH, obj->flags, &id);

	/* fail */
	if (!baseCpy)
	{
		free(obj);
		return NULL;
	}

	int val = _starpu_ftruncate(id,size);
	/* fail */
	if (val < 0)
	{
		_STARPU_DISP("Could not truncate file, ftruncate failed with error '%s'\n", strerror(errno));
		close(id);
		unlink(baseCpy);
		free(baseCpy);
		free(obj);
		return NULL;
	}

	_starpu_unistd_init(obj, id, baseCpy, size);

	return obj;
}

/* free memory on disk */
void starpu_unistd_global_free(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

	_starpu_unistd_close(tmp);
	unlink(tmp->path);
	_starpu_rmtemp_many(tmp->path, TEMP_HIERARCHY_DEPTH);
	_starpu_unistd_fini(tmp);
}

/* open an existing memory on disk */
void *starpu_unistd_global_open(struct starpu_unistd_global_obj *obj, void *base, void *pos, size_t size)
{
	struct starpu_unistd_base *fileBase = (struct starpu_unistd_base *) base;
	/* create template */
	char *baseCpy;
	_STARPU_MALLOC(baseCpy, strlen(fileBase->path)+1+strlen(pos)+1);

	snprintf(baseCpy, strlen(fileBase->path)+1+strlen(pos)+1, "%s/%s", fileBase->path, (char *)pos);

	int id = open(baseCpy, obj->flags);
	if (id < 0)
	{
		free(obj);
		free(baseCpy);
		return NULL;
	}

	_starpu_unistd_init(obj, id, baseCpy, size);

	return obj;
}

/* free memory without delete it */
void starpu_unistd_global_close(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

	_starpu_unistd_close(tmp);
	_starpu_unistd_fini(tmp);
}

/* read the memory disk */
int starpu_unistd_global_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;
	starpu_ssize_t nb;
	int fd = tmp->descriptor;

#ifdef HAVE_PREAD
	if (fd >= 0)
		nb = pread(fd, buf, size, offset);
	else
#endif
	{
		if (tmp->descriptor >= 0)
			STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);
		else
			fd = _starpu_unistd_reopen(obj);

		int res = lseek(fd, offset, SEEK_SET);
		STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd lseek for read failed: offset %lu got errno %d", (unsigned long) offset, errno);

		nb = read(fd, buf, size);

		if (tmp->descriptor >= 0)
			STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
		else
			_starpu_unistd_reclose(fd);

	}

	STARPU_ASSERT_MSG(nb >= 0, "Starpu Disk unistd read failed: size %lu got errno %d", (unsigned long) size, errno);

	return nb;
}

#if defined(HAVE_LIBAIO_H)
void *starpu_unistd_global_async_read(void *base, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_unistd_base * fileBase = (struct starpu_unistd_base *) base;
        struct starpu_unistd_global_obj *tmp = obj;
	struct starpu_unistd_wait * event;
	_STARPU_CALLOC(event, 1,sizeof(*event));
	event->type = STARPU_UNISTD_AIOCB;
        struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
        struct iocb *iocb = &starpu_aiocb->iocb;
        starpu_aiocb->obj = obj;
        int fd = tmp->descriptor;
	int err;

        if (fd < 0)
                fd = _starpu_unistd_reopen(obj);

	starpu_aiocb->len = size;
        starpu_aiocb->finished = 0;
        starpu_aiocb->base = fileBase;
	io_prep_pread(iocb, fd, buf, size, offset);
	if ((err = io_submit(fileBase->ctx, 1, &iocb)) < 0)
	{
		_STARPU_DISP("Warning: io_submit returned %d (%s)\n", err, strerror(err));
                if (tmp->descriptor < 0)
                        _starpu_unistd_reclose(fd);
                iocb = NULL;
        }

        struct starpu_unistd_aiocb_link *l;
        _STARPU_MALLOC(l, sizeof(*l));
        l->aiocb = iocb;
        l->starpu_aiocb = starpu_aiocb;
        STARPU_PTHREAD_MUTEX_LOCK(&fileBase->mutex);
        HASH_ADD_PTR(fileBase->hashtable, aiocb, l);
        STARPU_PTHREAD_MUTEX_UNLOCK(&fileBase->mutex);

        return event;
}
#elif defined(HAVE_AIO_H)
void *starpu_unistd_global_async_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_unistd_global_obj *tmp = obj;
	struct starpu_unistd_wait * event;
	_STARPU_CALLOC(event, 1,sizeof(*event));
	event->type = STARPU_UNISTD_AIOCB;
        struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
        struct aiocb *aiocb = &starpu_aiocb->aiocb;
        starpu_aiocb->obj = obj;
        int fd = tmp->descriptor;

        if (fd < 0)
                fd = _starpu_unistd_reopen(obj);

        aiocb->aio_fildes = fd;
        aiocb->aio_offset = offset;
        aiocb->aio_nbytes = size;
        aiocb->aio_buf = buf;
        aiocb->aio_reqprio = 0;
        aiocb->aio_lio_opcode = LIO_NOP;

        if (aio_read(aiocb) < 0)
        {
		_STARPU_DISP("Warning: aio_read returned %d (%s)\n", errno, strerror(errno));
                if (tmp->descriptor < 0)
                        _starpu_unistd_reclose(fd);
        }

        return event;
}
#endif

int starpu_unistd_global_full_read(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void **ptr, size_t *size, unsigned dst_node)
{
        struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;
	int fd = tmp->descriptor;

	if (fd < 0)
		fd = _starpu_unistd_reopen(obj);
#ifdef STARPU_HAVE_WINDOWS
	*size = _filelength(fd);
#else
	struct stat st;
	int ret = fstat(fd, &st);
	STARPU_ASSERT(ret==0);

	*size = st.st_size;
#endif
	if (tmp->descriptor < 0)
		_starpu_unistd_reclose(fd);

	/* Allocated aligned buffer */
	_starpu_malloc_flags_on_node(dst_node, ptr, *size, 0);
	return starpu_unistd_global_read(base, obj, *ptr, 0, *size);
}

/* write on the memory disk */
int starpu_unistd_global_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size)
{
	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;
	int res;
	int fd = tmp->descriptor;

#ifdef HAVE_PWRITE
	if (fd >= 0)
		res = pwrite(fd, buf, size, offset);
	else
#endif
	{
		if (tmp->descriptor >= 0)
			STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);
		else
			fd = _starpu_unistd_reopen(obj);

		res = lseek(fd, offset, SEEK_SET);
		STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd lseek for write failed: offset %lu got errno %d", (unsigned long) offset, errno);

		res = write(fd, buf, size);

		if (tmp->descriptor >= 0)
			STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);
		else
			_starpu_unistd_reclose(fd);
	}

	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd write failed: size %lu got errno %d", (unsigned long) size, errno);
	return 0;
}

#if defined(HAVE_LIBAIO_H)
void *starpu_unistd_global_async_write(void *base, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_unistd_base * fileBase = (struct starpu_unistd_base *) base;
        struct starpu_unistd_global_obj *tmp = obj;
	struct starpu_unistd_wait * event;
	_STARPU_CALLOC(event, 1,sizeof(*event));
	event->type = STARPU_UNISTD_AIOCB;
        struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
        struct iocb *iocb = &starpu_aiocb->iocb;
        starpu_aiocb->obj = obj;
        int fd = tmp->descriptor;
	int err;

        if (fd < 0)
                fd = _starpu_unistd_reopen(obj);

	starpu_aiocb->len = size;
        starpu_aiocb->finished = 0;
        starpu_aiocb->base = fileBase;
	io_prep_pwrite(iocb, fd, buf, size, offset);
	if ((err = io_submit(fileBase->ctx, 1, &iocb)) < 0)
        {
		_STARPU_DISP("Warning: io_submit returned %d (%s)\n", err, strerror(err));
                if (tmp->descriptor < 0)
                        _starpu_unistd_reclose(fd);
                iocb = NULL;
        }

        struct starpu_unistd_aiocb_link *l;
        _STARPU_MALLOC(l, sizeof(*l));
        l->aiocb = iocb;
        l->starpu_aiocb = starpu_aiocb;
        STARPU_PTHREAD_MUTEX_LOCK(&fileBase->mutex);
        HASH_ADD_PTR(fileBase->hashtable, aiocb, l);
        STARPU_PTHREAD_MUTEX_UNLOCK(&fileBase->mutex);

        return event;
}
#elif defined(HAVE_AIO_H)
void *starpu_unistd_global_async_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size)
{
        struct starpu_unistd_global_obj *tmp = obj;
	struct starpu_unistd_wait * event;
	_STARPU_CALLOC(event, 1,sizeof(*event));
	event->type = STARPU_UNISTD_AIOCB;
        struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
        struct aiocb *aiocb = &starpu_aiocb->aiocb;
        starpu_aiocb->obj = obj;
        int fd = tmp->descriptor;

        if (fd < 0)
                fd = _starpu_unistd_reopen(obj);

        aiocb->aio_fildes = fd;
        aiocb->aio_offset = offset;
        aiocb->aio_nbytes = size;
        aiocb->aio_buf = buf;
        aiocb->aio_reqprio = 0;
        aiocb->aio_lio_opcode = LIO_NOP;

        if (aio_write(aiocb) < 0)
        {
		_STARPU_DISP("Warning: aio_write returned %d (%s)\n", errno, strerror(errno));
                if (tmp->descriptor < 0)
                        _starpu_unistd_reclose(fd);
                aiocb = NULL;
        }

        return event;
}
#endif

int starpu_unistd_global_full_write(void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *ptr, size_t size)
{
        struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

        /* update file size to realise the next good full_read */
        if(size != tmp->size)
        {
		int fd = tmp->descriptor;

		if (fd < 0)
			fd = _starpu_unistd_reopen(obj);
		int val = _starpu_ftruncate(fd,size);
		if (tmp->descriptor < 0)
			_starpu_unistd_reclose(fd);
		STARPU_ASSERT(val == 0);
		tmp->size = size;
        }

	return starpu_unistd_global_write(base, obj, ptr, 0, size);
}

#if HAVE_AIO_H
void * starpu_unistd_global_async_full_read (void * base, void * obj, void ** ptr, size_t * size, unsigned dst_node)
{
        struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;
	int fd = tmp->descriptor;

	if (fd < 0)
		fd = _starpu_unistd_reopen(obj);
#ifdef STARPU_HAVE_WINDOWS
	*size = _filelength(fd);
#else
	struct stat st;
	int ret = fstat(fd, &st);
	STARPU_ASSERT(ret==0);

	*size = st.st_size;
#endif
	if (tmp->descriptor < 0)
		_starpu_unistd_reclose(fd);

	/* Allocated aligned buffer */
	_starpu_malloc_flags_on_node(dst_node, ptr, *size, 0);
	return starpu_unistd_global_async_read(base, obj, *ptr, 0, *size);
}

void * starpu_unistd_global_async_full_write (void * base, void * obj, void * ptr, size_t size)
{
        struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) obj;

        /* update file size to realise the next good full_read */
        if(size != tmp->size)
        {
		int fd = tmp->descriptor;

		if (fd < 0)
			fd = _starpu_unistd_reopen(obj);
		int val = _starpu_ftruncate(fd,size);
		if (tmp->descriptor < 0)
			_starpu_unistd_reclose(fd);
		STARPU_ASSERT(val == 0);
		tmp->size = size;
        }

	return starpu_unistd_global_async_write(base, obj, ptr, 0, size);
}
#endif

#ifdef STARPU_UNISTD_USE_COPY
static void * starpu_unistd_internal_thread(void * arg)
{
	struct starpu_unistd_copy_thread * internal_copy_thread = (struct starpu_unistd_copy_thread *) arg;

	while (internal_copy_thread->run || !starpu_unistd_work_copy_list_empty(&internal_copy_thread->list))
	{
		STARPU_PTHREAD_MUTEX_LOCK(&internal_copy_thread->mutex);
		if (internal_copy_thread->run && starpu_unistd_work_copy_list_empty(&internal_copy_thread->list))
                        STARPU_PTHREAD_COND_WAIT(&internal_copy_thread->cond, &internal_copy_thread->mutex);
		STARPU_PTHREAD_MUTEX_UNLOCK(&internal_copy_thread->mutex);

		if (!starpu_unistd_work_copy_list_empty(&internal_copy_thread->list))
		{
			STARPU_PTHREAD_MUTEX_LOCK(&internal_copy_thread->mutex);
			struct starpu_unistd_work_copy * work = starpu_unistd_work_copy_list_pop_back(&internal_copy_thread->list);
			STARPU_PTHREAD_MUTEX_UNLOCK(&internal_copy_thread->mutex);

			starpu_ssize_t ret = copy_file_range(work->fd_src, &work->off_src, work->fd_dst, &work->off_dst, work->len, work->flags);

			if (ret == -1 && (errno == ENOSYS || errno == EINVAL))
			{
				void *buf;
				/* System call not supported, or glibc
				 * compatibility layer does not work (e.g.
				 * because we use O_DIRECT and glibc doesn't
				 * align the buffer), avoid submitting more
				 * copies.  */
				starpu_unistd_copy_works = 0;

				/* And do the copy by hand for this time */
				starpu_malloc(&buf, work->len);
				ret = pread(work->fd_src, buf, work->len, work->off_src);
				STARPU_ASSERT_MSG(ret >= 0, "Reading failed (errno %d)", errno);
				STARPU_ASSERT_MSG((size_t) ret == work->len, "Reading failed (value %ld instead of %ld)", (long)ret, (long)work->len);
				ret = pwrite(work->fd_dst, buf, work->len, work->off_dst);
				STARPU_ASSERT_MSG(ret >= 0, "Writing failed (errno %d)", errno);
				STARPU_ASSERT_MSG((size_t) ret == work->len, "Writing failed (value %ld instead of %ld)", (long)ret, (long)work->len);
				starpu_free(buf);
			}
			else
			{
				STARPU_ASSERT_MSG(ret >= 0, "Copy_file_range failed (errno %d)", errno);
				STARPU_ASSERT_MSG((size_t) ret == work->len, "Copy_file_range failed (value %ld instead of %ld)", (long)ret, (long)work->len);
			}

			starpu_sem_post(&work->finished);

			/* Don't free work, it's done when tested/waited are completed */
		}

	}

	return NULL;
}

static void initialize_working_thread(struct starpu_unistd_copy_thread *internal_copy_thread)
{
	STARPU_PTHREAD_MUTEX_INIT(&internal_copy_thread->mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&internal_copy_thread->cond, NULL);
	internal_copy_thread->run = 1;
	starpu_unistd_work_copy_list_init(&internal_copy_thread->list);
	STARPU_PTHREAD_CREATE(&internal_copy_thread->thread, NULL, starpu_unistd_internal_thread, internal_copy_thread);
}
#endif

/* create a new copy of parameter == base */
void *starpu_unistd_global_plug(void *parameter, starpu_ssize_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_base * base;
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

#if defined(HAVE_LIBAIO_H)
        STARPU_PTHREAD_MUTEX_INIT(&base->mutex, NULL);
        base->hashtable = NULL;
        unsigned nb_event = MAX_PENDING_REQUESTS_PER_NODE + MAX_PENDING_PREFETCH_REQUESTS_PER_NODE + MAX_PENDING_IDLE_REQUESTS_PER_NODE;
        memset(&base->ctx, 0, sizeof(base->ctx));
	int ret = io_setup(nb_event, &base->ctx);
	STARPU_ASSERT(ret == 0);
#endif

#ifdef STARPU_UNISTD_USE_COPY
	base->disk_index = starpu_unistd_nb_disk_opened;
	starpu_unistd_nb_disk_opened++;

	unsigned i;
	for (i = 0; i < starpu_unistd_nb_disk_opened; i++)
	{
		initialize_working_thread(&copy_thread[i][base->disk_index]);
		/* don't initialize twice this case */
		if (i != base->disk_index)
			initialize_working_thread(&copy_thread[base->disk_index][i]);
	}
#endif

	return (void *) base;
}

#ifdef STARPU_UNISTD_USE_COPY
static void ending_working_thread(struct starpu_unistd_copy_thread *internal_copy_thread)
{
	STARPU_PTHREAD_MUTEX_LOCK(&internal_copy_thread->mutex);
	internal_copy_thread->run = 0;
	STARPU_PTHREAD_COND_BROADCAST(&internal_copy_thread->cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&internal_copy_thread->mutex);

	STARPU_PTHREAD_JOIN(internal_copy_thread->thread, NULL);

	STARPU_PTHREAD_MUTEX_DESTROY(&internal_copy_thread->mutex);
	STARPU_PTHREAD_COND_DESTROY(&internal_copy_thread->cond);
}
#endif

/* free memory allocated for the base */
void starpu_unistd_global_unplug(void *base)
{
	struct starpu_unistd_base * fileBase = (struct starpu_unistd_base *) base;
#if defined(HAVE_LIBAIO_H)
        STARPU_PTHREAD_MUTEX_DESTROY(&fileBase->mutex);
        io_destroy(fileBase->ctx);
#endif
	if (fileBase->created)
		rmdir(fileBase->path);

#ifdef STARPU_UNISTD_USE_COPY
	unsigned i;
	for (i = 0; i < fileBase->disk_index+1; i++)
	{
		ending_working_thread(&copy_thread[i][fileBase->disk_index]);
		/* don't uninitialize twice this case */
		if (i != fileBase->disk_index)
			ending_working_thread(&copy_thread[fileBase->disk_index][i]);
	}
	starpu_unistd_nb_disk_opened--;

#endif

	free(fileBase->path);
	free(fileBase);
}

int get_unistd_global_bandwidth_between_disk_and_main_ram(unsigned node, void *base)
{
	int res;
	unsigned iter;
	double timing_slowness, timing_latency;
	double start;
	double end;
	struct starpu_unistd_base * fileBase = (struct starpu_unistd_base *) base;

	srand(time(NULL));
	char *buf;
	starpu_malloc_flags((void *) &buf, STARPU_DISK_SIZE_MIN, 0);
	STARPU_ASSERT(buf != NULL);
	memset(buf, 0, STARPU_DISK_SIZE_MIN);

	/* allocate memory */
	void *mem = _starpu_disk_alloc(node, STARPU_DISK_SIZE_MIN);
	/* fail to alloc */
	if (mem == NULL)
		return 0;

	struct starpu_unistd_global_obj *tmp = (struct starpu_unistd_global_obj *) mem;

	/* Measure upload slowness */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		int fd = tmp->descriptor;

		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, STARPU_DISK_SIZE_MIN, NULL);

		if (fd < 0)
			fd = _starpu_unistd_reopen(tmp);
#ifdef STARPU_HAVE_WINDOWS
		res = _commit(fd);
#else
		res = fsync(fd);
#endif
		if (tmp->descriptor < 0)
			_starpu_unistd_reclose(fd);

		STARPU_ASSERT_MSG(res == 0, "bandwidth computation failed");
	}
	end = starpu_timing_now();
	timing_slowness = end - start;

	/* free memory */
	starpu_free_flags(buf, STARPU_DISK_SIZE_MIN, 0);

	starpu_malloc_flags((void *) &buf, MEM_SIZE, 0);
	STARPU_ASSERT(buf != NULL);

	memset(buf, 0, MEM_SIZE);

	/* Measure latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; ++iter)
	{
		int fd = tmp->descriptor;

		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, (rand() % (STARPU_DISK_SIZE_MIN/MEM_SIZE)) * MEM_SIZE, MEM_SIZE, NULL);

		if (fd < 0)
			fd = _starpu_unistd_reopen(tmp);
#ifdef STARPU_HAVE_WINDOWS
		res = _commit(fd);
#else
		res = fsync(fd);
#endif
		if (tmp->descriptor < 0)
			_starpu_unistd_reclose(fd);

		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");
	}
	end = starpu_timing_now();
	timing_latency = end - start;

	_starpu_disk_free(node, mem, STARPU_DISK_SIZE_MIN);
	starpu_free_flags(buf, MEM_SIZE, 0);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*STARPU_DISK_SIZE_MIN, (NITER/timing_slowness)*STARPU_DISK_SIZE_MIN,
			timing_latency/NITER, timing_latency/NITER, node, fileBase->path);
	return 1;
}

void starpu_unistd_global_wait_request(void *async_channel)
{
	struct starpu_unistd_wait * event = async_channel;
	switch (event->type)
	{
		case STARPU_UNISTD_AIOCB :
		{
#if defined(HAVE_LIBAIO_H)
			struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
			struct io_event event;

			int values = -1;
			int myerrno = EAGAIN;
			while(!starpu_aiocb->finished || (values <= 0 && (myerrno == EAGAIN || myerrno == EINTR)))
			{
				/* Wait the answer of the request timeout IS NULL */
				values = io_getevents(starpu_aiocb->base->ctx, 1, 1, &event, NULL);
				if (values < 0)
					myerrno = -values;
				if (values > 0)
				{
					//we may catch an other request...
					STARPU_PTHREAD_MUTEX_LOCK(&starpu_aiocb->base->mutex);

					struct starpu_unistd_aiocb_link *l = NULL;
					HASH_FIND_PTR(starpu_aiocb->base->hashtable, &event.obj, l);
					STARPU_ASSERT(l != NULL);

					HASH_DEL(starpu_aiocb->base->hashtable, l);
					STARPU_PTHREAD_MUTEX_UNLOCK(&starpu_aiocb->base->mutex);
					((struct starpu_unistd_aiocb *) l->starpu_aiocb)->finished = 1;
					free(l);
				}
			}
#elif defined(HAVE_AIO_H)
			struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
			const struct aiocb *aiocb = &starpu_aiocb->aiocb;
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
#endif
			break;
		}

#ifdef STARPU_UNISTD_USE_COPY
		case STARPU_UNISTD_COPY :
		{
			starpu_sem_wait(&event->event.event_copy->finished);
			break;
		}
#endif

		default :
			STARPU_ABORT_MSG();
			break;
	}
}

int starpu_unistd_global_test_request(void *async_channel)
{
	struct starpu_unistd_wait * event = async_channel;
	switch (event->type)
	{
		case STARPU_UNISTD_AIOCB :
		{
#if defined(HAVE_LIBAIO_H)
			struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
			struct io_event event;
			struct timespec ts;
			int ret;

			if (starpu_aiocb->finished)
				return 1;

			memset(&ts, 0, sizeof(ts));

			/* Test the answer of the request */
			ret = io_getevents(starpu_aiocb->base->ctx, 0, 1, &event, &ts);

			if (ret == 1)
			{
				//we may catch an other request...
				STARPU_PTHREAD_MUTEX_LOCK(&starpu_aiocb->base->mutex);

				struct starpu_unistd_aiocb_link *l = NULL;
				HASH_FIND_PTR(starpu_aiocb->base->hashtable, &event.obj, l);
				STARPU_ASSERT(l != NULL);

				HASH_DEL(starpu_aiocb->base->hashtable, l);
				STARPU_PTHREAD_MUTEX_UNLOCK(&starpu_aiocb->base->mutex);
				((struct starpu_unistd_aiocb *) l->starpu_aiocb)->finished = 1;
				free(l);

				if (starpu_aiocb->finished)
					return 1;
			}

			return 0;
#elif defined(HAVE_AIO_H)
			struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
			const struct aiocb *aiocb = &starpu_aiocb->aiocb;
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
			if (ret == EINTR || ret == EINPROGRESS || ret == EAGAIN)
				return 0;
			/* an error occured */
			STARPU_ABORT_MSG("aio_error returned %d", ret);
#endif
			break;
		}

#ifdef STARPU_UNISTD_USE_COPY
		case STARPU_UNISTD_COPY :
		{
			return starpu_sem_trywait(&event->event.event_copy->finished) == 0;
		}
#endif

		default :
			STARPU_ABORT_MSG();
			break;
	}

	return 0;
}

void starpu_unistd_global_free_request(void *async_channel)
{
	struct starpu_unistd_wait * event = async_channel;
	switch (event->type)
	{
		case STARPU_UNISTD_AIOCB :
		{
#if defined(HAVE_LIBAIO_H)
			struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
			struct iocb *iocb = &starpu_aiocb->iocb;
			if (starpu_aiocb->obj->descriptor < 0)
				_starpu_unistd_reclose(iocb->aio_fildes);
			free(event);
#elif defined(HAVE_AIO_H)
			struct starpu_unistd_aiocb *starpu_aiocb = &event->event.event_aiocb;
			struct aiocb *aiocb = &starpu_aiocb->aiocb;
			if (starpu_aiocb->obj->descriptor < 0)
				_starpu_unistd_reclose(aiocb->aio_fildes);
			aio_return(aiocb);
			free(event);
#endif
			break;
		}

#ifdef STARPU_UNISTD_USE_COPY
		case STARPU_UNISTD_COPY :
		{
			starpu_sem_destroy(&event->event.event_copy->finished);

			int fd_src = event->event.event_copy->obj_src->descriptor;
			if (fd_src < 0)
				_starpu_unistd_reclose(event->event.event_copy->fd_src);
			int fd_dst = event->event.event_copy->obj_dst->descriptor;
			if (fd_dst < 0)
				_starpu_unistd_reclose(event->event.event_copy->fd_dst);

			starpu_unistd_work_copy_delete(event->event.event_copy);
			free(event);
			break;
		}
#endif

		default :
			STARPU_ABORT_MSG();
			break;
	}
}


#ifdef STARPU_UNISTD_USE_COPY
void *  starpu_unistd_global_copy(void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size)
{
	struct starpu_unistd_global_obj * unistd_obj_src = obj_src;
	struct starpu_unistd_global_obj * unistd_obj_dst = obj_dst;
	struct starpu_unistd_base * unistd_base_src = base_src;
	struct starpu_unistd_base * unistd_base_dst = base_dst;

	if (starpu_unistd_copy_works == 0)
		/* It didn't work previously, don't bother submitting more.  */
		return NULL;

	struct starpu_unistd_wait * event;
	_STARPU_CALLOC(event, 1,sizeof(*event));
	event->type = STARPU_UNISTD_COPY;

	int fd_src = unistd_obj_src->descriptor;
	if (fd_src < 0)
		fd_src = _starpu_unistd_reopen(obj_src);
	int fd_dst = unistd_obj_dst->descriptor;
	if (fd_dst < 0)
		fd_dst = _starpu_unistd_reopen(obj_dst);

	struct starpu_unistd_work_copy * work = starpu_unistd_work_copy_new();
	work->fd_src = fd_src;
	work->fd_dst = fd_dst;
	work->obj_src = unistd_obj_src;
	work->obj_dst = unistd_obj_dst;
	work->off_src = offset_src;
	work->off_dst = offset_dst;
	work->len = size;
	/* currently not used by copy_file_range */
	work->flags = 0;
	starpu_sem_init(&work->finished, 0, 0);

	event->event.event_copy = work;

	struct starpu_unistd_copy_thread * thread = &copy_thread[unistd_base_src->disk_index][unistd_base_dst->disk_index];

	STARPU_PTHREAD_MUTEX_LOCK(&thread->mutex);
	starpu_unistd_work_copy_list_push_front(&thread->list, work);
        STARPU_PTHREAD_COND_BROADCAST(&thread->cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&thread->mutex);

	return event;
}
#endif
