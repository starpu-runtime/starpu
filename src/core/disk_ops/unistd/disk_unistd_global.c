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
#ifdef HAVE_AIO_H
#include <aio.h>
#endif
#include <errno.h>

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <core/disk_ops/unistd/disk_unistd_global.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memory_manager.h>

#ifdef STARPU_HAVE_WINDOWS
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
	unsigned int sizeBase = strlen(base) + 1 + strlen(template)+1;

	char * baseCpy = malloc(sizeBase*sizeof(char));
	STARPU_ASSERT(baseCpy != NULL);

	strcpy(baseCpy, (char *) base);
	strcat(baseCpy,"/");
	strcat(baseCpy,template);
#ifdef STARPU_LINUX_SYS
	id = mkostemp(baseCpy, obj->flags);
#elif defined(STARPU_HAVE_WINDOWS)
	/* size in windows is a multiple of char */
	_mktemp(baseCpy);
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
	obj->path = baseCpy;
	obj->size = size;

	return (void *) obj;
}


/* free memory on disk */
 void
starpu_unistd_global_free (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

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
	char * baseCpy = malloc(strlen(base)+1+strlen(pos)+1);
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
 void 
starpu_unistd_global_close (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	STARPU_PTHREAD_MUTEX_DESTROY(&tmp->mutex);

	close(tmp->descriptor);
	free(tmp->path);
	free(tmp);	
}


/* read the memory disk */
 int 
starpu_unistd_global_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, void * async_channel STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);

	int res = lseek(tmp->descriptor, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd lseek for read failed: offset %lu got errno %d", (unsigned long) offset, errno);

	ssize_t nb = read(tmp->descriptor, buf, size);
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd read failed: size %lu got errno %d", (unsigned long) size, errno);
	
	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return nb;
}


#ifdef HAVE_AIO_H
int
starpu_unistd_global_async_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, void * async_channel)
{
        struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

        struct _starpu_async_channel * channel = (struct _starpu_async_channel *) async_channel;
        struct aiocb *aiocb = &channel->event.disk_event._starpu_aiocb_disk;

        memset(aiocb, 0, sizeof(struct aiocb));

        aiocb->aio_fildes = tmp->descriptor;
        aiocb->aio_offset = offset;
        aiocb->aio_nbytes = size;
        aiocb->aio_buf = buf;
        aiocb->aio_reqprio = 0;
        aiocb->aio_lio_opcode = LIO_NOP;

        return aio_read(aiocb);
}
#endif

int
starpu_unistd_global_full_read(unsigned node, void *base STARPU_ATTRIBUTE_UNUSED, void * obj, void ** ptr, size_t * size)
{
        struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

        *size = tmp->size;
        *ptr = malloc(*size);
	return _starpu_disk_read(node, STARPU_MAIN_RAM, obj, *ptr, 0, *size, NULL);
}


/* write on the memory disk */
 int 
starpu_unistd_global_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size, void * async_channel STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

	STARPU_PTHREAD_MUTEX_LOCK(&tmp->mutex);
	
	int res = lseek(tmp->descriptor, offset, SEEK_SET); 
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd lseek for write failed: offset %lu got errno %d", (unsigned long) offset, errno);

	ssize_t nb = write (tmp->descriptor, buf, size);
	STARPU_ASSERT_MSG(res >= 0, "Starpu Disk unistd write failed: size %lu got errno %d", (unsigned long) size, errno);

	STARPU_PTHREAD_MUTEX_UNLOCK(&tmp->mutex);

	return nb;
}


#ifdef HAVE_AIO_H
int
starpu_unistd_global_async_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, void * async_channel)
{
        struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

        struct _starpu_async_channel * channel = (struct _starpu_async_channel *) async_channel;
        struct aiocb *aiocb = &channel->event.disk_event._starpu_aiocb_disk ;
        memset(aiocb, 0, sizeof(struct aiocb));

        aiocb->aio_fildes = tmp->descriptor;
        aiocb->aio_offset = offset;
        aiocb->aio_nbytes = size;
        aiocb->aio_buf = buf;
        aiocb->aio_reqprio = 0;
        aiocb->aio_lio_opcode = LIO_NOP;

        return aio_write(aiocb);
}
#endif

int
starpu_unistd_global_full_write (unsigned node, void * base STARPU_ATTRIBUTE_UNUSED, void * obj, void * ptr, size_t size)
{
        struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) obj;

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

                        STARPU_ASSERT_MSG(val >= 0,"StarPU Error to truncate file in UNISTD full_write function");
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
	int res;
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
	
	struct starpu_unistd_global_obj * tmp = (struct starpu_unistd_global_obj *) mem;

	/* Measure upload slowness */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; ++iter)
	{
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, 0, SIZE_BENCH, NULL);

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(tmp->descriptor);
#else
		res = fsync(tmp->descriptor);
#endif

		STARPU_ASSERT_MSG(res == 0, "bandwidth computation failed");
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
		_starpu_disk_write(STARPU_MAIN_RAM, node, mem, buf, rand() % (SIZE_BENCH -1) , getpagesize(), NULL);

#ifdef STARPU_HAVE_WINDOWS
		res = _commit(tmp->descriptor);
#else
		res = fsync(tmp->descriptor);
#endif

		STARPU_ASSERT_MSG(res == 0, "Latency computation failed");
	}
	gettimeofday(&end, NULL);
	timing_latency = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	_starpu_disk_free(node, mem, SIZE_BENCH);
	starpu_free(buf);

	_starpu_save_bandwidth_and_latency_disk((NITER/timing_slowness)*1000000, (NITER/timing_slowness)*1000000,
					       timing_latency/NITER, timing_latency/NITER, node);
	return 1;
}

#ifdef HAVE_AIO_H
void
starpu_unistd_global_wait_request(void * async_channel)
{
        struct _starpu_async_channel * channel = (struct _starpu_async_channel *) async_channel;
        const struct aiocb * aiocb = &channel->event.disk_event._starpu_aiocb_disk;
        const struct aiocb * list[1];
        list[0] = aiocb;
        int values = -1;
        int error_disk = EAGAIN;
        while(values < 0 || error_disk == EAGAIN)
        {
                /* Wait the answer of the request TIMESTAMP IS NULL */
                values = aio_suspend(list, 1, NULL);
                error_disk = errno;
        }
}

int
starpu_unistd_global_test_request(void * async_channel)
{
        struct timespec time_wait_request;
        time_wait_request.tv_sec = 0;
        time_wait_request.tv_nsec = 0;

        struct _starpu_async_channel * channel = (struct _starpu_async_channel *) async_channel;
        const struct aiocb * aiocb = &channel->event.disk_event._starpu_aiocb_disk;
        const struct aiocb * list[1];
        list[0] = aiocb;
        int values = -1;
        int error_disk = EAGAIN;

        /* Wait the answer of the request */
        values = aio_suspend(list, 1, &time_wait_request);
        error_disk = errno;
        /* request is finished */
        if (values == 0)
                return 1;
        /* values == -1 */
        if (error_disk == EAGAIN)
                return 0;
        /* an error occured */
        STARPU_ABORT();
}
#endif
