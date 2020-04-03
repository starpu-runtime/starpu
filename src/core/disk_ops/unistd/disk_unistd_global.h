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

#ifndef __DISK_UNISTD_GLOBAL_H__
#define __DISK_UNISTD_GLOBAL_H__

/** @file */

#include <fcntl.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define STARPU_UNISTD_USE_COPY 1
#if !defined(HAVE_COPY_FILE_RANGE) && !defined(__NR_copy_file_range)
#undef STARPU_UNISTD_USE_COPY
#endif

struct starpu_unistd_global_obj
{
        int descriptor;
        char * path;
        size_t size;
	int flags;
	starpu_pthread_mutex_t mutex;
};

void * starpu_unistd_global_alloc (struct starpu_unistd_global_obj * obj, void *base, size_t size);
void starpu_unistd_global_free (void *base, void *obj, size_t size);
void * starpu_unistd_global_open (struct starpu_unistd_global_obj * obj, void *base, void *pos, size_t size);
void starpu_unistd_global_close (void *base, void *obj, size_t size);
int starpu_unistd_global_read (void *base, void *obj, void *buf, off_t offset, size_t size);
int starpu_unistd_global_write (void *base, void *obj, const void *buf, off_t offset, size_t size);
void * starpu_unistd_global_plug (void *parameter, starpu_ssize_t size);
void starpu_unistd_global_unplug (void *base);
int get_unistd_global_bandwidth_between_disk_and_main_ram(unsigned node, void *base);
void* starpu_unistd_global_async_read (void *base, void *obj, void *buf, off_t offset, size_t size);
void* starpu_unistd_global_async_write (void *base, void *obj, void *buf, off_t offset, size_t size);
void * starpu_unistd_global_async_full_write (void * base, void * obj, void * ptr, size_t size);
void * starpu_unistd_global_async_full_read (void * base, void * obj, void ** ptr, size_t * size, unsigned dst_node);
void starpu_unistd_global_wait_request(void * async_channel);
int starpu_unistd_global_test_request(void * async_channel);
void starpu_unistd_global_free_request(void * async_channel);
int starpu_unistd_global_full_read(void *base, void * obj, void ** ptr, size_t * size, unsigned dst_node);
int starpu_unistd_global_full_write (void * base, void * obj, void * ptr, size_t size);
#ifdef STARPU_UNISTD_USE_COPY
void *  starpu_unistd_global_copy(void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size);
#endif
#endif
