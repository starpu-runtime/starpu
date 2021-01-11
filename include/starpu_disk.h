/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_DISK_H__
#define __STARPU_DISK_H__

#include <sys/types.h>
#include <starpu_config.h>

/* list of functions to use on disk */
struct starpu_disk_ops
{
	 void *  (*plug)   (void *parameter, starpu_ssize_t size);
	 void    (*unplug) (void *base);

	 int    (*bandwidth)    (unsigned node, void *base);

	 void *  (*alloc)  (void *base, size_t size);
	 void    (*free)   (void *base, void *obj, size_t size);

	 void *  (*open)   (void *base, void *pos, size_t size);     /* open an existing file */
	 void    (*close)  (void *base, void *obj, size_t size);

	 int     (*read)   (void *base, void *obj, void *buf, off_t offset, size_t size);
	 int     (*write)  (void *base, void *obj, const void *buf, off_t offset, size_t size);

	 int	(*full_read)    (void * base, void * obj, void ** ptr, size_t * size);
	 int 	(*full_write)   (void * base, void * obj, void * ptr, size_t size);

	 void *  (*async_write)  (void *base, void *obj, void *buf, off_t offset, size_t size);
	 void *  (*async_read)   (void *base, void *obj, void *buf, off_t offset, size_t size);

	 void *	(*async_full_read)    (void * base, void * obj, void ** ptr, size_t * size);
	 void *	(*async_full_write)   (void * base, void * obj, void * ptr, size_t size);

	 void *  (*copy)   (void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size);
	 void   (*wait_request) (void * async_channel);
	 int    (*test_request) (void * async_channel);
	 void   (*free_request)(void * async_channel);

	/* TODO: readv, writev, read2d, write2d, etc. */
};

/* Posix functions to use disk memory */
extern struct starpu_disk_ops starpu_disk_stdio_ops;
extern struct starpu_disk_ops starpu_disk_unistd_ops;
extern struct starpu_disk_ops starpu_disk_unistd_o_direct_ops;
extern struct starpu_disk_ops starpu_disk_leveldb_ops;

void starpu_disk_close(unsigned node, void *obj, size_t size);

void *starpu_disk_open(unsigned node, void *pos, size_t size);

int starpu_disk_register(struct starpu_disk_ops *func, void *parameter, starpu_ssize_t size);

#define STARPU_DISK_SIZE_MIN (16*1024*1024)

extern int starpu_disk_swap_node;

#endif /* __STARPU_DISK_H__ */
