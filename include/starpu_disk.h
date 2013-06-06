
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

#ifndef __STARPU_DISK_H__
#define __STARPU_DISK_H__

typedef void * (*disk_function)(void *, unsigned);

/* list of functions to use on disk */
struct disk_ops {
 	 void *  (*alloc)  (void *base, size_t size);
	 void    (*free)   (void *base, void *obj, size_t size);
	 void *  (*open)   (void *base, void *pos, size_t size);
	 void    (*close)  (void *base, void *obj, size_t size);
	ssize_t  (*read)   (void *base, void *obj, void *buf, off_t offset, size_t size);        /* ~= pread */
	ssize_t  (*write)  (void *base, void *obj, const void *buf, off_t offset, size_t size); 
	/* readv, writev, read2d, write2d, etc. */
	 void *  (*plug)   (void *parameter);
	 void    (*unplug) (void *base);
	  int    (*copy)   (void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size);
	 void    (*bandwidth) (void *base, unsigned node);
};


/* Posix functions to use disk memory */
extern struct disk_ops write_on_file;


/* interface to create and to free a memory disk */
unsigned starpu_disk_register(struct disk_ops * func, void *parameter, size_t size);

void starpu_disk_unregister(unsigned node);


/* interface to manipulate memory disk */
void * starpu_disk_alloc (unsigned node, size_t size);

void starpu_disk_free (unsigned node, void *obj, size_t size);

ssize_t starpu_disk_read(unsigned node, void *obj, void *buf, off_t offset, size_t size);

ssize_t starpu_disk_write(unsigned node, void *obj, const void *buf, off_t offset, size_t size);

int starpu_disk_copy(unsigned node_src, void* obj_src, off_t offset_src, unsigned node_dst, void* obj_dst, off_t offset_dst, size_t size);

/* interface to compare memory disk */

int starpu_is_same_kind_disk(unsigned node1, unsigned node2);

#endif /* __STARPU_DISK_H__ */
