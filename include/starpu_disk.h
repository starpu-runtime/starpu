
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

/* list of functions to use on disk */
struct starpu_disk_ops {
 	 void *  (*alloc)  (void *base, size_t size);
	 void    (*free)   (void *base, void *obj, size_t size);
	 void *  (*open)   (void *base, void *pos, size_t size);     /* open an existing file */
	 void    (*close)  (void *base, void *obj, size_t size);
	ssize_t  (*read)   (void *base, void *obj, void *buf, off_t offset, size_t size, void * async); 
	ssize_t  (*write)  (void *base, void *obj, const void *buf, off_t offset, size_t size, void * async); 
	/* readv, writev, read2d, write2d, etc. */
	 void *  (*plug)   (void *parameter);
	 void    (*unplug) (void *base);
	  int    (*copy)   (void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size);
	  int    (*bandwidth) (unsigned node);
};


/* Posix functions to use disk memory */
extern struct starpu_disk_ops starpu_disk_stdio_ops;
extern struct starpu_disk_ops starpu_disk_unistd_ops;
extern struct starpu_disk_ops starpu_disk_unistd_o_direct_ops;

/*functions to add an existing memory */
void starpu_disk_close(unsigned node, void *obj, size_t size);

void * starpu_disk_open(unsigned node, void *pos, size_t size);

/* interface to create and to free a memory disk */
int starpu_disk_register(struct starpu_disk_ops * func, void *parameter, size_t size);

#endif /* __STARPU_DISK_H__ */
