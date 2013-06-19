
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

#ifndef __DISK_H__
#define __DISK_H__

#define SIZE_DISK_MIN (1024*1024)

#define STARPU_DISK_ALL 1
#define STARPU_DISK_NO_RECLAIM 2

/* interface to manipulate memory disk */
void * _starpu_disk_alloc (unsigned node, size_t size);

void _starpu_disk_free (unsigned node, void *obj, size_t size);

void _starpu_disk_close(unsigned node, void *obj, size_t size);

void * _starpu_disk_open(unsigned node, void *pos, size_t size);

ssize_t _starpu_disk_read(unsigned node, void *obj, void *buf, off_t offset, size_t size);

ssize_t _starpu_disk_write(unsigned node, void *obj, const void *buf, off_t offset, size_t size);

int _starpu_disk_copy(unsigned node_src, void* obj_src, off_t offset_src, unsigned node_dst, void* obj_dst, off_t offset_dst, size_t size);

/* interface to compare memory disk */

int _starpu_is_same_kind_disk(unsigned node1, unsigned node2);

/* change disk flag */

void _starpu_set_disk_flag(unsigned node, int flag);
int _starpu_get_disk_flag(unsigned node);

#endif /* __DISK_H__ */
