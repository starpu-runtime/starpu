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

#ifndef __DRIVER_DISK_H__
#define __DRIVER_DISK_H__

int _starpu_disk_copy_src_to_disk(void * src, size_t src_offset, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size);

int _starpu_disk_copy_disk_to_src(void * src, size_t src_offset, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size);

int _starpu_disk_copy_disk_to_disk(void * src, size_t src_offset, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size);

#endif
