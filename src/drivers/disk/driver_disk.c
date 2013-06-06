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

#include <starpu.h>
#include <starpu_profiling.h>

int _starpu_disk_copy_src_to_disk(void * src, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	STARPU_ASSERT(starpu_node_get_kind(src_node) == STARPU_CPU_RAM);

	starpu_disk_write(dst_node, dst, src, dst_offset, size);
	return 0;
}


int _starpu_disk_copy_disk_to_src(void * src, size_t src_offset, unsigned src_node, void * dst, unsigned dst_node, size_t size)
{
	STARPU_ASSERT(starpu_node_get_kind(dst_node) == STARPU_CPU_RAM);

	starpu_disk_read(src_node, src, dst, src_offset, size);
	return 0;
}


int _starpu_disk_copy_disk_to_disk(void * src, size_t src_offset, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	void * tmp = malloc(size*sizeof(char));

	starpu_disk_read(src_node, src, tmp, src_offset, size);

	starpu_disk_write(dst_node, dst, tmp, dst_offset, size);

	free(tmp);

	return 0;
}
