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

#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <core/disk_ops/unistd/disk_unistd_global.h>

/* ------------------- use UNISTD to write on disk -------------------  */

/* allocation memory on disk */
static void *
starpu_unistd_o_direct_alloc (void *base, size_t size)
{
        struct starpu_unistd_global_obj * obj = malloc(sizeof(struct starpu_unistd_global_obj));
        STARPU_ASSERT(obj != NULL);
        /* only flags change between unistd and unistd_o_direct */
        obj->flags = O_RDWR | O_DIRECT;
        return starpu_unistd_global_alloc (obj, base, size);
}

/* open an existing memory on disk */
static void *
starpu_unistd_o_direct_open (void *base, void *pos, size_t size)
{
        struct starpu_unistd_global_obj * obj = malloc(sizeof(struct starpu_unistd_global_obj));
        STARPU_ASSERT(obj != NULL);
        /* only flags change between unistd and unistd_o_direct */
        obj->flags = O_RDWR | O_DIRECT;
        return starpu_unistd_global_open (obj, base, pos, size);

}


/* read the memory disk */
static ssize_t 
starpu_unistd_o_direct_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, void * _starpu_aiocb_disk)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "You can only read a multiple of page size %u Bytes (Here %u)", getpagesize(), (int) size);

	STARPU_ASSERT_MSG((((uintptr_t) buf) % getpagesize()) == 0, "You have to use starpu_malloc function");

	return starpu_unistd_global_read (base, obj, buf, offset, size, _starpu_aiocb_disk);
}


/* write on the memory disk */
static ssize_t 
starpu_unistd_o_direct_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size, void * _starpu_aiocb_disk)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "You can only write a multiple of page size %u Bytes (Here %u)", getpagesize(), (int) size);

	STARPU_ASSERT_MSG((((uintptr_t)buf) % getpagesize()) == 0, "You have to use starpu_malloc function");

	return starpu_unistd_global_write (base, obj, buf, offset, size, _starpu_aiocb_disk);
}


/* create a new copy of parameter == base */
static void * 
starpu_unistd_o_direct_plug (void *parameter)
{
	starpu_malloc_set_align(getpagesize());

	return starpu_unistd_global_plug (parameter);
}

struct starpu_disk_ops starpu_disk_unistd_o_direct_ops = {
	.alloc = starpu_unistd_o_direct_alloc,
	.free = starpu_unistd_global_free,
	.open = starpu_unistd_o_direct_open,
	.close = starpu_unistd_global_close,
	.read = starpu_unistd_o_direct_read,
	.write = starpu_unistd_o_direct_write,
	.plug = starpu_unistd_o_direct_plug,
	.unplug = starpu_unistd_global_unplug,
	.copy = NULL,
	.bandwidth = get_unistd_global_bandwidth_between_disk_and_main_ram
};
