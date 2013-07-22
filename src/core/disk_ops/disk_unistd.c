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
starpu_unistd_alloc (void *base, size_t size)
{
        struct starpu_unistd_global_obj * obj = malloc(sizeof(struct starpu_unistd_global_obj));
        STARPU_ASSERT(obj != NULL);
	/* only flags change between unistd and unistd_o_direct */
	obj->flags = O_RDWR | O_BINARY;
	return starpu_unistd_global_alloc (obj, base, size);
}

/* open an existing memory on disk */
static void * 
starpu_unistd_open (void *base, void *pos, size_t size)
{
	struct starpu_unistd_global_obj * obj = malloc(sizeof(struct starpu_unistd_global_obj));
	STARPU_ASSERT(obj != NULL);
	/* only flags change between unistd and unistd_o_direct */
	obj->flags = O_RDWR | O_BINARY;
	return starpu_unistd_global_open (obj, base, pos, size);	

}

struct starpu_disk_ops starpu_disk_unistd_ops = {
	.alloc = starpu_unistd_alloc,
	.free = starpu_unistd_global_free,
	.open = starpu_unistd_open,
	.close = starpu_unistd_global_close,
	.read = starpu_unistd_global_read,
	.write = starpu_unistd_global_write,
	.plug = starpu_unistd_global_plug,
	.unplug = starpu_unistd_global_unplug,
	.copy = NULL,
	.bandwidth = get_unistd_global_bandwidth_between_disk_and_main_ram,
#ifdef HAVE_AIO_H
	.async_read = starpu_unistd_global_async_read,
	.async_write = starpu_unistd_global_async_write,
	.wait_request = starpu_unistd_global_wait_request,
	.test_request = starpu_unistd_global_test_request,
#endif
        .full_read = starpu_unistd_global_full_read,
        .full_write = starpu_unistd_global_full_write
};
