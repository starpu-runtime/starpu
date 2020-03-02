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

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>

#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <starpu.h>
#include <core/disk.h>
#include <core/perfmodel/perfmodel.h>
#include <core/disk_ops/unistd/disk_unistd_global.h>

/* ------------------- use UNISTD to write on disk -------------------  */

/* allocation memory on disk */
static void *starpu_unistd_o_direct_alloc(void *base, size_t size)
{
        struct starpu_unistd_global_obj *obj;
	_STARPU_MALLOC(obj, sizeof(struct starpu_unistd_global_obj));
        /* only flags change between unistd and unistd_o_direct */
        obj->flags = O_RDWR | O_DIRECT | O_BINARY;
        return starpu_unistd_global_alloc (obj, base, size);
}

/* open an existing memory on disk */
static void *starpu_unistd_o_direct_open(void *base, void *pos, size_t size)
{
        struct starpu_unistd_global_obj *obj;
	_STARPU_MALLOC(obj, sizeof(struct starpu_unistd_global_obj));
        /* only flags change between unistd and unistd_o_direct */
        obj->flags = O_RDWR | O_DIRECT | O_BINARY;
        return starpu_unistd_global_open (obj, base, pos, size);
}

/* read the memory disk */
static int starpu_unistd_o_direct_read(void *base, void *obj, void *buf, off_t offset, size_t size)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "You can only read a multiple of page size %u Bytes (Here %d)", getpagesize(), (int) size);

	STARPU_ASSERT_MSG((((uintptr_t) buf) % getpagesize()) == 0, "You have to use starpu_malloc function to get aligned buffers for the unistd_o_direct variant");

	return starpu_unistd_global_read (base, obj, buf, offset, size);
}

/* write on the memory disk */
static int starpu_unistd_o_direct_write(void *base, void *obj, const void *buf, off_t offset, size_t size)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "You can only write a multiple of page size %u Bytes (Here %d)", getpagesize(), (int) size);

	STARPU_ASSERT_MSG((((uintptr_t)buf) % getpagesize()) == 0, "You have to use starpu_malloc function to get aligned buffers for the unistd_o_direct variant");

	return starpu_unistd_global_write (base, obj, buf, offset, size);
}

/* create a new copy of parameter == base */
static void *starpu_unistd_o_direct_plug(void *parameter, starpu_ssize_t size)
{
	starpu_malloc_set_align(getpagesize());

	return starpu_unistd_global_plug (parameter, size);
}

#if defined(HAVE_AIO_H) || defined(HAVE_LIBAIO_H)
void *starpu_unistd_o_direct_global_async_read(void *base, void *obj, void *buf, off_t offset, size_t size)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "The unistd_o_direct variant can only read a multiple of page size %lu Bytes (Here %lu). Use the non-o_direct unistd variant if your data is not a multiple of %lu",
			(unsigned long) getpagesize(), (unsigned long) size, (unsigned long) getpagesize());

	STARPU_ASSERT_MSG((((uintptr_t) buf) % getpagesize()) == 0, "You have to use starpu_malloc function to get aligned buffers for the unistd_o_direct variant");

	return starpu_unistd_global_async_read (base, obj, buf, offset, size);
}

void *starpu_unistd_o_direct_global_async_write(void *base, void *obj, void *buf, off_t offset, size_t size)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "The unistd_o_direct variant can only write a multiple of page size %lu Bytes (Here %lu). Use the non-o_direct unistd variant if your data is not a multiple of %lu",
			(unsigned long) getpagesize(), (unsigned long) size, (unsigned long) getpagesize());

	STARPU_ASSERT_MSG((((uintptr_t)buf) % getpagesize()) == 0, "You have to use starpu_malloc function to get aligned buffers for the unistd_o_direct variant");

	return starpu_unistd_global_async_write (base, obj, buf, offset, size);
}
#endif

#ifdef STARPU_UNISTD_USE_COPY
void *  starpu_unistd_o_direct_global_copy(void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size)
{

	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "The unistd_o_direct variant can only write a multiple of page size %lu Bytes (Here %lu). Use the non-o_direct unistd variant if your data is not a multiple of %lu",
			(unsigned long) getpagesize(), (unsigned long) size, (unsigned long) getpagesize());

	return starpu_unistd_global_copy(base_src, obj_src, offset_src, base_dst, obj_dst, offset_dst, size);
}

#endif

int starpu_unistd_o_direct_global_full_write(void *base, void *obj, void *ptr, size_t size)
{
	STARPU_ASSERT_MSG((size % getpagesize()) == 0, "The unistd_o_direct variant can only write a multiple of page size %lu Bytes (Here %lu). Use the non-o_direct unistd variant if your data is not a multiple of %lu",
			(unsigned long) getpagesize(), (unsigned long) size, (unsigned long) getpagesize());

	STARPU_ASSERT_MSG((((uintptr_t)ptr) % getpagesize()) == 0, "You have to use starpu_malloc function to get aligned buffers for the unistd_o_direct variant");

	return starpu_unistd_global_full_write(base, obj, ptr, size);
}

struct starpu_disk_ops starpu_disk_unistd_o_direct_ops =
{
	.alloc = starpu_unistd_o_direct_alloc,
	.free = starpu_unistd_global_free,
	.open = starpu_unistd_o_direct_open,
	.close = starpu_unistd_global_close,
	.read = starpu_unistd_o_direct_read,
	.write = starpu_unistd_o_direct_write,
	.plug = starpu_unistd_o_direct_plug,
	.unplug = starpu_unistd_global_unplug,
#ifdef STARPU_UNISTD_USE_COPY
	.copy = starpu_unistd_o_direct_global_copy,
#else
	.copy = NULL,
#endif
	.bandwidth = get_unistd_global_bandwidth_between_disk_and_main_ram,
#if defined(HAVE_AIO_H) || defined(HAVE_LIBAIO_H)
        .async_read = starpu_unistd_o_direct_global_async_read,
        .async_write = starpu_unistd_o_direct_global_async_write,
        .wait_request = starpu_unistd_global_wait_request,
        .test_request = starpu_unistd_global_test_request,
	.free_request = starpu_unistd_global_free_request,
	.async_full_read = starpu_unistd_global_async_full_read,
	.async_full_write = starpu_unistd_global_async_full_write,
#endif
	.full_read = starpu_unistd_global_full_read,
	.full_write = starpu_unistd_o_direct_global_full_write
};
