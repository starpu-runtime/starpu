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

#ifndef __DISK_H__
#define __DISK_H__

/** @file */

#define STARPU_DISK_ALL 1
#define STARPU_DISK_NO_RECLAIM 2

#ifdef __cplusplus
extern "C"
{
#endif

#include <datawizard/copy_driver.h>
#include <datawizard/malloc.h>

#pragma GCC visibility push(hidden)

/** interface to manipulate memory disk */
void * _starpu_disk_alloc (int devid, size_t size) STARPU_ATTRIBUTE_MALLOC;

void _starpu_disk_free (int devid, void *obj, size_t size);
/** src_dev is a disk device, dst_dev is for the moment the STARPU_MAIN_RAM */
int _starpu_disk_read(int src_dev, int dst_dev, void *obj, void *buf, off_t offset, size_t size, struct _starpu_async_channel * async_channel);
/** src_dev is for the moment the STARU_MAIN_RAM, dst_dev is a disk device */
int _starpu_disk_write(int src_dev, int dst_dev, void *obj, void *buf, off_t offset, size_t size, struct _starpu_async_channel * async_channel);

int _starpu_disk_full_read(int src_dev, int dst_dev, void * obj, void ** ptr, size_t * size, struct _starpu_async_channel * async_channel);
int _starpu_disk_full_write(int src_dev, int dst_dev, void * obj, void * ptr, size_t size, struct _starpu_async_channel * async_channel);

int _starpu_disk_copy(int dev_src, void* obj_src, off_t offset_src, int dev_dst, void* obj_dst, off_t offset_dst, size_t size, struct _starpu_async_channel * async_channel);

/** force the request to compute */
void starpu_disk_wait_request(struct _starpu_async_channel *async_channel);
/** return 1 if the request is finished, 0 if not finished */
int starpu_disk_test_request(struct _starpu_async_channel *async_channel);
void starpu_disk_free_request(struct _starpu_async_channel *async_channel);

/** interface to compare memory disk */
int _starpu_disk_can_copy(int devid1, int devid2);

/** change disk flag */
void _starpu_set_disk_flag(int devid, int flag);
int _starpu_get_disk_flag(int devid);

/** unregister disk */
void _starpu_disk_unregister(void);

void _starpu_swap_init(void);

static inline struct _starpu_disk_event *_starpu_disk_get_event(union _starpu_async_channel_event *_event)
{
	struct _starpu_disk_event *event;
	STARPU_STATIC_ASSERT(sizeof(*event) <= sizeof(*_event));
	event = (struct _starpu_disk_event *) _event;
	return event;
}

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* __DISK_H__ */
