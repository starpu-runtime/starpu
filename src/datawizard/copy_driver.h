/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

#ifndef __COPY_DRIVER_H__
#define __COPY_DRIVER_H__

/** @file */

#ifdef HAVE_AIO_H
#include <aio.h>
#endif

#include <common/config.h>
#include <common/list.h>

#pragma GCC visibility push(hidden)

#ifdef __cplusplus
extern "C"
{
#endif

struct _starpu_data_request;
struct _starpu_data_replicate;

enum _starpu_may_alloc
{
	_STARPU_DATAWIZARD_DO_NOT_ALLOC,
	_STARPU_DATAWIZARD_DO_ALLOC,
	_STARPU_DATAWIZARD_ONLY_FAST_ALLOC
};


LIST_TYPE(_starpu_disk_backend_event,
	void *backend_event;
);

struct _starpu_disk_event
{
	unsigned memory_node;
	unsigned node;
	struct _starpu_disk_backend_event_list * requests;

	void * ptr;
	size_t size;
	starpu_data_handle_t handle;
};

/** this is a structure that can be queried to see whether an asynchronous
 * transfer has terminated or not */
union _starpu_async_channel_event
{
	char data[40];
};

struct _starpu_async_channel
{
	union _starpu_async_channel_event event;
	const struct _starpu_node_ops *node_ops;
	/** Which node to polling when needing ACK msg */
	struct _starpu_mp_node *polling_node_sender;
	struct _starpu_mp_node *polling_node_receiver;
	/** Used to know if the acknowlegdment msg is arrived from sinks */
	volatile int starpu_mp_common_finished_sender;
	volatile int starpu_mp_common_finished_receiver;
};

void _starpu_wake_all_blocked_workers_on_node(unsigned nodeid);

int _starpu_driver_copy_data_1_to_1(starpu_data_handle_t handle,
				    struct _starpu_data_replicate *src_replicate,
				    struct _starpu_data_replicate *dst_replicate,
				    unsigned donotread,
				    struct _starpu_data_request *req,
				    enum _starpu_may_alloc may_alloc,
				    enum starpu_is_prefetch prefetch);

int _starpu_copy_interface_any_to_any(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

unsigned _starpu_driver_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif // __COPY_DRIVER_H__
