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

#include <starpu.h>
#include <core/disk.h>
#include <starpu_profiling.h>
#include <drivers/disk/driver_disk.h>
#include <drivers/cpu/driver_cpu.h>
#include <datawizard/coherency.h>

int _starpu_disk_copy_src_to_disk(void * src, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size, void * async_channel)
{
	STARPU_ASSERT(starpu_node_get_kind(src_node) == STARPU_CPU_RAM);

	return _starpu_disk_write(src_node, dst_node, dst, src, dst_offset, size, async_channel);
}

int _starpu_disk_copy_disk_to_src(void * src, size_t src_offset, unsigned src_node, void * dst, unsigned dst_node, size_t size, void * async_channel)
{
	STARPU_ASSERT(starpu_node_get_kind(dst_node) == STARPU_CPU_RAM);

	return _starpu_disk_read(src_node, dst_node, src, dst, src_offset, size, async_channel);
}

int _starpu_disk_copy_disk_to_disk(void * src, size_t src_offset, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size, void * async_channel)
{
	STARPU_ASSERT(starpu_node_get_kind(src_node) == STARPU_DISK_RAM && starpu_node_get_kind(dst_node) == STARPU_DISK_RAM);

	return _starpu_disk_copy(src_node, src, src_offset, dst_node, dst, dst_offset, size, async_channel);
}

unsigned _starpu_disk_test_request_completion(struct _starpu_async_channel *async_channel)
{
	unsigned success = starpu_disk_test_request(async_channel);
	if (async_channel->event.disk_event.ptr != NULL && success)
	{
		if (async_channel->event.disk_event.handle != NULL)
		{
			/* read is finished, we can already unpack */
			async_channel->event.disk_event.handle->ops->unpack_data(async_channel->event.disk_event.handle, async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size);
		}
		else
		{
			/* write is finished, ptr was allocated in pack_data */
			_starpu_free_flags_on_node(async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size, 0);
		}
	}
	return success;
}

void _starpu_disk_wait_request_completion(struct _starpu_async_channel *async_channel)
{
	starpu_disk_wait_request(async_channel);
	if (async_channel->event.disk_event.ptr != NULL)
	{
		if (async_channel->event.disk_event.handle != NULL)
		{
			/* read is finished, we can already unpack */
			async_channel->event.disk_event.handle->ops->unpack_data(async_channel->event.disk_event.handle, async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size);
		}
		else
		{
			/* write is finished, ptr was allocated in pack_data */
			_starpu_free_flags_on_node(async_channel->event.disk_event.node, async_channel->event.disk_event.ptr, async_channel->event.disk_event.size, 0);
		}
	}
}

int _starpu_disk_copy_interface_from_disk_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_DISK_RAM && dst_kind == STARPU_CPU_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (req && !starpu_asynchronous_copy_disabled())
	{
		req->async_channel.node_ops = &_starpu_driver_disk_node_ops;
		req->async_channel.event.disk_event.requests = NULL;
		req->async_channel.event.disk_event.ptr = NULL;
		req->async_channel.event.disk_event.handle = NULL;
	}
	if(copy_methods->any_to_any)
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled()  ? &req->async_channel : NULL);
	else
	{
		void *obj = starpu_data_handle_to_pointer(handle, src_node);
		void * ptr = NULL;
		size_t size = 0;
		ret = _starpu_disk_full_read(src_node, dst_node, obj, &ptr, &size, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
		if (ret == 0)
		{
			/* read is already finished, we can already unpack */
			handle->ops->unpack_data(handle, dst_node, ptr, size);
		}
		else if (ret == -EAGAIN)
		{
			STARPU_ASSERT(req);
			req->async_channel.event.disk_event.ptr = ptr;
			req->async_channel.event.disk_event.node = dst_node;
			req->async_channel.event.disk_event.size = size;
			req->async_channel.event.disk_event.handle = handle;
		}
		STARPU_ASSERT(ret == 0 || ret == -EAGAIN);
	}

	return ret;
}

int _starpu_disk_copy_interface_from_disk_to_disk(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_DISK_RAM && dst_kind == STARPU_DISK_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (req && !starpu_asynchronous_copy_disabled())
	{
		req->async_channel.node_ops = &_starpu_driver_disk_node_ops;
		req->async_channel.event.disk_event.requests = NULL;
		req->async_channel.event.disk_event.ptr = NULL;
		req->async_channel.event.disk_event.handle = NULL;
	}
	ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
	return ret;
}

int _starpu_disk_copy_interface_from_cpu_to_disk(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_DISK_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (req && !starpu_asynchronous_copy_disabled())
	{
		req->async_channel.node_ops = &_starpu_driver_disk_node_ops;
		req->async_channel.event.disk_event.requests = NULL;
		req->async_channel.event.disk_event.ptr = NULL;
		req->async_channel.event.disk_event.handle = NULL;
	}

	if(copy_methods->any_to_any)
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
	else
	{
		void *obj = starpu_data_handle_to_pointer(handle, dst_node);
		void * ptr = NULL;
		starpu_ssize_t size = 0;
		handle->ops->pack_data(handle, src_node, &ptr, &size);
		ret = _starpu_disk_full_write(src_node, dst_node, obj, ptr, size, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
		if (ret == 0)
		{
			/* write is already finished, ptr was allocated in pack_data */
			_starpu_free_flags_on_node(src_node, ptr, size, 0);
		}
		else if (ret == -EAGAIN)
		{
			STARPU_ASSERT(req);
			req->async_channel.event.disk_event.ptr = ptr;
			req->async_channel.event.disk_event.node = src_node;
			req->async_channel.event.disk_event.size = size;
		}
		STARPU_ASSERT(ret == 0 || ret == -EAGAIN);
	}

	return ret;
}

int _starpu_disk_copy_data_from_disk_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_DISK_RAM && dst_kind == STARPU_CPU_RAM);

	return _starpu_disk_copy_disk_to_src((void*) src, src_offset, src_node,
					     (void*) (dst + dst_offset), dst_node,
					     size, async_channel);
}

int _starpu_disk_copy_data_from_disk_to_disk(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_DISK_RAM && dst_kind == STARPU_DISK_RAM);

	return _starpu_disk_copy_disk_to_disk((void*) src, src_offset, src_node,
					      (void*) dst, dst_offset, dst_node,
					      size, async_channel);
}

int _starpu_disk_copy_data_from_cpu_to_disk(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_DISK_RAM);

	return _starpu_disk_copy_src_to_disk((void*) (src + src_offset), src_node,
					     (void*) dst, dst_offset, dst_node,
					     size, async_channel);
}

int _starpu_disk_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	/* Each worker can manage disks but disk <-> disk is not always allowed */
	switch (starpu_node_get_kind(handling_node))
	{
		case STARPU_CPU_RAM:
			return 1;
		case STARPU_DISK_RAM:
			return _starpu_disk_can_copy(node, handling_node);
		default:
			return 0;
	}
}

uintptr_t _starpu_disk_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	(void) flags;
	uintptr_t addr = 0;
	addr = (uintptr_t) _starpu_disk_alloc(dst_node, size);
	return addr;
}

void _starpu_disk_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	(void) flags;
	_starpu_disk_free(dst_node, (void *) addr , size);
}

struct _starpu_node_ops _starpu_driver_disk_node_ops =
{
	.copy_interface_to[STARPU_UNUSED] = NULL,
	.copy_interface_to[STARPU_CPU_RAM] = _starpu_disk_copy_interface_from_disk_to_cpu,
	.copy_interface_to[STARPU_CUDA_RAM] = NULL,
	.copy_interface_to[STARPU_OPENCL_RAM] = NULL,
	.copy_interface_to[STARPU_DISK_RAM] = _starpu_disk_copy_interface_from_disk_to_disk,
	.copy_interface_to[STARPU_MIC_RAM] = NULL,
	.copy_interface_to[STARPU_MPI_MS_RAM] = NULL,

	.copy_data_to[STARPU_UNUSED] = NULL,
	.copy_data_to[STARPU_CPU_RAM] = _starpu_disk_copy_data_from_disk_to_cpu,
	.copy_data_to[STARPU_CUDA_RAM] = NULL,
	.copy_data_to[STARPU_OPENCL_RAM] = NULL,
	.copy_data_to[STARPU_DISK_RAM] = _starpu_disk_copy_data_from_disk_to_disk,
	.copy_data_to[STARPU_MIC_RAM] = NULL,
	.copy_data_to[STARPU_MPI_MS_RAM] = NULL,

	/* TODO: copy2D/3D? */

	.wait_request_completion = _starpu_disk_wait_request_completion,
	.test_request_completion = _starpu_disk_test_request_completion,
	.is_direct_access_supported = _starpu_disk_is_direct_access_supported,
	.malloc_on_node = _starpu_disk_malloc_on_node,
	.free_on_node = _starpu_disk_free_on_node,
	.name = "disk driver"
};
