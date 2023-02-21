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

#include <starpu.h>
#include <core/disk.h>
#include <starpu_profiling.h>
#include <drivers/disk/driver_disk.h>
#include <drivers/cpu/driver_cpu.h>
#include <datawizard/coherency.h>
#include <datawizard/memory_nodes.h>

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "Disk",
	.worker_archtype = (enum starpu_worker_archtype) -1,
	.ops = &_starpu_driver_disk_node_ops,
};

void _starpu_disk_preinit(void)
{
	_starpu_memory_driver_info_register(STARPU_DISK_RAM, &memory_driver_info);
}

uintptr_t _starpu_disk_malloc_on_device(int dst_dev, size_t size, int flags)
{
	(void) flags;
	uintptr_t addr = 0;
	addr = (uintptr_t) _starpu_disk_alloc(dst_dev, size);
	return addr;
}

void _starpu_disk_free_on_device(int dst_dev, uintptr_t addr, size_t size, int flags)
{
	(void) flags;
	_starpu_disk_free(dst_dev, (void *) addr , size);
}

int _starpu_disk_copy_src_to_disk(void * src, int src_dev, void * dst, size_t dst_offset, int dst_dev, size_t size, void * async_channel)
{
	return _starpu_disk_write(src_dev, dst_dev, dst, src, dst_offset, size, async_channel);
}

int _starpu_disk_copy_disk_to_src(void * src, size_t src_offset, int src_dev, void * dst, int dst_dev, size_t size, void * async_channel)
{
	return _starpu_disk_read(src_dev, dst_dev, src, dst, src_offset, size, async_channel);
}

int _starpu_disk_copy_disk_to_disk(void * src, size_t src_offset, int src_dev, void * dst, size_t dst_offset, int dst_dev, size_t size, void * async_channel)
{
	return _starpu_disk_copy(src_dev, src, src_offset, dst_dev, dst, dst_offset, size, async_channel);
}

unsigned _starpu_disk_test_request_completion(struct _starpu_async_channel *async_channel)
{
	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&async_channel->event);
	unsigned success = starpu_disk_test_request(async_channel);
	if (disk_event->ptr != NULL && success)
	{
		if (disk_event->handle != NULL)
		{
			/* read is finished, we can already unpack */
			disk_event->handle->ops->unpack_data(disk_event->handle, disk_event->node, disk_event->ptr, disk_event->size);
		}
		else
		{
			/* write is finished, ptr was allocated in pack_data */
			_starpu_free_flags_on_node(disk_event->node, disk_event->ptr, disk_event->size, 0);
		}
	}
	return success;
}

void _starpu_disk_wait_request_completion(struct _starpu_async_channel *async_channel)
{
	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&async_channel->event);
	starpu_disk_wait_request(async_channel);
	if (disk_event->ptr != NULL)
	{
		if (disk_event->handle != NULL)
		{
			/* read is finished, we can already unpack */
			disk_event->handle->ops->unpack_data(disk_event->handle, disk_event->node, disk_event->ptr, disk_event->size);
		}
		else
		{
			/* write is finished, ptr was allocated in pack_data */
			_starpu_free_flags_on_node(disk_event->node, disk_event->ptr, disk_event->size, 0);
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
	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&req->async_channel.event);

	if (req && !starpu_asynchronous_copy_disabled())
	{
		req->async_channel.node_ops = &_starpu_driver_disk_node_ops;
		disk_event->requests = NULL;
		disk_event->ptr = NULL;
		disk_event->handle = NULL;
	}
	if(copy_methods->any_to_any)
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled()  ? &req->async_channel : NULL);
	else
	{
		void *obj = starpu_data_handle_to_pointer(handle, src_node);
		void * ptr = NULL;
		size_t size = 0;
		int src_dev = starpu_memory_node_get_devid(src_node);
		int dst_dev = starpu_memory_node_get_devid(dst_node);
		ret = _starpu_disk_full_read(src_dev, dst_dev, obj, &ptr, &size, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
		if (ret == 0)
		{
			/* read is already finished, we can already unpack */
			handle->ops->unpack_data(handle, dst_node, ptr, size);
		}
		else if (ret == -EAGAIN)
		{
			STARPU_ASSERT(req);
			disk_event->ptr = ptr;
			disk_event->node = dst_node;
			disk_event->size = size;
			disk_event->handle = handle;
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
		struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&req->async_channel.event);
		req->async_channel.node_ops = &_starpu_driver_disk_node_ops;
		disk_event->requests = NULL;
		disk_event->ptr = NULL;
		disk_event->handle = NULL;
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
	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&req->async_channel.event);

	if (req && !starpu_asynchronous_copy_disabled())
	{
		req->async_channel.node_ops = &_starpu_driver_disk_node_ops;
		disk_event->requests = NULL;
		disk_event->ptr = NULL;
		disk_event->handle = NULL;
	}

	if(copy_methods->any_to_any)
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
	else
	{
		void *obj = starpu_data_handle_to_pointer(handle, dst_node);
		void * ptr = NULL;
		starpu_ssize_t size = 0;
		handle->ops->pack_data(handle, src_node, &ptr, &size);
		int src_dev = starpu_memory_node_get_devid(src_node);
		int dst_dev = starpu_memory_node_get_devid(dst_node);
		ret = _starpu_disk_full_write(src_dev, dst_dev, obj, ptr, size, req && !starpu_asynchronous_copy_disabled() ? &req->async_channel : NULL);
		if (ret == 0)
		{
			/* write is already finished, ptr was allocated in pack_data */
			_starpu_free_flags_on_node(src_node, ptr, size, 0);
		}
		else if (ret == -EAGAIN)
		{
			STARPU_ASSERT(req);
			disk_event->ptr = ptr;
			disk_event->node = src_node;
			disk_event->size = size;
		}
		STARPU_ASSERT(ret == 0 || ret == -EAGAIN);
	}

	return ret;
}

int _starpu_disk_copy_data_from_disk_to_cpu(uintptr_t src, size_t src_offset, int src_dev, uintptr_t dst, size_t dst_offset, int dst_dev, size_t size, struct _starpu_async_channel *async_channel)
{
	return _starpu_disk_copy_disk_to_src((void*) src, src_offset, src_dev,
					     (void*) (dst + dst_offset), dst_dev,
					     size, async_channel);
}

int _starpu_disk_copy_data_from_disk_to_disk(uintptr_t src, size_t src_offset, int src_dev, uintptr_t dst, size_t dst_offset, int dst_dev, size_t size, struct _starpu_async_channel *async_channel)
{
	return _starpu_disk_copy_disk_to_disk((void*) src, src_offset, src_dev,
					      (void*) dst, dst_offset, dst_dev,
					      size, async_channel);
}

int _starpu_disk_copy_data_from_cpu_to_disk(uintptr_t src, size_t src_offset, int src_dev, uintptr_t dst, size_t dst_offset, int dst_dev, size_t size, struct _starpu_async_channel *async_channel)
{
	return _starpu_disk_copy_src_to_disk((void*) (src + src_offset), src_dev,
					     (void*) dst, dst_offset, dst_dev,
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
		{
			int dev = starpu_memory_node_get_devid(node);
			int handling_dev = starpu_memory_node_get_devid(node);
			return _starpu_disk_can_copy(dev, handling_dev);
		}
		default:
			return 0;
	}
}

struct _starpu_node_ops _starpu_driver_disk_node_ops =
{
	.name = "disk driver",

	.malloc_on_device = _starpu_disk_malloc_on_device,
	.free_on_device = _starpu_disk_free_on_device,

	.is_direct_access_supported = _starpu_disk_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_disk_copy_interface_from_disk_to_cpu,
	.copy_interface_to[STARPU_DISK_RAM] = _starpu_disk_copy_interface_from_disk_to_disk,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_disk_copy_interface_from_cpu_to_disk,
	.copy_interface_from[STARPU_DISK_RAM] = _starpu_disk_copy_interface_from_disk_to_disk,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_disk_copy_data_from_disk_to_cpu,
	.copy_data_to[STARPU_DISK_RAM] = _starpu_disk_copy_data_from_disk_to_disk,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_disk_copy_data_from_cpu_to_disk,
	.copy_data_from[STARPU_DISK_RAM] = _starpu_disk_copy_data_from_disk_to_disk,

	/* TODO: copy2D/3D? */

	.wait_request_completion = _starpu_disk_wait_request_completion,
	.test_request_completion = _starpu_disk_test_request_completion,
};
