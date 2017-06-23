/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Corentin Salingue
 * Copyright (C) 2015, 2016, 2017  CNRS
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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <common/config.h>
#include <core/debug.h>
#include <core/disk.h>
#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>
#include <core/topology.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memalloc.h>

#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>
#include <profiling/profiling.h>
#include <common/uthash.h>

struct disk_register
{
	unsigned node;
	void *base;
	struct starpu_disk_ops *functions;
	/* disk condition (1 = all authorizations,  */
	int flag;
};

static int add_disk_in_list(unsigned node, struct starpu_disk_ops *func, void *base);
static int get_location_with_node(unsigned node);

static struct disk_register **disk_register_list = NULL;
static int disk_number = -1;
static int size_register_list = 2;

int starpu_disk_swap_node = -1;

static void add_async_event(struct _starpu_async_channel * channel, void * event)
{
        if (!event)
                return; 

        if (channel->event.disk_event.requests == NULL)
        {
                channel->event.disk_event.requests = _starpu_disk_backend_event_list_new();
        }

        struct _starpu_disk_backend_event * disk_event = _starpu_disk_backend_event_new();
        disk_event->backend_event = event;

        /* Store event at the end of the list */
        _starpu_disk_backend_event_list_push_back(channel->event.disk_event.requests, disk_event);
}

int starpu_disk_register(struct starpu_disk_ops *func, void *parameter, starpu_ssize_t size)
{
	STARPU_ASSERT_MSG(size < 0 || size >= STARPU_DISK_SIZE_MIN,"Minimum disk size is %u Bytes ! (Here %u) \n", (int) STARPU_DISK_SIZE_MIN, (int) size);
	/* register disk */
	unsigned memory_node = _starpu_memory_node_register(STARPU_DISK_RAM, 0);

	_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
	_starpu_register_bus(memory_node, STARPU_MAIN_RAM);

	/* connect disk */
	void *base = func->plug(parameter, size);

	/* remember it */
	int n STARPU_ATTRIBUTE_UNUSED = add_disk_in_list(memory_node,func,base);

#ifdef STARPU_SIMGRID
	char name[16];
	snprintf(name, sizeof(name), "DISK%d", n);
	msg_host_t host = _starpu_simgrid_get_host_by_name(name);
	STARPU_ASSERT_MSG(host, "Could not find disk %s in platform file", name);
	_starpu_simgrid_memory_node_set_host(memory_node, host);
#endif

	int ret = func->bandwidth(memory_node);
	/* have a problem with the disk */
	if (ret == 0)
		return -ENOENT;
	if (size >= 0)
		_starpu_memory_manager_set_global_memory_size(memory_node, size);
	return memory_node;
}

void _starpu_disk_unregister(void)
{
	int i;

	/* search disk and delete it */
	for (i = 0; i <= disk_number; ++i)
	{
		_starpu_set_disk_flag(disk_register_list[i]->node, STARPU_DISK_NO_RECLAIM);
		_starpu_free_all_automatically_allocated_buffers(disk_register_list[i]->node);

		/* don't forget to unplug */
		disk_register_list[i]->functions->unplug(disk_register_list[i]->base);
		free(disk_register_list[i]);
	}

	/* no disk in the list -> delete the list */
	disk_number--;

	if (disk_register_list != NULL && disk_number == -1)
	{
		free(disk_register_list);
		disk_register_list = NULL;
	}
}

/* interface between user and disk memory */

void *_starpu_disk_alloc(unsigned node, size_t size)
{
	int pos = get_location_with_node(node);
	return disk_register_list[pos]->functions->alloc(disk_register_list[pos]->base, size);
}

void _starpu_disk_free(unsigned node, void *obj, size_t size)
{
	int pos = get_location_with_node(node);
	disk_register_list[pos]->functions->free(disk_register_list[pos]->base, obj, size);
}

/* src_node == disk node and dst_node == STARPU_MAIN_RAM */
int _starpu_disk_read(unsigned src_node, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, struct _starpu_async_channel *channel)
{
        void * event;
	int pos = get_location_with_node(src_node);

        if (channel != NULL)
	{
		if (disk_register_list[pos]->functions->async_read == NULL)
			channel = NULL;
		else
		{
			channel->event.disk_event.memory_node = src_node;

			_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
			event = disk_register_list[pos]->functions->async_read(disk_register_list[pos]->base, obj, buf, offset, size);
			_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);

                        add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[pos]->functions->read(disk_register_list[pos]->base, obj, buf, offset, size);
		return 0;
	}
	return -EAGAIN;
}

/* src_node == STARPU_MAIN_RAM and dst_node == disk node */
int _starpu_disk_write(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node, void *obj, void *buf, off_t offset, size_t size, struct _starpu_async_channel *channel)
{
        void * event;
	int pos = get_location_with_node(dst_node);

        if (channel != NULL)
        {
		if (disk_register_list[pos]->functions->async_write == NULL)
			channel = NULL;
		else
                {
			channel->event.disk_event.memory_node = dst_node;

			_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
			event = disk_register_list[pos]->functions->async_write(disk_register_list[pos]->base, obj, buf, offset, size);
        		_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);

                        add_async_event(channel, event);
		}
        }
        /* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
        {
		disk_register_list[pos]->functions->write(disk_register_list[pos]->base, obj, buf, offset, size);
        	return 0;
        }
        return -EAGAIN;
}

int _starpu_disk_copy(unsigned node_src, void *obj_src, off_t offset_src, unsigned node_dst, void *obj_dst, off_t offset_dst, size_t size, struct _starpu_async_channel *channel)
{
	int pos_src = get_location_with_node(node_src);
	int pos_dst = get_location_with_node(node_dst);
	/* both nodes have same copy function */
        void * event;
	channel->event.disk_event.memory_node = node_src;
	event = disk_register_list[pos_src]->functions->copy(disk_register_list[pos_src]->base, obj_src, offset_src,
											       disk_register_list[pos_dst]->base, obj_dst, offset_dst,
											       size);
        add_async_event(channel, event);

	STARPU_ASSERT(event);
	return -EAGAIN;
}

int _starpu_disk_full_read(unsigned src_node, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, void *obj, void **ptr, size_t *size, struct _starpu_async_channel *channel)
{
        void * event;
	int pos = get_location_with_node(src_node);

	if (channel != NULL)
	{
		if (disk_register_list[pos]->functions->async_full_read == NULL)
			channel = NULL;
		else
		{
			channel->event.disk_event.memory_node = src_node;

			_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
			event = disk_register_list[pos]->functions->async_full_read(disk_register_list[pos]->base, obj, ptr, size);
			_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);

                        add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[pos]->functions->full_read(disk_register_list[pos]->base, obj, ptr, size);
		return 0;
	}
	return -EAGAIN;
}

int _starpu_disk_full_write(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node, void *obj, void *ptr, size_t size, struct _starpu_async_channel *channel)
{
        void * event;
	int pos = get_location_with_node(dst_node);

	if (channel != NULL)
	{
		if (disk_register_list[pos]->functions->async_full_write == NULL)
			channel = NULL;
		else
		{
			channel->event.disk_event.memory_node = dst_node;

			_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
			event = disk_register_list[pos]->functions->async_full_write(disk_register_list[pos]->base, obj, ptr, size);
			_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);

                        add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[pos]->functions->full_write(disk_register_list[pos]->base, obj, ptr, size);
		return 0;
	}
	return -EAGAIN;
}

void *starpu_disk_open(unsigned node, void *pos, size_t size)
{
	int position = get_location_with_node(node);
	return disk_register_list[position]->functions->open(disk_register_list[position]->base, pos, size);
}

void starpu_disk_close(unsigned node, void *obj, size_t size)
{
	int position = get_location_with_node(node);
	disk_register_list[position]->functions->close(disk_register_list[position]->base, obj, size);
}

void starpu_disk_wait_request(struct _starpu_async_channel *async_channel)
{
	int position = get_location_with_node(async_channel->event.disk_event.memory_node);

        if (async_channel->event.disk_event.requests != NULL && !_starpu_disk_backend_event_list_empty(async_channel->event.disk_event.requests))
        {
                struct _starpu_disk_backend_event * event = _starpu_disk_backend_event_list_begin(async_channel->event.disk_event.requests);
                struct _starpu_disk_backend_event * next;

                /* Wait all events in the list and remove them */
                while (event != _starpu_disk_backend_event_list_end(async_channel->event.disk_event.requests))
                {
                        next = _starpu_disk_backend_event_list_next(event);

                        disk_register_list[position]->functions->wait_request(event->backend_event);

                        disk_register_list[position]->functions->free_request(event->backend_event);

                        _starpu_disk_backend_event_list_erase(async_channel->event.disk_event.requests, event);

                        _starpu_disk_backend_event_delete(event);

                        event = next;
                }

                /* Remove the list because it doesn't contain any event */
                _starpu_disk_backend_event_list_delete(async_channel->event.disk_event.requests);
                async_channel->event.disk_event.requests = NULL;
        }
}

int starpu_disk_test_request(struct _starpu_async_channel *async_channel)
{
	int position = get_location_with_node(async_channel->event.disk_event.memory_node);

        if (async_channel->event.disk_event.requests != NULL && !_starpu_disk_backend_event_list_empty(async_channel->event.disk_event.requests))
        {
                struct _starpu_disk_backend_event * event = _starpu_disk_backend_event_list_begin(async_channel->event.disk_event.requests);
                struct _starpu_disk_backend_event * next;

                /* Wait all events in the list and remove them */
                while (event != _starpu_disk_backend_event_list_end(async_channel->event.disk_event.requests))
                {
                        next = _starpu_disk_backend_event_list_next(event);

                        int res = disk_register_list[position]->functions->test_request(event->backend_event);

                                if (res)
                                {
                                        disk_register_list[position]->functions->free_request(event->backend_event);

                                        _starpu_disk_backend_event_list_erase(async_channel->event.disk_event.requests, event);

                                        _starpu_disk_backend_event_delete(event);
                                }

                        event = next;
                }

                /* Remove the list because it doesn't contain any event */
                if (_starpu_disk_backend_event_list_empty(async_channel->event.disk_event.requests))
                {
                        _starpu_disk_backend_event_list_delete(async_channel->event.disk_event.requests);
                        async_channel->event.disk_event.requests = NULL;
                }
        }

	return async_channel->event.disk_event.requests == NULL;
}

void starpu_disk_free_request(struct _starpu_async_channel *async_channel)
{
/* It does not have any sense to use this function currently because requests are freed in test of wait functions */
        STARPU_ABORT();

/*	int position = get_location_with_node(async_channel->event.disk_event.memory_node);
	if (async_channel->event.disk_event.backend_event)
		disk_register_list[position]->functions->free_request(async_channel->event.disk_event.backend_event);
*/
}

static int add_disk_in_list(unsigned node,  struct starpu_disk_ops *func, void *base)
{
	int n;

	/* initialization */
	if (disk_register_list == NULL)
	{
		_STARPU_MALLOC(disk_register_list, size_register_list*sizeof(struct disk_register *));
	}
	/* small size -> new size  */
	if ((disk_number+1) > size_register_list)
	{
		size_register_list *= 2;
		_STARPU_REALLOC(disk_register_list, size_register_list*sizeof(struct disk_register *));
	}

	struct disk_register *dr;
	_STARPU_MALLOC(dr, sizeof(struct disk_register));
	dr->node = node;
	dr->base = base;
	dr->flag = STARPU_DISK_ALL;
	dr->functions = func;
	n = ++disk_number;
	disk_register_list[n] = dr;
	return n;
}

static int get_location_with_node(unsigned node)
{
#ifdef STARPU_DEVEL
#warning optimize with a MAXNODE array
#endif
	int i;
	for (i = 0; i <= disk_number; ++i)
		if (disk_register_list[i]->node == node)
			return i;
	STARPU_ASSERT_MSG(false, "Disk node not found !(%u) ", node);
	return -1;
}

int _starpu_is_same_kind_disk(unsigned node1, unsigned node2)
{
	if (starpu_node_get_kind(node1) == STARPU_DISK_RAM && starpu_node_get_kind(node2) == STARPU_DISK_RAM)
	{
		int pos1 = get_location_with_node(node1);
		int pos2 = get_location_with_node(node2);
		if (disk_register_list[pos1]->functions == disk_register_list[pos2]->functions)
			/* they must have a copy function */
			if (disk_register_list[pos1]->functions->copy != NULL)
				return 1;
	}
	return 0;
}

void _starpu_set_disk_flag(unsigned node, int flag)
{
	int pos = get_location_with_node(node);
	disk_register_list[pos]->flag = flag;
}

int _starpu_get_disk_flag(unsigned node)
{
	int pos = get_location_with_node(node);
	return disk_register_list[pos]->flag;
}

void _starpu_swap_init(void)
{
	char *backend;
	char *path;
	starpu_ssize_t size;
	struct starpu_disk_ops *ops;

	path = starpu_getenv("STARPU_DISK_SWAP");
	if (!path)
		return;

	backend = starpu_getenv("STARPU_DISK_SWAP_BACKEND");
	if (!backend)
	{
		_starpu_mkpath(path, S_IRWXU);
		ops = &starpu_disk_unistd_ops;
	}
	else if (!strcmp(backend, "stdio"))
	{
		_starpu_mkpath(path, S_IRWXU);
		ops = &starpu_disk_stdio_ops;
	}
	else if (!strcmp(backend, "unistd"))
	{
		_starpu_mkpath(path, S_IRWXU);
		ops = &starpu_disk_unistd_ops;
	}
	else if (!strcmp(backend, "unistd_o_direct"))
	{
#ifdef STARPU_LINUX_SYS
		_starpu_mkpath(path, S_IRWXU);
		ops = &starpu_disk_unistd_o_direct_ops;
#else
		_STARPU_DISP("Warning: o_direct support is not compiled in, could not enable disk swap");
		return;
#endif

	}
	else if (!strcmp(backend, "leveldb"))
	{
#ifdef STARPU_HAVE_LEVELDB
		ops = &starpu_disk_leveldb_ops;
#else
		_STARPU_DISP("Warning: leveldb support is not compiled in, could not enable disk swap");
		return;
#endif
	}
        else if (!strcmp(backend, "hdf5"))
        {
#ifdef STARPU_HAVE_HDF5
                _starpu_mkpath(path, S_IRWXU);
                ops = &starpu_disk_hdf5_ops;
#else
		_STARPU_DISP("Warning: hdf5 support is not compiled in, could not enable disk swap");
		return;
#endif
        }
	else
	{
		_STARPU_DISP("Warning: unknown disk swap backend %s, could not enable disk swap", backend);
		return;
	}

	size = starpu_get_env_number_default("STARPU_DISK_SWAP_SIZE", -1);

	starpu_disk_swap_node = starpu_disk_register(ops, path, ((size_t) size) << 20);
	if (starpu_disk_swap_node < 0)
	{
		_STARPU_DISP("Warning: could not enable disk swap %s on %s with size %ld, could not enable disk swap", backend, path, (long) size);
		return;
	}
}
