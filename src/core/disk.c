/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <drivers/disk/driver_disk.h>
#include <profiling/profiling.h>
#include <common/uthash.h>

struct disk_register
{
	void *base;
	struct starpu_disk_ops *functions;
	/* disk condition (1 = all authorizations,  */
	int flag;
};

static int add_disk_in_list(int devid, struct starpu_disk_ops *func, void *base);

static struct disk_register *disk_register_list[STARPU_NMAXDEVS];
static int disk_number = 0;

int starpu_disk_swap_node = -1;

static void add_async_event(struct _starpu_async_channel * channel, void * event)
{
	if (!event)
		return;

	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&channel->event);
	if (disk_event->requests == NULL)
	{
		disk_event->requests = _starpu_disk_backend_event_list_new();
	}

	struct _starpu_disk_backend_event * backend_event = _starpu_disk_backend_event_new();
	backend_event->backend_event = event;

	/* Store event at the end of the list */
	_starpu_disk_backend_event_list_push_back(disk_event->requests, backend_event);
}

int starpu_disk_register(struct starpu_disk_ops *func, void *parameter, starpu_ssize_t size)
{
	STARPU_ASSERT_MSG(size < 0 || size >= STARPU_DISK_SIZE_MIN, "Minimum disk size is %d Bytes ! (Here %d) \n", (int) STARPU_DISK_SIZE_MIN, (int) size);
	/* register disk */
	int disk_device = STARPU_ATOMIC_ADD(&disk_number, 1) - 1;
	unsigned disk_memnode = _starpu_memory_node_register(STARPU_DISK_RAM, disk_device);

	/* Connect the disk memory node to all numa memory nodes */
	int nb_numa_nodes = starpu_memory_nodes_get_numa_count();
	int numa_node;
	for (numa_node = 0; numa_node < nb_numa_nodes; numa_node++)
	{
		_starpu_register_bus(disk_memnode, numa_node);
		_starpu_register_bus(numa_node, disk_memnode);
	}

	/* Any worker can manage disk memnode */
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned worker;
	for (worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		/* But prefer to use only CPU workers if possible */
		if (starpu_worker_get_type(worker) == STARPU_CPU_WORKER)
		{
			struct _starpu_worker *workerarg = &config->workers[worker];
			_starpu_memory_node_add_nworkers(disk_memnode);
			_starpu_worker_drives_memory_node(workerarg, disk_memnode);
		}
	}

	if (!_starpu_memory_node_get_nworkers(disk_memnode))
	{
		/* Bleh, no CPU worker to drive the disk, use non-CPU workers too */
		for (worker = 0; worker < starpu_worker_get_count(); worker++)
		{
			if (starpu_worker_get_type(worker) != STARPU_CPU_WORKER)
			{
				struct _starpu_worker *workerarg = &config->workers[worker];
				_starpu_memory_node_add_nworkers(disk_memnode);
				_starpu_worker_drives_memory_node(workerarg, disk_memnode);
			}
		}
	}

	//Add bus for disk <-> disk copy
	if (func->copy != NULL)
	{
		int disk;
		for (disk = 0; disk < STARPU_NMAXDEVS; disk++)
		{
			if (disk_register_list[disk] != NULL && disk_register_list[disk]->functions->copy != NULL && disk_register_list[disk]->functions->copy == func->copy)
			{
				unsigned node = starpu_memory_devid_find_node(disk, STARPU_DISK_RAM);
				_starpu_register_bus(disk_memnode, node);
				_starpu_register_bus(node, disk_memnode);
			}
		}
	}

	/* connect disk */
	void *base = func->plug(parameter, size);

	/* remember it */
	int n STARPU_ATTRIBUTE_UNUSED = add_disk_in_list(disk_device, func, base);

#ifdef STARPU_SIMGRID
	char name[16];
	snprintf(name, sizeof(name), "DISK%d", n);
	starpu_sg_host_t host = _starpu_simgrid_get_host_by_name(name);
	STARPU_ASSERT_MSG(host, "Could not find disk %s in platform file", name);
	_starpu_simgrid_memory_node_set_host(disk_memnode, host);
#endif

	int ret = func->bandwidth(disk_memnode, base);
	/* have a problem with the disk */
	if (ret == 0)
		return -ENOENT;
	if (size >= 0)
		_starpu_memory_manager_set_global_memory_size(disk_memnode, size);

	_starpu_mem_chunk_disk_register(disk_memnode);

	return disk_memnode;
}

void _starpu_disk_unregister(void)
{
	int i;

	/* search disk and delete it */
	for (i = 0; i < STARPU_NMAXDEVS; ++i)
	{
		if (disk_register_list[i] == NULL)
			continue;

		_starpu_set_disk_flag(i, STARPU_DISK_NO_RECLAIM);
		unsigned node = starpu_memory_devid_find_node(i, STARPU_DISK_RAM);
		_starpu_free_all_automatically_allocated_buffers(node);

		/* don't forget to unplug */
		disk_register_list[i]->functions->unplug(disk_register_list[i]->base);
		free(disk_register_list[i]);
		disk_register_list[i] = NULL;
	}

	/* no disk in the list -> delete the list */

	STARPU_ASSERT_MSG(disk_number == 0, "Some disks are not unregistered !");
}

/* interface between user and disk memory */

void *_starpu_disk_alloc(int devid, size_t size)
{
	return disk_register_list[devid]->functions->alloc(disk_register_list[devid]->base, size);
}

void _starpu_disk_free(int devid, void *obj, size_t size)
{
	disk_register_list[devid]->functions->free(disk_register_list[devid]->base, obj, size);
}

/* src_dev == disk dev and dst_dev == STARPU_MAIN_RAM */
int _starpu_disk_read(int src_dev, int dst_dev STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, struct _starpu_async_channel *channel)
{
	void *event = NULL;

	if (channel != NULL)
	{
		if (disk_register_list[src_dev]->functions->async_read == NULL)
			channel = NULL;
		else
		{
			double start;
			unsigned src_node = starpu_memory_devid_find_node(src_dev, STARPU_DISK_RAM);
			_starpu_disk_get_event(&channel->event)->memory_node = src_node;

			starpu_interface_start_driver_copy_async_devid(src_dev, STARPU_DISK_RAM, dst_dev, STARPU_CPU_RAM, &start);

			event = disk_register_list[src_dev]->functions->async_read(disk_register_list[src_dev]->base, obj, buf, offset, size);
			starpu_interface_end_driver_copy_async_devid(src_dev, STARPU_DISK_RAM, dst_dev, STARPU_CPU_RAM, start);

			add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[src_dev]->functions->read(disk_register_list[src_dev]->base, obj, buf, offset, size);
		return 0;
	}
	return -EAGAIN;
}

/* src_dev == STARPU_MAIN_RAM and dst_dev == disk dev */
int _starpu_disk_write(int src_dev STARPU_ATTRIBUTE_UNUSED, int dst_dev, void *obj, void *buf, off_t offset, size_t size, struct _starpu_async_channel *channel)
{
	void *event = NULL;

	if (channel != NULL)
	{
		if (disk_register_list[dst_dev]->functions->async_write == NULL)
			channel = NULL;
		else
		{
			double start;
			unsigned dst_node = starpu_memory_devid_find_node(dst_dev, STARPU_DISK_RAM);
			_starpu_disk_get_event(&channel->event)->memory_node = dst_node;

			starpu_interface_start_driver_copy_async_devid(src_dev, STARPU_CPU_RAM, dst_dev, STARPU_DISK_RAM, &start);
			event = disk_register_list[dst_dev]->functions->async_write(disk_register_list[dst_dev]->base, obj, buf, offset, size);
			starpu_interface_end_driver_copy_async_devid(src_dev, STARPU_CPU_RAM, dst_dev, STARPU_DISK_RAM, start);

			add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[dst_dev]->functions->write(disk_register_list[dst_dev]->base, obj, buf, offset, size);
		return 0;
	}
	return -EAGAIN;
}

int _starpu_disk_copy(int src_dev, void *obj_src, off_t offset_src, int dst_dev, void *obj_dst, off_t offset_dst, size_t size, struct _starpu_async_channel *channel)
{
	/* both nodes have same copy function */
	void * event = NULL;

	if (channel)
	{
		unsigned src_node = starpu_memory_devid_find_node(src_dev, STARPU_DISK_RAM);
		_starpu_disk_get_event(&channel->event)->memory_node = src_node;
		event = disk_register_list[src_dev]->functions->copy(disk_register_list[src_dev]->base,
								      obj_src, offset_src,
								      disk_register_list[dst_dev]->base,
								      obj_dst, offset_dst, size);
		add_async_event(channel, event);
	}

	/* Something goes wrong with copy disk to disk... */
	if (!event)
	{
		if (channel || starpu_asynchronous_copy_disabled())
			disk_register_list[src_dev]->functions->copy = NULL;

		/* perform a read, and after a write... */
		void * ptr;
		int ret = _starpu_malloc_flags_on_node(STARPU_MAIN_RAM, &ptr, size, 0);
		STARPU_ASSERT_MSG(ret == 0, "Cannot allocate %zu bytes to perform disk to disk operation", size);

		ret = _starpu_disk_read(src_dev, 0, obj_src, ptr, offset_src, size, NULL);
		STARPU_ASSERT_MSG(ret == 0, "Cannot read %zu bytes to perform disk to disk copy", size);
		ret = _starpu_disk_write(0, dst_dev, obj_dst, ptr, offset_dst, size, NULL);
		STARPU_ASSERT_MSG(ret == 0, "Cannot write %zu bytes to perform disk to disk copy", size);

		_starpu_free_flags_on_node(STARPU_MAIN_RAM, ptr, size, 0);

		return 0;
	}

	STARPU_ASSERT(event);
	return -EAGAIN;
}

/* src_dev == disk dev and dst_dev == STARPU_CPU_RAM */
int _starpu_disk_full_read(int src_dev, int dst_dev, void *obj, void **ptr, size_t *size, struct _starpu_async_channel *channel)
{
	void *event = NULL;
	unsigned dst_node = starpu_memory_devid_find_node(dst_dev, STARPU_CPU_RAM);

	if (channel != NULL)
	{
		if (disk_register_list[src_dev]->functions->async_full_read == NULL)
			channel = NULL;
		else
		{
			double start;
			unsigned src_node = starpu_memory_devid_find_node(src_dev, STARPU_DISK_RAM);
			_starpu_disk_get_event(&channel->event)->memory_node = src_node;

			starpu_interface_start_driver_copy_async_devid(src_dev, STARPU_DISK_RAM, dst_dev, STARPU_CPU_RAM, &start);
			event = disk_register_list[src_dev]->functions->async_full_read(disk_register_list[src_dev]->base, obj, ptr, size, dst_node);
			starpu_interface_end_driver_copy_async_devid(src_dev, STARPU_DISK_RAM, dst_dev, STARPU_CPU_RAM, start);

			add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[src_dev]->functions->full_read(disk_register_list[src_dev]->base, obj, ptr, size, dst_node);
		return 0;
	}
	return -EAGAIN;
}

/* src_dev == STARPU_CPU_RAM and dst_dev == disk dev */
int _starpu_disk_full_write(int src_dev STARPU_ATTRIBUTE_UNUSED, int dst_dev, void *obj, void *ptr, size_t size, struct _starpu_async_channel *channel)
{
	void *event = NULL;

	if (channel != NULL)
	{
		if (disk_register_list[dst_dev]->functions->async_full_write == NULL)
			channel = NULL;
		else
		{
			double start;
			unsigned dst_node = starpu_memory_devid_find_node(dst_dev, STARPU_DISK_RAM);
			_starpu_disk_get_event(&channel->event)->memory_node = dst_node;

			starpu_interface_start_driver_copy_async_devid(src_dev, STARPU_CPU_RAM, dst_dev, STARPU_DISK_RAM, &start);
			event = disk_register_list[dst_dev]->functions->async_full_write(disk_register_list[dst_dev]->base, obj, ptr, size);
			starpu_interface_end_driver_copy_async_devid(src_dev, STARPU_CPU_RAM, dst_dev, STARPU_DISK_RAM, start);

			add_async_event(channel, event);
		}
	}
	/* asynchronous request failed or synchronous request is asked */
	if (channel == NULL || !event)
	{
		disk_register_list[dst_dev]->functions->full_write(disk_register_list[dst_dev]->base, obj, ptr, size);
		return 0;
	}
	return -EAGAIN;
}

void *starpu_disk_open(unsigned node, void *pos, size_t size)
{
	int devid = starpu_memory_node_get_devid(node);
	return disk_register_list[devid]->functions->open(disk_register_list[devid]->base, pos, size);
}

void starpu_disk_close(unsigned node, void *obj, size_t size)
{
	int devid = starpu_memory_node_get_devid(node);
	disk_register_list[devid]->functions->close(disk_register_list[devid]->base, obj, size);
}

void starpu_disk_wait_request(struct _starpu_async_channel *async_channel)
{
	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&async_channel->event);
	unsigned node = disk_event->memory_node;
	int devid = starpu_memory_node_get_devid(node);

	if (disk_event->requests != NULL && !_starpu_disk_backend_event_list_empty(disk_event->requests))
	{
		struct _starpu_disk_backend_event * event = _starpu_disk_backend_event_list_begin(disk_event->requests);
		struct _starpu_disk_backend_event * next;

		/* Wait all events in the list and remove them */
		while (event != _starpu_disk_backend_event_list_end(disk_event->requests))
		{
			next = _starpu_disk_backend_event_list_next(event);

			disk_register_list[devid]->functions->wait_request(event->backend_event);

			disk_register_list[devid]->functions->free_request(event->backend_event);

			_starpu_disk_backend_event_list_erase(disk_event->requests, event);

			_starpu_disk_backend_event_delete(event);

			event = next;
		}

		/* Remove the list because it doesn't contain any event */
		_starpu_disk_backend_event_list_delete(disk_event->requests);
		disk_event->requests = NULL;
	}
}

int starpu_disk_test_request(struct _starpu_async_channel *async_channel)
{
	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&async_channel->event);
	unsigned node = disk_event->memory_node;
	int devid = starpu_memory_node_get_devid(node);

	if (disk_event->requests != NULL && !_starpu_disk_backend_event_list_empty(disk_event->requests))
	{
		struct _starpu_disk_backend_event * event = _starpu_disk_backend_event_list_begin(disk_event->requests);
		struct _starpu_disk_backend_event * next;

		/* Wait all events in the list and remove them */
		while (event != _starpu_disk_backend_event_list_end(disk_event->requests))
		{
			next = _starpu_disk_backend_event_list_next(event);

			int res = disk_register_list[devid]->functions->test_request(event->backend_event);

				if (res)
				{
					disk_register_list[devid]->functions->free_request(event->backend_event);

					_starpu_disk_backend_event_list_erase(disk_event->requests, event);

					_starpu_disk_backend_event_delete(event);
				}

			event = next;
		}

		/* Remove the list because it doesn't contain any event */
		if (_starpu_disk_backend_event_list_empty(disk_event->requests))
		{
			_starpu_disk_backend_event_list_delete(disk_event->requests);
			disk_event->requests = NULL;
		}
	}

	return disk_event->requests == NULL;
}

void starpu_disk_free_request(struct _starpu_async_channel *async_channe STARPU_ATTRIBUTE_UNUSED)
{
/* It does not have any sense to use this function currently because requests are freed in test of wait functions */
	STARPU_ABORT();

/*	struct _starpu_disk_event *disk_event = _starpu_disk_get_event(&async_channel->event);
	int position = get_location_with_node(disk_event->memory_node);
	if (disk_event->backend_event)
		disk_register_list[position]->functions->free_request(disk_event->backend_event);
*/
}

static int add_disk_in_list(int devid,  struct starpu_disk_ops *func, void *base)
{
	struct disk_register *dr;
	_STARPU_MALLOC(dr, sizeof(struct disk_register));
	dr->base = base;
	dr->flag = STARPU_DISK_ALL;
	dr->functions = func;
	disk_register_list[devid] = dr;
	return devid;
}

int _starpu_disk_can_copy(int devid1, int devid2)
{
	if (disk_register_list[devid1]->functions == disk_register_list[devid2]->functions)
		/* they must have a copy function */
		if (disk_register_list[devid1]->functions->copy != NULL)
			return 1;
	return 0;
}

void _starpu_set_disk_flag(int devid, int flag)
{
	disk_register_list[devid]->flag = flag;
}

int _starpu_get_disk_flag(int devid)
{
	return disk_register_list[devid]->flag;
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
		ops = &starpu_disk_unistd_ops;
	}
	else if (!strcmp(backend, "stdio"))
	{
		ops = &starpu_disk_stdio_ops;
	}
	else if (!strcmp(backend, "unistd"))
	{
		ops = &starpu_disk_unistd_ops;
	}
	else if (!strcmp(backend, "unistd_o_direct"))
	{
#ifdef STARPU_LINUX_SYS
		ops = &starpu_disk_unistd_o_direct_ops;
#else
		_STARPU_DISP("Warning: o_direct support is not compiled in, could not enable disk swap\n");
		return;
#endif

	}
	else if (!strcmp(backend, "leveldb"))
	{
#ifdef STARPU_HAVE_LEVELDB
		ops = &starpu_disk_leveldb_ops;
#else
		_STARPU_DISP("Warning: leveldb support is not compiled in, could not enable disk swap\n");
		return;
#endif
	}
	else if (!strcmp(backend, "hdf5"))
	{
#ifdef STARPU_HAVE_HDF5
		ops = &starpu_disk_hdf5_ops;
#else
		_STARPU_DISP("Warning: hdf5 support is not compiled in, could not enable disk swap\n");
		return;
#endif
	}
	else
	{
		_STARPU_DISP("Warning: unknown disk swap backend %s, could not enable disk swap\n", backend);
		return;
	}

	size = starpu_getenv_number_default("STARPU_DISK_SWAP_SIZE", -1);

	starpu_disk_swap_node = starpu_disk_register(ops, path, ((size_t) size) << 20);
	if (starpu_disk_swap_node < 0)
	{
		_STARPU_DISP("Warning: could not enable disk swap %s on %s with size %ld, could not enable disk swap\n", backend, path, (long) size);
		return;
	}
}
