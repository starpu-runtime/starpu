/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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


#include <mpi.h>
#include <errno.h>

#include <starpu.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_common.h>

#include <datawizard/memory_nodes.h>

#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/source_common.h>

struct _starpu_mp_node *_starpu_mpi_ms_src_get_actual_thread_mp_node()
{
	struct _starpu_worker *actual_worker = _starpu_get_local_worker_key();
	STARPU_ASSERT(actual_worker);

	int devid = actual_worker->devid;
	STARPU_ASSERT(devid >= 0 && devid < STARPU_MAXMPIDEVS);

	return _starpu_src_nodes[STARPU_MPI_MS_WORKER][devid];
}

void _starpu_mpi_source_init(struct _starpu_mp_node *node)
{
        _starpu_mpi_common_mp_initialize_src_sink(node);
        //TODO
}

void _starpu_mpi_source_deinit(struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED)
{

}

struct _starpu_mp_node *_starpu_src_common_get_mp_node_from_memory_node(int memory_node)
{
        int devid = starpu_memory_node_get_devid(memory_node);
	enum starpu_worker_archtype archtype = starpu_memory_node_get_worker_archtype(memory_node);
        STARPU_ASSERT_MSG(devid >= 0 && devid < STARPU_MAXMPIDEVS, "bogus devid %d for memory node %d\n", devid, memory_node);

        return _starpu_src_nodes[archtype][devid];
}

int _starpu_mpi_src_allocate_memory(void ** addr, size_t size, unsigned memory_node)
{
        struct _starpu_mp_node *mp_node = _starpu_src_common_get_mp_node_from_memory_node(memory_node);
        return _starpu_src_common_allocate(mp_node, addr, size);
}

void _starpu_mpi_source_free_memory(void *addr, unsigned memory_node)
{
        struct _starpu_mp_node *mp_node = _starpu_src_common_get_mp_node_from_memory_node(memory_node);
        _starpu_src_common_free(mp_node, addr);
}

/* Transfer SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
static int _starpu_mpi_copy_ram_to_mpi_sync(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size)
{
        struct _starpu_mp_node *mp_node = _starpu_src_common_get_mp_node_from_memory_node(dst_node);
        return _starpu_src_common_copy_host_to_sink_sync(mp_node, src, dst, size);
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
static int _starpu_mpi_copy_mpi_to_ram_sync(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size)
{
        struct _starpu_mp_node *mp_node = _starpu_src_common_get_mp_node_from_memory_node(src_node);
        return _starpu_src_common_copy_sink_to_host_sync(mp_node, src, dst, size);
}

static int _starpu_mpi_copy_sink_to_sink_sync(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size)
{
        return _starpu_src_common_copy_sink_to_sink_sync(_starpu_src_common_get_mp_node_from_memory_node(src_node),
							 _starpu_src_common_get_mp_node_from_memory_node(dst_node),
							 src, dst, size);
}

static int _starpu_mpi_copy_mpi_to_ram_async(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size, void * event)
{
        struct _starpu_mp_node *mp_node = _starpu_src_common_get_mp_node_from_memory_node(src_node);
        return _starpu_src_common_copy_sink_to_host_async(mp_node, src, dst, size, event);
}

static int _starpu_mpi_copy_ram_to_mpi_async(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size, void * event)
{
        struct _starpu_mp_node *mp_node = _starpu_src_common_get_mp_node_from_memory_node(dst_node);
        return _starpu_src_common_copy_host_to_sink_async(mp_node, src, dst, size, event);
}

static int _starpu_mpi_copy_sink_to_sink_async(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size, void * event)
{
        return _starpu_src_common_copy_sink_to_sink_async(_starpu_src_common_get_mp_node_from_memory_node(src_node),
							  _starpu_src_common_get_mp_node_from_memory_node(dst_node),
							  src, dst, size, event);
}

unsigned _starpu_mpi_src_get_device_count()
{
        int nb_mpi_devices;

        if (!_starpu_mpi_common_is_mp_initialized())
                return 0;

        MPI_Comm_size(MPI_COMM_WORLD, &nb_mpi_devices);

        //Remove one for master
        nb_mpi_devices = nb_mpi_devices - 1;

        return nb_mpi_devices;
}

void *_starpu_mpi_src_worker(void *arg)
{
#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
        struct _starpu_worker_set *worker_set_mpi = (struct _starpu_worker_set *) arg;
        int nbsinknodes = _starpu_mpi_src_get_device_count();

        int workersetnum;
        for (workersetnum = 0; workersetnum < nbsinknodes; workersetnum++)
        {
                struct _starpu_worker_set * worker_set = &worker_set_mpi[workersetnum];
#else
                struct _starpu_worker_set *worker_set = arg;
#endif

                /* As all workers of a set share common data, we just use the first
		 * one for intializing the following stuffs. */
                struct _starpu_worker *baseworker = &worker_set->workers[0];
                struct _starpu_machine_config *config = baseworker->config;
                unsigned baseworkerid = baseworker - config->workers;
                unsigned devid = baseworker->devid;
                unsigned i;

                /* unsigned memnode = baseworker->memory_node; */

                _starpu_driver_start(baseworker, STARPU_CPU_WORKER, 0);

#ifdef STARPU_USE_FXT
                for (i = 1; i < worker_set->nworkers; i++)
                        _starpu_worker_start(&worker_set->workers[i], STARPU_MPI_MS_WORKER, 0);
#endif

                // Current task for a thread managing a worker set has no sense.
                _starpu_set_current_task(NULL);

                for (i = 0; i < config->topology.nworker[STARPU_MPI_MS_WORKER][devid]; i++)
                {
                        struct _starpu_worker *worker = &config->workers[baseworkerid+i];
                        snprintf(worker->name, sizeof(worker->name), "MPI_MS %u core %u", devid, i);
                        snprintf(worker->short_name, sizeof(worker->short_name), "MPI_MS %u.%u", devid, i);
                }

#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
                {
                        char thread_name[16];
                        snprintf(thread_name, sizeof(thread_name), "MPI_MS");
                        starpu_pthread_setname(thread_name);
                }
#else
                {
                        char thread_name[16];
                        snprintf(thread_name, sizeof(thread_name), "MPI_MS %u", devid);
                        starpu_pthread_setname(thread_name);
                }
#endif

                for (i = 0; i < worker_set->nworkers; i++)
                {
                        struct _starpu_worker *worker = &worker_set->workers[i];
                        _STARPU_TRACE_WORKER_INIT_END(worker->workerid);
                }

#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
                _starpu_src_common_init_switch_env(workersetnum);
        }  /* for */

        /* set the worker zero for the main thread */
        for (workersetnum = 0; workersetnum < nbsinknodes; workersetnum++)
        {
                struct _starpu_worker_set * worker_set = &worker_set_mpi[workersetnum];
                struct _starpu_worker *baseworker = &worker_set->workers[0];
#endif

                /* tell the main thread that this one is ready */
                STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
                baseworker->status = STATUS_UNKNOWN;
                worker_set->set_is_initialized = 1;
                STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
                STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
        }
#endif

#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
        _starpu_src_common_workers_set(worker_set_mpi, nbsinknodes, _starpu_src_nodes[STARPU_MPI_MS_WORKER]);
#else
        _starpu_src_common_worker(worker_set, baseworkerid, _starpu_src_nodes[STARPU_MPI_MS_WORKER][devid]);
#endif

        return NULL;
}

int _starpu_mpi_copy_interface_from_mpi_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_MPI_MS_RAM && dst_kind == STARPU_CPU_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mpi_ms_copy_disabled() || !(copy_methods->mpi_ms_to_ram_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->mpi_ms_to_ram || copy_methods->any_to_any);
		if (copy_methods->mpi_ms_to_ram)
			copy_methods->mpi_ms_to_ram(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_mpi_node_ops;
		if(copy_methods->mpi_ms_to_ram_async)
			ret = copy_methods->mpi_ms_to_ram_async(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
	}
	return ret;
}

int _starpu_mpi_copy_interface_from_mpi_to_mpi(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_MPI_MS_RAM && dst_kind == STARPU_MPI_MS_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mpi_ms_copy_disabled() || !(copy_methods->mpi_ms_to_mpi_ms_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->mpi_ms_to_mpi_ms || copy_methods->any_to_any);
		if (copy_methods->mpi_ms_to_mpi_ms)
			copy_methods->mpi_ms_to_mpi_ms(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_mpi_node_ops;
		if(copy_methods->mpi_ms_to_mpi_ms_async)
			ret = copy_methods->mpi_ms_to_mpi_ms_async(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
	}
	return ret;
}

int _starpu_mpi_copy_interface_from_cpu_to_mpi(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_MPI_MS_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_mpi_ms_copy_disabled() || !(copy_methods->ram_to_mpi_ms_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->ram_to_mpi_ms || copy_methods->any_to_any);
		if (copy_methods->ram_to_mpi_ms)
			copy_methods->ram_to_mpi_ms(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_mpi_node_ops;
		if(copy_methods->ram_to_mpi_ms_async)
			ret = copy_methods->ram_to_mpi_ms_async(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
	}
	return ret;
}

int _starpu_mpi_copy_data_from_mpi_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_MPI_MS_RAM && dst_kind == STARPU_CPU_RAM);

	if (async_channel)
		return _starpu_mpi_copy_mpi_to_ram_async((void*) (src + src_offset), src_node,
							 (void*) (dst + dst_offset), dst_node,
							 size, async_channel);
	else
		return _starpu_mpi_copy_mpi_to_ram_sync((void*) (src + src_offset), src_node,
							(void*) (dst + dst_offset), dst_node,
							size);
}

int _starpu_mpi_copy_data_from_mpi_to_mpi(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_MPI_MS_RAM && dst_kind == STARPU_MPI_MS_RAM);

	if (async_channel)
		return _starpu_mpi_copy_sink_to_sink_async((void*) (src + src_offset), src_node,
							   (void*) (dst + dst_offset), dst_node,
							   size, async_channel);
	else
		return _starpu_mpi_copy_sink_to_sink_sync((void*) (src + src_offset), src_node,
							  (void*) (dst + dst_offset), dst_node,
							  size);
}

int _starpu_mpi_copy_data_from_cpu_to_mpi(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_MPI_MS_RAM);

	if (async_channel)
		return _starpu_mpi_copy_ram_to_mpi_async((void*) (src + src_offset), src_node,
							 (void*) (dst + dst_offset), dst_node,
							 size, async_channel);
	else
		return _starpu_mpi_copy_ram_to_mpi_sync((void*) (src + src_offset), src_node,
							(void*) (dst + dst_offset), dst_node,
							size);
}

int _starpu_mpi_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	(void) node;
	enum starpu_node_kind kind = starpu_node_get_kind(handling_node);
	return (kind == STARPU_MPI_MS_RAM);
}

uintptr_t _starpu_mpi_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	(void) flags;
	uintptr_t addr = 0;
	if (_starpu_mpi_src_allocate_memory((void **)(&addr), size, dst_node))
		addr = 0;
	return addr;
}

void _starpu_mpi_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	(void) flags;
	(void) size;
	_starpu_mpi_source_free_memory((void*) addr, dst_node);
}

struct _starpu_node_ops _starpu_driver_mpi_node_ops =
{
	.copy_interface_to[STARPU_CPU_RAM] = _starpu_mpi_copy_interface_from_mpi_to_cpu,
	.copy_interface_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_interface_from_mpi_to_mpi,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_mpi_copy_data_from_mpi_to_cpu,
	.copy_data_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_data_from_mpi_to_mpi,

	/* TODO: copy2D/3D? */

	.wait_request_completion = _starpu_mpi_common_wait_request_completion,
	.test_request_completion = _starpu_mpi_common_test_event,
	.is_direct_access_supported = _starpu_mpi_is_direct_access_supported,
	.malloc_on_node = _starpu_mpi_malloc_on_node,
	.free_on_node = _starpu_mpi_free_on_node,
	.name = "mpi driver"
};
