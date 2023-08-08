/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_USE_MPI_MASTER_SLAVE
static unsigned mpi_bindid_init[STARPU_MAXMPIDEVS] = { };
static unsigned mpi_bindid[STARPU_MAXMPIDEVS];
static unsigned mpi_memory_init[STARPU_MAXMPIDEVS] = { };
static unsigned mpi_memory_nodes[STARPU_MAXMPIDEVS];

static struct _starpu_worker_set mpi_worker_set[STARPU_MAXMPIDEVS];
#endif

struct _starpu_mp_node *_starpu_mpi_ms_src_get_actual_thread_mp_node()
{
	struct _starpu_worker *actual_worker = _starpu_get_local_worker_key();
	STARPU_ASSERT(actual_worker);

	int devid = actual_worker->devid;
	STARPU_ASSERT(devid >= 0 && devid < STARPU_MAXMPIDEVS);

	return _starpu_src_nodes[STARPU_MPI_MS_WORKER][devid];
}

/* Configure one MPI slaves for run */
static void __starpu_init_mpi_config(struct _starpu_machine_topology *topology,
				     struct _starpu_machine_config *config,
				     unsigned mpi_idx)
{
	int nhwcores;
	_starpu_src_common_sink_nbcores(_starpu_src_nodes[STARPU_MPI_MS_WORKER][mpi_idx], &nhwcores);
	STARPU_ASSERT(mpi_idx < STARPU_NMAXDEVS);
	topology->nhwworker[STARPU_MPI_MS_WORKER][mpi_idx] = nhwcores;

	int nmpicores;
	nmpicores = starpu_getenv_number("STARPU_NMPIMSTHREADS");

	_starpu_topology_check_ndevices(&nmpicores, nhwcores, 0, INT_MAX, 0, "STARPU_NMPIMSTHREADS", "MPI cores", "");

	mpi_worker_set[mpi_idx].workers = &config->workers[topology->nworkers];
	mpi_worker_set[mpi_idx].nworkers = nmpicores;
	_starpu_src_nodes[STARPU_MPI_MS_WORKER][mpi_idx]->baseworkerid = topology->nworkers;

	_starpu_topology_configure_workers(topology, config,
			STARPU_MPI_MS_WORKER,
			mpi_idx, mpi_idx, 0, 0,
			nmpicores, 1, &mpi_worker_set[mpi_idx],
			_starpu_mpi_common_multiple_thread  ? NULL : mpi_worker_set);
}

/* Determine which devices we will use */
void _starpu_init_mpi_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config,
			     struct starpu_conf *user_conf, int no_mp_config)
{
	int i;

	/* Discover and configure the mp topology. That means:
	 * - discover the number of mp nodes;
	 * - initialize each discovered node;
	 * - discover the local topology (number of PUs/devices) of each node;
	 * - configure the workers accordingly.
	 */

	for (i = 0; i < (int) (sizeof(mpi_worker_set)/sizeof(mpi_worker_set[0])); i++)
		mpi_worker_set[i].workers = NULL;

	int nmpims = user_conf->nmpi_ms;

	if (nmpims != 0)
	{
		/* Discover and initialize the number of MPI nodes through the mp
		 * infrastructure. */
		unsigned nhwmpidevices = _starpu_mpi_src_get_device_count();

		if (nmpims == -1)
			/* Nothing was specified, so let's use the number of
			 * detected mpi devices. ! */
			nmpims = nhwmpidevices;
		else
		{
			if ((unsigned) nmpims > nhwmpidevices)
			{
				/* The user requires more MPI devices than there is available */
				_STARPU_MSG("# Warning: %d MPI Master-Slave devices requested. Only %u available.\n",
					    nmpims, nhwmpidevices);
				nmpims = nhwmpidevices;
			}
			/* Let's make sure this value is OK. */
			if (nmpims > STARPU_MAXMPIDEVS)
			{
				_STARPU_DISP("Warning: %d MPI MS devices requested. Only %d enabled. Use configure option --enable-maxmpidev=xxx to update the maximum value of supported MPI MS devices.\n", nmpims, STARPU_MAXMPIDEVS);
				nmpims = STARPU_MAXMPIDEVS;
			}
		}
	}

	topology->ndevices[STARPU_MPI_MS_WORKER] = nmpims;

	/* if user don't want to use MPI slaves, we close the slave processes */
	if (no_mp_config && topology->ndevices[STARPU_MPI_MS_WORKER] == 0)
	{
		_starpu_mpi_common_mp_deinit();
		exit(0);
	}

	if (!no_mp_config)
	{
		for (i = 0; i < nmpims; i++)
			_starpu_src_nodes[STARPU_MPI_MS_WORKER][i] = _starpu_mp_common_node_create(STARPU_NODE_MPI_SOURCE, i);

		for (i = 0; i < nmpims; i++)
			__starpu_init_mpi_config(topology, config, i);
	}
}

/* Bind the driver on a CPU core */
void _starpu_mpi_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;

	if (mpi_bindid_init[devid])
	{
	}
	else
	{
		mpi_bindid_init[devid] = 1;
		if (_starpu_mpi_common_multiple_thread || devid == 0)
			mpi_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
		else
			mpi_bindid[devid] = mpi_bindid[0];
	}
}

/* Set up memory and buses */
void _starpu_mpi_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	unsigned devid = workerarg->devid;
	unsigned numa, devid2;

	if (mpi_memory_init[devid])
	{
		memory_node = mpi_memory_nodes[devid];
	}
	else
	{
		mpi_memory_init[devid] = 1;
		memory_node = mpi_memory_nodes[devid] = _starpu_memory_node_register(STARPU_MPI_MS_RAM, devid);

		_starpu_memory_node_set_mapped(memory_node);

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_register_bus(numa, memory_node);
			_starpu_register_bus(memory_node, numa);
		}
		for (devid2 = 0; devid2 < STARPU_MAXMPIDEVS; devid2++)
			if (mpi_memory_init[devid2]) {
				_starpu_register_bus(mpi_memory_nodes[devid], mpi_memory_nodes[devid2]);
				_starpu_register_bus(mpi_memory_nodes[devid2], mpi_memory_nodes[devid]);
			}

	}
	//This worker can manage transfers on NUMA nodes
	for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
			_starpu_worker_drives_memory_node(&workerarg->set->workers[0], numa);

	_starpu_worker_drives_memory_node(&workerarg->set->workers[0], memory_node);

	if (!_starpu_mpi_common_multiple_thread)
	{
		/* MPI driver thread can manage all slave memories if we disable the MPI multiple thread */
		int findworker;
		for (findworker = 0; findworker < workerarg->workerid; findworker++)
		{
			struct _starpu_worker *findworkerarg = &config->workers[findworker];
			if (findworkerarg->arch == STARPU_MPI_MS_WORKER)
			{
				_starpu_worker_drives_memory_node(workerarg, findworkerarg->memory_node);
				_starpu_worker_drives_memory_node(findworkerarg, memory_node);
			}
		}
	}

	workerarg->bindid = mpi_bindid[devid];
	_starpu_memory_node_add_nworkers(memory_node);

	workerarg->memory_node = memory_node;
}

static void _starpu_deinit_mpi_node(int devid)
{
	_starpu_mp_common_send_command(_starpu_src_nodes[STARPU_MPI_MS_WORKER][devid], STARPU_MP_COMMAND_EXIT, NULL, 0);

	_starpu_mp_common_node_destroy(_starpu_src_nodes[STARPU_MPI_MS_WORKER][devid]);
}


void _starpu_deinit_mpi_config(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned i;

	for (i = 0; i < topology->ndevices[STARPU_MPI_MS_WORKER]; i++)
		_starpu_deinit_mpi_node(i);
}


void _starpu_mpi_source_init(struct _starpu_mp_node *node)
{
	_starpu_mpi_common_mp_initialize_src_sink(node);
	//TODO
}


void _starpu_mpi_source_deinit(struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED)
{

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
	struct _starpu_worker *worker0 = arg;
	struct _starpu_worker_set *set = worker0->set;
	struct _starpu_worker_set *worker_set_mpi = set;
	int nbsinknodes = _starpu_mpi_common_multiple_thread ? 1 : _starpu_mpi_src_get_device_count();

	int workersetnum;
	for (workersetnum = 0; workersetnum < nbsinknodes; workersetnum++)
	{
		struct _starpu_worker_set * worker_set = &worker_set_mpi[workersetnum];

		/* As all workers of a set share common data, we just use the first
		 * one for initializing the following stuffs. */
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

		{
			char thread_name[16];
			if (_starpu_mpi_common_multiple_thread)
				snprintf(thread_name, sizeof(thread_name), "MPI_MS %u", devid);
			else
				snprintf(thread_name, sizeof(thread_name), "MPI_MS");
			starpu_pthread_setname(thread_name);
		}

		for (i = 0; i < worker_set->nworkers; i++)
		{
			struct _starpu_worker *worker = &worker_set->workers[i];
			_STARPU_TRACE_WORKER_INIT_END(worker->workerid);
		}

		_starpu_src_common_init_switch_env(workersetnum);
	}  /* for */

	_starpu_src_common_workers_set(worker_set_mpi, nbsinknodes, &_starpu_src_nodes[STARPU_MPI_MS_WORKER][worker_set_mpi->workers[0].devid]);

	return NULL;
}

int _starpu_mpi_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	(void) node;
	enum starpu_node_kind kind = starpu_node_get_kind(handling_node);
	return (kind == STARPU_MPI_MS_RAM);
}

uintptr_t _starpu_mpi_map(uintptr_t src, size_t src_offset, unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node, size_t size, int *ret)
{
	uintptr_t map_addr = _starpu_src_common_map(dst_node, src+src_offset, size);
	if(map_addr == 0)
	{
		*ret=-ENOMEM;
	}
	else
	{
		*ret = 0;
	}
	return map_addr;
}

int _starpu_mpi_unmap(uintptr_t src STARPU_ATTRIBUTE_UNUSED, size_t src_offset STARPU_ATTRIBUTE_UNUSED, unsigned src_node STARPU_ATTRIBUTE_UNUSED, uintptr_t dst, unsigned dst_node, size_t size)
{
	_starpu_src_common_unmap(dst_node, dst, size);

	return 0;
}

int _starpu_mpi_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	(void) src;
	(void) src_offset;
	(void) src_node;
	(void) dst;
	(void) dst_offset;
	(void) dst_node;
	(void) size;

	/* Memory mappings are cache-coherent */
	return 0;
}

struct _starpu_node_ops _starpu_driver_mpi_ms_node_ops =
{
	.name = "mpi driver",

	.malloc_on_node = _starpu_src_common_allocate,
	.free_on_node = _starpu_src_common_free,

	.is_direct_access_supported = _starpu_mpi_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_copy_interface_any_to_any,
	.copy_interface_to[STARPU_MPI_MS_RAM] = _starpu_copy_interface_any_to_any,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_copy_interface_any_to_any,
	.copy_interface_from[STARPU_MPI_MS_RAM] = _starpu_copy_interface_any_to_any,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_src_common_copy_data_sink_to_host,
	.copy_data_to[STARPU_MPI_MS_RAM] = _starpu_src_common_copy_data_sink_to_sink,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_src_common_copy_data_host_to_sink,
	.copy_data_from[STARPU_MPI_MS_RAM] = _starpu_src_common_copy_data_sink_to_sink,

	/* TODO: copy2D/3D? */

	.wait_request_completion = _starpu_mpi_common_wait_request_completion,
	.test_request_completion = _starpu_mpi_common_test_event,

	.map[STARPU_CPU_RAM] = _starpu_mpi_map,
	.unmap[STARPU_CPU_RAM] = _starpu_mpi_unmap,
	.update_map[STARPU_CPU_RAM] = _starpu_mpi_update_map,
};
