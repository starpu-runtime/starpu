/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <errno.h>

#include <starpu.h>
#include <drivers/tcpip/driver_tcpip_source.h>
#include <drivers/tcpip/driver_tcpip_common.h>

#include <datawizard/memory_nodes.h>

#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/source_common.h>

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
static unsigned tcpip_bindid_init[STARPU_MAXTCPIPDEVS] = { };
static unsigned tcpip_bindid[STARPU_MAXTCPIPDEVS];
static unsigned tcpip_memory_init[STARPU_MAXTCPIPDEVS] = { };
static unsigned tcpip_memory_nodes[STARPU_MAXTCPIPDEVS];

static struct _starpu_worker_set tcpip_worker_set[STARPU_MAXTCPIPDEVS];
#endif

struct _starpu_mp_node *_starpu_tcpip_ms_src_get_actual_thread_mp_node()
{
	struct _starpu_worker *actual_worker = _starpu_get_local_worker_key();
	STARPU_ASSERT(actual_worker);

	int devid = actual_worker->devid;
	STARPU_ASSERT(devid >= 0 && devid < STARPU_MAXTCPIPDEVS);

	return _starpu_src_nodes[STARPU_TCPIP_MS_WORKER][devid];
}

static void __starpu_init_tcpip_config(struct _starpu_machine_topology * topology,
				    struct _starpu_machine_config *config,
				    unsigned tcpip_idx)
{
	int nbcores;
	_starpu_src_common_sink_nbcores(_starpu_src_nodes[STARPU_TCPIP_MS_WORKER][tcpip_idx], &nbcores);
	STARPU_ASSERT(tcpip_idx < STARPU_NMAXDEVS);
	topology->nhwworker[STARPU_TCPIP_MS_WORKER][tcpip_idx] = nbcores;

	int ntcpipcores;
	ntcpipcores = starpu_getenv_number("STARPU_NTCPIPMSTHREADS");

	_starpu_topology_check_ndevices(&ntcpipcores, nbcores, 0, INT_MAX, 0, "STARPU_NTCPIPMSTHREADS", "TCPIP cores", "");

	tcpip_worker_set[tcpip_idx].workers = &config->workers[topology->nworkers];
	tcpip_worker_set[tcpip_idx].nworkers = ntcpipcores;
	_starpu_src_nodes[STARPU_TCPIP_MS_WORKER][tcpip_idx]->baseworkerid = topology->nworkers;

	_starpu_topology_configure_workers(topology, config,
					 STARPU_TCPIP_MS_WORKER,
					 tcpip_idx, tcpip_idx, 0, 0,
					ntcpipcores, 1, &tcpip_worker_set[tcpip_idx],
					_starpu_tcpip_common_multiple_thread  ? NULL : tcpip_worker_set);
}

/* Determine which devices we will use */
void _starpu_init_tcpip_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config,
				   struct starpu_conf *user_conf, int no_mp_config)
{
	int i;

	/* Discover and configure the mp topology. That means:
	 * - discover the number of mp nodes;
	 * - initialize each discovered node;
	 * - discover the local topology (number of PUs/devices) of each node;
	 * - configure the workers accordingly.
	 */

	for (i = 0; i < (int) (sizeof(tcpip_worker_set)/sizeof(tcpip_worker_set[0])); i++)
		tcpip_worker_set[i].workers = NULL;

	int ntcpipms = user_conf->ntcpip_ms;

	if(ntcpipms != 0)
	{
		/* Discover and initialize the number of TCPIP nodes through the mp
		 * infrastructure. */
		unsigned nhwtcpipdevices = _starpu_tcpip_src_get_device_count();

		if (ntcpipms == -1)
			/* Nothing was specified, so let's use the number of
			 * detected tcpip devices. ! */
			ntcpipms = nhwtcpipdevices;
		else
		{
			if ((unsigned) ntcpipms > nhwtcpipdevices)
			{
				/* The user requires more TCPIP devices than there is available */
				_STARPU_MSG("# Warning: %d TCPIP Master-Slave devices requested. Only %u available.\n",
					    ntcpipms, nhwtcpipdevices);
				ntcpipms = nhwtcpipdevices;
			}
			/*Let's make sure this value is OK.*/
			if(ntcpipms > STARPU_MAXTCPIPDEVS)
			{
				_STARPU_DISP("# Warning: %d TCPIP Master-Slave devices requested. Only %u enabled. Use configure options --enable-maxtcpipdev=xxx to update the maximum value of supported TCPIP MS devices.\n",
					    ntcpipms, STARPU_MAXTCPIPDEVS);
				ntcpipms = STARPU_MAXTCPIPDEVS;
			}
		}
	}

	topology->ndevices[STARPU_TCPIP_MS_WORKER] = ntcpipms;

	/* if user don't want to use TCPIP slaves, we close the slave processes */
	if (no_mp_config && topology->ndevices[STARPU_TCPIP_MS_WORKER] == 0)
	{
		_starpu_tcpip_common_mp_deinit();
		exit(0);
	}

	if (!no_mp_config)
	{
		for (i = 0; i < ntcpipms; i++)
			_starpu_src_nodes[STARPU_TCPIP_MS_WORKER][i] = _starpu_mp_common_node_create(STARPU_NODE_TCPIP_SOURCE, i);

		for (i = 0; i < ntcpipms; i++)
			__starpu_init_tcpip_config(topology, config, i);
	}
}

/*Bind the driver on a CPU core*/
void _starpu_tcpip_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;

	if (tcpip_bindid_init[devid])
	{
	}
	else
	{
		tcpip_bindid_init[devid] = 1;
		if (_starpu_tcpip_common_multiple_thread || devid == 0)
			tcpip_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
		else
			tcpip_bindid[devid] = tcpip_bindid[0];
	}

	workerarg->bindid = tcpip_bindid[devid];
}

/*Set up memory and buses*/
void _starpu_tcpip_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	unsigned devid = workerarg->devid;
	unsigned numa;

	if (tcpip_memory_init[devid])
	{
		memory_node = tcpip_memory_nodes[devid];
	}
	else
	{
		tcpip_memory_init[devid] = 1;
		memory_node = tcpip_memory_nodes[devid] = _starpu_memory_node_register(STARPU_TCPIP_MS_RAM, devid);

		_starpu_memory_node_set_mapped(memory_node);

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_register_bus(numa, memory_node);
			_starpu_register_bus(memory_node, numa);
		}

	}
	//This worker can manage transfers on NUMA nodes
	for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
			_starpu_worker_drives_memory_node(&workerarg->set->workers[0], numa);

	_starpu_worker_drives_memory_node(&workerarg->set->workers[0], memory_node);

	if (!_starpu_tcpip_common_multiple_thread)
	{
		/* TCP/IP driver thread can manage all slave memories if we disable the TCP/IP multiple thread */
		int findworker;
		for (findworker = 0; findworker < workerarg->workerid; findworker++)
		{
			struct _starpu_worker *findworkerarg = &config->workers[findworker];
			if (findworkerarg->arch == STARPU_TCPIP_MS_WORKER)
			{
				_starpu_worker_drives_memory_node(workerarg, findworkerarg->memory_node);
				_starpu_worker_drives_memory_node(findworkerarg, memory_node);
			}
		}
	}

	_starpu_memory_node_add_nworkers(memory_node);

	workerarg->memory_node = memory_node;
}

static void _starpu_deinit_tcpip_node(int devid)
{
	_starpu_mp_common_send_command(_starpu_src_nodes[STARPU_TCPIP_MS_WORKER][devid], STARPU_MP_COMMAND_EXIT, NULL, 0);

	_starpu_mp_common_node_destroy(_starpu_src_nodes[STARPU_TCPIP_MS_WORKER][devid]);
}


void _starpu_deinit_tcpip_config(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned i;

	for (i = 0; i < topology->ndevices[STARPU_TCPIP_MS_WORKER]; i++)
		_starpu_deinit_tcpip_node(i);
}


void _starpu_tcpip_source_init(struct _starpu_mp_node *node)
{
	_starpu_tcpip_common_mp_initialize_src_sink(node);
	//TODO
}


void _starpu_tcpip_source_deinit(struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED)
{

}

unsigned _starpu_tcpip_src_get_device_count()
{
	int nmpims = starpu_getenv_number("STARPU_TCPIP_MS_SLAVES");
	if (nmpims == -1)
		/* No slave */
		nmpims = 0;
	return nmpims;
}

void *_starpu_tcpip_src_worker(void *arg)
{
	struct _starpu_worker *worker0 = arg;
	struct _starpu_worker_set *set = worker0->set;
	struct _starpu_worker_set *worker_set_tcpip = set;
	int nbsinknodes = _starpu_tcpip_common_multiple_thread ? 1 : _starpu_tcpip_src_get_device_count();

	int workersetnum;
	for (workersetnum = 0; workersetnum < nbsinknodes; workersetnum++)
	{
		struct _starpu_worker_set * worker_set = &worker_set_tcpip[workersetnum];

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
			_starpu_worker_start(&worker_set->workers[i], STARPU_TCPIP_MS_WORKER, 0);
#endif

		// Current task for a thread managing a worker set has no sense.
		_starpu_set_current_task(NULL);

		for (i = 0; i < config->topology.nworker[STARPU_TCPIP_MS_WORKER][devid]; i++)
		{
			struct _starpu_worker *worker = &config->workers[baseworkerid+i];
			snprintf(worker->name, sizeof(worker->name), "TCPIP_MS %u core %u", devid, i);
			snprintf(worker->short_name, sizeof(worker->short_name), "TCPIP_MS %u.%u", devid, i);
		}


		char thread_name[16];
		if (_starpu_tcpip_common_multiple_thread)
			snprintf(thread_name, sizeof(thread_name), "TCPIP_MS %u", devid);
		else
			snprintf(thread_name, sizeof(thread_name), "TCPIP_MS");
		starpu_pthread_setname(thread_name);

		for (i = 0; i < worker_set->nworkers; i++)
		{
			struct _starpu_worker *worker = &worker_set->workers[i];
			_STARPU_TRACE_WORKER_INIT_END(worker->workerid);
		}

		_starpu_src_common_init_switch_env(workersetnum);
	}  /* for */

	_starpu_src_common_workers_set(worker_set_tcpip, nbsinknodes, &_starpu_src_nodes[STARPU_TCPIP_MS_WORKER][worker_set_tcpip->workers[0].devid]);

	return NULL;
}

int _starpu_tcpip_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	(void) node;
	enum starpu_node_kind kind = starpu_node_get_kind(handling_node);
	return (kind == STARPU_TCPIP_MS_RAM);
}

uintptr_t _starpu_tcpip_map(uintptr_t src, size_t src_offset, unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node, size_t size, int *ret)
{
	if(!_starpu_tcpip_mp_has_local())
	{
		*ret=-EXDEV;
		return 0;
	}

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

int _starpu_tcpip_unmap(uintptr_t src STARPU_ATTRIBUTE_UNUSED, size_t src_offset STARPU_ATTRIBUTE_UNUSED, unsigned src_node STARPU_ATTRIBUTE_UNUSED, uintptr_t dst, unsigned dst_node, size_t size)
{
	_starpu_src_common_unmap(dst_node, dst, size);

	return 0;
}

int _starpu_tcpip_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size)
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
struct _starpu_node_ops _starpu_driver_tcpip_ms_node_ops =
{
	.name = "tcpip driver",

	.malloc_on_node = _starpu_src_common_allocate,
	.free_on_node = _starpu_src_common_free,

	.is_direct_access_supported = _starpu_tcpip_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_copy_interface_any_to_any,
	.copy_interface_to[STARPU_TCPIP_MS_RAM] = _starpu_copy_interface_any_to_any,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_copy_interface_any_to_any,
	.copy_interface_from[STARPU_TCPIP_MS_RAM] = _starpu_copy_interface_any_to_any,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_src_common_copy_data_sink_to_host,
	.copy_data_to[STARPU_TCPIP_MS_RAM] = _starpu_src_common_copy_data_sink_to_sink,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_src_common_copy_data_host_to_sink,
	.copy_data_from[STARPU_TCPIP_MS_RAM] = _starpu_src_common_copy_data_sink_to_sink,

	.wait_request_completion = _starpu_tcpip_common_wait_request_completion,
	.test_request_completion = _starpu_tcpip_common_test_event,

	.map[STARPU_CPU_RAM] = _starpu_tcpip_map,
	.unmap[STARPU_CPU_RAM] = _starpu_tcpip_unmap,
	.update_map[STARPU_CPU_RAM] = _starpu_tcpip_update_map,
};
