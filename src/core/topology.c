/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
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

#include <stdlib.h>
#include <stdio.h>
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <core/workers.h>
#include <core/debug.h>
#include <core/devices.h>
#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/hip/driver_hip.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/max/driver_max_fpga.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_common.h>
#include <drivers/tcpip/driver_tcpip_source.h>
#include <drivers/tcpip/driver_tcpip_common.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/opencl/driver_opencl_utils.h>
#include <profiling/profiling.h>
#include <datawizard/datastats.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/memory_manager.h>

#include <common/uthash.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif

#endif

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

static unsigned topology_is_initialized = 0;
static int nobind;
static int numa_enabled = -1;

/* For checking whether two workers share the same PU, indexed by PU number */
static int cpu_worker[STARPU_MAXCPUS];
static char * cpu_name[STARPU_MAXCPUS];
static unsigned nb_numa_nodes = 0;
static int numa_memory_nodes_to_hwloclogid[STARPU_MAXNUMANODES]; /* indexed by StarPU numa node to convert in hwloc logid */
static int numa_memory_nodes_to_physicalid[STARPU_MAXNUMANODES]; /* indexed by StarPU numa node to convert in physical id */
static unsigned numa_bus_id[STARPU_MAXNUMANODES*STARPU_MAXNUMANODES];

#define STARPU_NUMA_UNINITIALIZED (-2)
#define STARPU_NUMA_MAIN_RAM (-1)

unsigned _starpu_may_bind_automatically[STARPU_NARCH] = { 0 };

unsigned starpu_memory_nodes_get_numa_count(void)
{
	return nb_numa_nodes;
}

#if defined(STARPU_HAVE_HWLOC)
static hwloc_obj_t numa_get_obj(hwloc_obj_t obj)
{
#if HWLOC_API_VERSION >= 0x00020000
	while (obj && obj->memory_first_child == NULL)
		obj = obj->parent;

	if (!obj)
		return NULL;

	return obj->memory_first_child;
#else
	while (obj && obj->type != HWLOC_OBJ_NUMANODE)
		obj = obj->parent;

	/* Note: If we don't find a "node" obj before the root, this means
	 * hwloc does not know whether there are numa nodes or not, so
	 * we should not use a per-node sampling in that case. */
	return obj;
#endif
}
static int numa_get_logical_id(hwloc_obj_t obj)
{
	STARPU_ASSERT(obj);
	obj = numa_get_obj(obj);
	if (!obj)
		return 0;
	return obj->logical_index;
}

static int numa_get_physical_id(hwloc_obj_t obj)
{
	STARPU_ASSERT(obj);
	obj = numa_get_obj(obj);
	if (!obj)
		return 0;
	return obj->os_index;
}
#endif

int _starpu_get_logical_numa_node_worker(unsigned workerid)
{
#if defined(STARPU_HAVE_HWLOC)
	STARPU_ASSERT(numa_enabled != -1);
	if (numa_enabled)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config() ;
		struct _starpu_machine_topology *topology = &config->topology ;

		hwloc_obj_t obj;
		switch(worker->arch)
		{
			case STARPU_CPU_WORKER:
				obj = hwloc_get_obj_by_type(topology->hwtopology, HWLOC_OBJ_PU, worker->bindid) ;
				break;
			default:
				STARPU_ABORT();
		}

		return numa_get_logical_id(obj);
	}
	else
#endif
	{
		(void) workerid; /* unused */
		return STARPU_NUMA_MAIN_RAM;
	}
}

/* This returns the exact NUMA node next to a worker */
static int _starpu_get_physical_numa_node_worker(unsigned workerid)
{
#if defined(STARPU_HAVE_HWLOC)
	STARPU_ASSERT(numa_enabled != -1);
	if (numa_enabled)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config() ;
		struct _starpu_machine_topology *topology = &config->topology ;

		hwloc_obj_t obj;
		switch(worker->arch)
		{
			case STARPU_CPU_WORKER:
				obj = hwloc_get_obj_by_type(topology->hwtopology, HWLOC_OBJ_PU, worker->bindid) ;
				break;
			default:
				STARPU_ABORT();
		}

		return numa_get_physical_id(obj);
	}
	else
#endif
	{
		(void) workerid; /* unused */
		return STARPU_NUMA_MAIN_RAM;
	}
}

/* This returns the CPU NUMA memory close to a worker */
static int _starpu_get_logical_close_numa_node_worker(unsigned workerid)
{
#if defined(STARPU_HAVE_HWLOC)
	STARPU_ASSERT(numa_enabled != -1);
	if (numa_enabled)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config() ;
		struct _starpu_machine_topology *topology = &config->topology ;

		hwloc_obj_t obj = NULL;
		if (starpu_driver_info[worker->arch].get_hwloc_obj)
			obj = starpu_driver_info[worker->arch].get_hwloc_obj(topology, worker->devid);
		if (!obj)
			obj = hwloc_get_obj_by_type(topology->hwtopology, HWLOC_OBJ_PU, worker->bindid) ;

		return numa_get_logical_id(obj);
	}
	else
#endif
	{
		(void) workerid; /* unused */
		return STARPU_NUMA_MAIN_RAM;
	}
}

//TODO change this in an array
int starpu_memory_nodes_numa_hwloclogid_to_id(int logid)
{
	unsigned n;
	for (n = 0; n < nb_numa_nodes; n++)
		if (numa_memory_nodes_to_hwloclogid[n] == logid)
			return n;
	return -1;
}

int starpu_memory_nodes_numa_id_to_hwloclogid(unsigned id)
{
	STARPU_ASSERT(id < STARPU_MAXNUMANODES);
	return numa_memory_nodes_to_hwloclogid[id];
}

int starpu_memory_nodes_numa_devid_to_id(unsigned id)
{
	STARPU_ASSERT(id < STARPU_MAXNUMANODES);
	return numa_memory_nodes_to_physicalid[id];
}

//TODO change this in an array
int starpu_memory_nodes_numa_id_to_devid(int osid)
{
	unsigned n;
	for (n = 0; n < nb_numa_nodes; n++)
		if (numa_memory_nodes_to_physicalid[n] == osid)
			return n;
	return -1;
}

// TODO: cache the values instead of looking in hwloc each time

/* Avoid using this one, prefer _starpu_task_data_get_node_on_worker */
int _starpu_task_data_get_node_on_node(struct starpu_task *task, unsigned index, unsigned local_node)
{
	int node = STARPU_SPECIFIC_NODE_LOCAL;
	if (task->cl->specific_nodes)
		node = STARPU_CODELET_GET_NODE(task->cl, index);
	switch (node)
	{
	case STARPU_SPECIFIC_NODE_LOCAL:
		// TODO: rather find MCDRAM
		node = local_node;
		break;
	case STARPU_SPECIFIC_NODE_CPU:
		switch (starpu_node_get_kind(local_node))
		{
		case STARPU_CPU_RAM:
			node = local_node;
			break;
		default:
			// TODO: rather take close NUMA node
			node = STARPU_MAIN_RAM;
			break;
		}
		break;
	case STARPU_SPECIFIC_NODE_SLOW:
		// TODO: rather leave in DDR
		node = local_node;
		break;
	case STARPU_SPECIFIC_NODE_LOCAL_OR_CPU:
		{
			enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, index);
			if (mode & STARPU_R)
			{
				if (mode & STARPU_R && task->handles[index]->per_node[local_node].state != STARPU_INVALID)
				{
					/* It is here already, rather access it from here */
					node = local_node;
				}
				else
				{
					/* It is not here already, do not bother moving it */
					node = STARPU_MAIN_RAM;
				}
			}
			else
			{
				/* Nothing to read, consider where to write */
				starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, index);
				if (handle->wt_mask & (1 << STARPU_MAIN_RAM))
					/* Write through, better simply write to the main memory */
					node = STARPU_MAIN_RAM;
				else
					/* Better keep temporary data on the accelerator to save PCI bandwidth */
					node = local_node;
			}
			break;
		}
	case STARPU_SPECIFIC_NODE_NONE:
		return -1;
	}
	return node;
}

int _starpu_task_data_get_node_on_worker(struct starpu_task *task, unsigned index, unsigned worker)
{
	unsigned local_node = starpu_worker_get_memory_node(worker);
	int node = STARPU_SPECIFIC_NODE_LOCAL;
	if (task->cl->specific_nodes)
		node = STARPU_CODELET_GET_NODE(task->cl, index);
	switch (node)
	{
	case STARPU_SPECIFIC_NODE_LOCAL:
		// TODO: rather find MCDRAM
		node = local_node;
		break;
	case STARPU_SPECIFIC_NODE_CPU:
		node = starpu_memory_nodes_numa_hwloclogid_to_id(_starpu_get_logical_close_numa_node_worker(worker));
		if (node == -1)
			node = STARPU_MAIN_RAM;
		break;
	case STARPU_SPECIFIC_NODE_SLOW:
		// TODO: rather leave in DDR
		node = local_node;
		break;
	case STARPU_SPECIFIC_NODE_LOCAL_OR_CPU:
		{
			enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, index);
			if (mode & STARPU_R)
			{
				if (task->handles[index]->per_node[local_node].state != STARPU_INVALID)
				{
					/* It is here already, rather access it from here */
					node = local_node;
				}
				else
				{
					/* It is not here already, do not bother moving it */
					node = STARPU_MAIN_RAM;
				}
			}
			else
			{
				/* Nothing to read, consider where to write */
				starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, index);
				if (handle->wt_mask & (1 << STARPU_MAIN_RAM))
					/* Write through, better simply write to the main memory */
					node = STARPU_MAIN_RAM;
				else
					/* Better keep temporary data on the accelerator to save PCI bandwidth */
					node = local_node;
			}
			break;
		}
	case STARPU_SPECIFIC_NODE_NONE:
		return -1;
	}
	return node;
}

struct _starpu_worker *_starpu_get_worker_from_driver(struct starpu_driver *d)
{
	unsigned nworkers = starpu_worker_get_count();
	unsigned workerid;

	for (workerid = 0; workerid < nworkers; workerid++)
	{
		if (starpu_worker_get_type(workerid) == d->type)
		{
			struct _starpu_worker *worker;
			worker = _starpu_get_worker_struct(workerid);
			STARPU_ASSERT(worker->driver_ops);
			STARPU_ASSERT_MSG(worker->driver_ops->is_devid, "The driver operation 'is_devid' is not defined");
			if (worker->driver_ops->is_devid(d, worker))
				return worker;
		}
	}

	return NULL;
}

void _starpu_initialize_workers_deviceid(int *explicit_workers_gpuid,
					 int *current, int *workers_gpuid,
					 const char *varname, unsigned nhwgpus,
					 enum starpu_worker_archtype type)
{
	char *strval;
	unsigned i;

	*current = 0;

	/* conf->workers_gpuid indicates the successive GPU identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the varname env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	strval = starpu_getenv(varname);
	if (strval)
	{
		/* varname certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round
		 * robin fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1
		 * 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the varname
		 * env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap)
			{
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					workers_gpuid[i] = (unsigned)val;
					strval = endptr;
				}
				else
				{
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;

					/* there is no more values in the
					 * string */
					wrap = 1;

					workers_gpuid[i] = workers_gpuid[0];
				}
			}
			else
			{
				workers_gpuid[i] =
					workers_gpuid[i % number_of_entries];
			}
		}
	}
	else if (explicit_workers_gpuid)
	{
		/* we use the explicit value from the user */
		memcpy(workers_gpuid,
		       explicit_workers_gpuid,
		       STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else
	{
		/* by default, we take a round robin policy */
		if (nhwgpus > 0)
		     for (i = 0; i < STARPU_NMAXWORKERS; i++)
			  workers_gpuid[i] = (unsigned)(i % nhwgpus);

		/* StarPU can use sampling techniques to bind threads
		 * correctly */
		_starpu_may_bind_automatically[type] = 1;
	}
}

int _starpu_get_next_devid(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config, enum starpu_worker_archtype arch)
{
	if (topology->nworkers == STARPU_NMAXWORKERS)
		// Already full!
		return -1;

	unsigned i = ((config->current_devid[arch]++) % config->topology.ndevices[arch]);

	return (int)config->topology.workers_devid[arch][i];
}

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
static void _starpu_allocate_topology_userdata(hwloc_obj_t obj)
{
	unsigned i;

	_STARPU_CALLOC(obj->userdata,  1, sizeof(struct _starpu_hwloc_userdata));
	for (i = 0; i < obj->arity; i++)
		_starpu_allocate_topology_userdata(obj->children[i]);
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t child;
	for (child = obj->io_first_child; child; child = child->next_sibling)
		_starpu_allocate_topology_userdata(child);
#endif
}

static void _starpu_deallocate_topology_userdata(hwloc_obj_t obj)
{
	unsigned i;
	struct _starpu_hwloc_userdata *data = obj->userdata;

	STARPU_ASSERT(!data->worker_list || data->worker_list == (void*)-1);
	free(data);
	for (i = 0; i < obj->arity; i++)
		_starpu_deallocate_topology_userdata(obj->children[i]);
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t child;
	for (child = obj->io_first_child; child; child = child->next_sibling)
		_starpu_deallocate_topology_userdata(child);
#endif
}
#endif
#endif

static void _starpu_init_topology(struct _starpu_machine_config *config)
{
	/* Discover the topology, meaning finding all the available PUs for
	   the compiled drivers. These drivers MUST have been initialized
	   before calling this function. The discovered topology is filled in
	   CONFIG. */
	struct _starpu_machine_topology *topology = &config->topology;

	if (topology_is_initialized)
		return;

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif

#if defined(STARPU_USE_HIP)
	if (config->conf.nhip != 0)
		_starpu_init_hip();
#endif

#if defined(STARPU_USE_MAX_FPGA)
	if (config->conf.nmax_fpga != 0)
		_starpu_init_max_fpga();
#endif

	nobind = starpu_getenv_number("STARPU_WORKERS_NOBIND");

	topology->nhwdevices[STARPU_CPU_WORKER] = 1;
	topology->nhwworker[STARPU_CPU_WORKER][0] = 0;
	topology->nhwpus = 0;
	topology->nusedpus = 0;
	topology->firstusedpu = 0;

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
	int err;
	err = hwloc_topology_init(&topology->hwtopology);
	STARPU_ASSERT_MSG(err == 0, "Could not initialize Hwloc topology (%s)\n", strerror(errno));
	char *hwloc_input = starpu_getenv("STARPU_HWLOC_INPUT");
	if (hwloc_input && hwloc_input[0])
	{
		err = hwloc_topology_set_xml(topology->hwtopology, hwloc_input);
		if (err < 0) _STARPU_DISP("Could not load hwloc input %s\n", hwloc_input);
	}

	_starpu_topology_filter(topology->hwtopology);
	err = hwloc_topology_load(topology->hwtopology);
	STARPU_ASSERT_MSG(err == 0, "Could not load Hwloc topology (%s)%s%s%s\n", strerror(errno), hwloc_input ? " (input " : "", hwloc_input ? hwloc_input : "", hwloc_input ? ")" : "");

#ifdef HAVE_HWLOC_CPUKINDS_GET_NR
	int nr_kinds = hwloc_cpukinds_get_nr(topology->hwtopology, 0);
	if (nr_kinds > 1)
		_STARPU_DISP("Warning: there are several kinds of CPU on this system. For now StarPU assumes all CPU are equal\n");
#endif

	_starpu_allocate_topology_userdata(hwloc_get_root_obj(topology->hwtopology));
#endif
#endif

#ifdef STARPU_SIMGRID
	config->topology.nhwworker[STARPU_CPU_WORKER][0] = config->topology.nhwpus = _starpu_simgrid_get_nbhosts("CPU");
#elif defined(STARPU_HAVE_HWLOC)
	/* Discover the CPUs relying on the hwloc interface and fills CONFIG
	 * accordingly. */

	config->cpu_depth = hwloc_get_type_depth(topology->hwtopology, HWLOC_OBJ_CORE);
	config->pu_depth = hwloc_get_type_depth(topology->hwtopology, HWLOC_OBJ_PU);

	/* Would be very odd */
	STARPU_ASSERT(config->cpu_depth != HWLOC_TYPE_DEPTH_MULTIPLE);

	if (config->cpu_depth == HWLOC_TYPE_DEPTH_UNKNOWN)
	{
		/* unknown, using logical procesors as fallback */
		_STARPU_DISP("Warning: The OS did not report CPU cores. Assuming there is only one hardware thread per core.\n");
		config->cpu_depth = hwloc_get_type_depth(topology->hwtopology,
							 HWLOC_OBJ_PU);
	}

	topology->nhwworker[STARPU_CPU_WORKER][0] = hwloc_get_nbobjs_by_depth(topology->hwtopology, config->cpu_depth);
	topology->nhwpus =
	topology->nusedpus = hwloc_get_nbobjs_by_depth(topology->hwtopology, config->pu_depth);

	if (starpu_getenv_number_default("STARPU_WORKERS_GETBIND", 1))
	{
		/* Respect the existing binding */

		hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_t log_cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_t log_coreset = hwloc_bitmap_alloc();
		unsigned n, i, first, last, weight;
		int ret;

		do {
			/* Get the process binding (e.g. provided by the job scheduler) */
			ret = hwloc_get_cpubind(topology->hwtopology, cpuset, HWLOC_CPUBIND_THREAD);
			if (ret)
			{
				_STARPU_DISP("Warning: could not get current CPU binding: %s\n", strerror(errno));
				break;
			}

			/* Compute logical sets */
			n = hwloc_get_nbobjs_by_depth(topology->hwtopology, config->pu_depth);
			for (i = 0; i < n; i++)
			{
				hwloc_obj_t pu = hwloc_get_obj_by_depth(topology->hwtopology, config->pu_depth, i), core;

				if (!hwloc_bitmap_isset(cpuset, pu->os_index))
					continue;

				hwloc_bitmap_set(log_cpuset, i);

				core = pu;
				if (config->cpu_depth != config->pu_depth)
				{
					while (core && core->type != HWLOC_OBJ_CORE)
						core = core->parent;
					if (!core)
					{
						_STARPU_DISP("Warning: hwloc did not report a core above PU %d\n", i);
						break;
					}
				}
				hwloc_bitmap_set(log_coreset, core->logical_index);
			}

			/* Check that PU numbers are consecutive */
			first = hwloc_bitmap_first(log_cpuset);
			last = hwloc_bitmap_last(log_cpuset);
			weight = hwloc_bitmap_weight(log_cpuset);
			if (last - first + 1 != weight)
			{
				_STARPU_DISP("Warning: hwloc reported non-consecutive binding, this is not supported yet, sorry, please use STARPU_WORKERS_CPUID or STARPU_WORKERS_COREID to set this by hand\n");
				break;
			}

			topology->nusedpus = weight;
			topology->firstusedpu = first;
		} while(0);

		hwloc_bitmap_free(cpuset);
		hwloc_bitmap_free(log_cpuset);
		hwloc_bitmap_free(log_coreset);
	}

#elif defined(HAVE_SYSCONF)
	/* Discover the CPUs relying on the sysconf(3) function and fills
	 * CONFIG accordingly. */

	config->topology.nhwworker[STARPU_CPU_WORKER][0] =
	config->topology.nhwpus =
	config->topology.nusedpus =
		sysconf(_SC_NPROCESSORS_ONLN);

#elif defined(_WIN32)
	/* Discover the CPUs on Cygwin and MinGW systems. */

	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	config->topology.nhwworker[STARPU_CPU_WORKER][0] =
	config->topology.nhwpus =
	config->topology.nusedpus =
		sysinfo.dwNumberOfProcessors;
#else
#warning no way to know number of cores, assuming 1
	config->topology.nhwworker[STARPU_CPU_WORKER][0] =
	config->topology.nhwpus =
	config->topology.nusedpus =
		1;
#endif

	if (config->conf.ncuda != 0)
		_starpu_cuda_discover_devices(config);
	if (config->conf.nhip != 0)
		_starpu_hip_discover_devices(config);
	if (config->conf.nopencl != 0)
		_starpu_opencl_discover_devices(config);
	if (config->conf.nmax_fpga != 0)
		_starpu_max_fpga_discover_devices(config);
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	config->topology.nhwdevices[STARPU_MPI_MS_WORKER] = _starpu_mpi_src_get_device_count();
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	config->topology.nhwdevices[STARPU_TCPIP_MS_WORKER] = _starpu_tcpip_src_get_device_count();
#endif

	topology_is_initialized = 1;
}

/*
 * Bind workers on the different processors
 */
static void _starpu_initialize_workers_bindid(struct _starpu_machine_config *config)
{
	char *strval;
	unsigned i;

	struct _starpu_machine_topology *topology = &config->topology;
	STARPU_ASSERT_MSG(topology->nhwworker[STARPU_CPU_WORKER][0], "Unexpected value for topology->nhwworker[STARPU_CPU_WORKER][0] %u", topology->nhwworker[STARPU_CPU_WORKER][0]);
	int nhyperthreads = topology->nhwpus / topology->nhwworker[STARPU_CPU_WORKER][0];
	int scale = 1;

	config->current_bindid = 0;

	if (starpu_getenv("STARPU_WORKERS_CPUID") && starpu_getenv("STARPU_WORKERS_COREID"))
	{
		_STARPU_DISP("Warning: STARPU_WORKERS_CPUID and STARPU_WORKERS_COREID cannot be set at the same time. STARPU_WORKERS_CPUID will be used.\n");
	}

	/* conf->workers_bindid indicates the successive logical PU identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	strval = starpu_getenv("STARPU_WORKERS_CPUID");
	if (strval == NULL)
	{
		strval = starpu_getenv("STARPU_WORKERS_COREID");
		if (strval)
			scale = nhyperthreads;
	}

	if (strval)
	{
		/* STARPU_WORKERS_CPUID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round
		 * robin fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1
		 * 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the STARPU_WORKERS_CPUID
		 * env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap)
			{
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					topology->workers_bindid[i] = (unsigned)((val * scale) % topology->nusedpus) + topology->firstusedpu;
					strval = endptr;
					if (*strval == '-')
					{
						/* range of values */
						long int endval;
						strval++;
						if (*strval && *strval != ' ' && *strval != ',')
						{
							endval = strtol(strval, &endptr, 10);
							strval = endptr;
						}
						else
						{
							endval = topology->nusedpus / scale - 1;
							if (*strval)
								strval++;
						}
						for (val++; val <= endval && i < STARPU_NMAXWORKERS-1; val++)
						{
							i++;
							topology->workers_bindid[i] = (unsigned)((val * scale) % topology->nusedpus) + topology->firstusedpu;
						}
					}
					if (*strval == ',')
						strval++;
				}
				else
				{
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;

					/* there is no more values in the
					 * string */
					wrap = 1;

					topology->workers_bindid[i] =
						topology->workers_bindid[0];
				}
			}
			else
			{
				topology->workers_bindid[i] =
					topology->workers_bindid[i % number_of_entries];
			}
		}
		topology->workers_nbindid = number_of_entries;
	}
	else if (config->conf.use_explicit_workers_bindid)
	{
		/* we use the explicit value from the user */
		memcpy(topology->workers_bindid,
			config->conf.workers_bindid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
		topology->workers_nbindid = STARPU_NMAXWORKERS;
	}
	else
	{
		int nth_per_core = starpu_getenv_number_default("STARPU_NTHREADS_PER_CORE", 1);
		int k;
		int nbindids=0;
		STARPU_ASSERT_MSG(nth_per_core > 0 && nth_per_core <= nhyperthreads , "Incorrect number of hyperthreads");

		i = 0; /* PU number currently assigned */
		k = 0; /* Number of threads already put on the current core */
		while(nbindids < STARPU_NMAXWORKERS)
		{
			if (k >= nth_per_core)
			{
				/* We have already put enough workers on this
				 * core, skip remaining PUs from this core, and
				 * proceed with next core */
				i += nhyperthreads-nth_per_core;
				k = 0;
				continue;
			}

			/* Add a worker to this core, by using this logical PU */
			topology->workers_bindid[nbindids++] = (unsigned)(i % topology->nusedpus) + topology->firstusedpu;
			k++;
			i++;
		}
		topology->workers_nbindid = nbindids;
	}

	for (i = 0; i < STARPU_MAXCPUS;i++)
		cpu_worker[i] = STARPU_NOWORKERID;

	/* no binding yet */
	memset(&config->currently_bound, 0, sizeof(config->currently_bound));
	memset(&config->currently_shared, 0, sizeof(config->currently_shared));
}

static void _starpu_deinitialize_workers_bindid(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i;

	for (i = 0; i < STARPU_MAXCPUS;i++)
	{
		if (cpu_name[i])
		{
			free(cpu_name[i]);
			cpu_name[i] = NULL;
		}
	}

}

unsigned _starpu_get_next_bindid(struct _starpu_machine_config *config, unsigned flags,
				 unsigned *preferred_binding, unsigned npreferred)
{
	struct _starpu_machine_topology *topology = &config->topology;

	STARPU_ASSERT_MSG(topology_is_initialized, "The StarPU core is not initialized yet, have you called starpu_init?");

	unsigned current_preferred;
	unsigned nhyperthreads = topology->nhwpus / topology->nhwworker[STARPU_CPU_WORKER][0];
	unsigned workers_nbindid = topology->workers_nbindid;
	unsigned i;

	if (npreferred)
	{
		STARPU_ASSERT_MSG(preferred_binding, "Passing NULL pointer for parameter preferred_binding with a non-0 value of parameter npreferred");
	}

	/* loop over the preference list */
	for (current_preferred = 0;
	     current_preferred < npreferred;
	     current_preferred++)
	{
		/* can we bind the worker on the preferred core ? */
		unsigned requested_core = preferred_binding[current_preferred];
		unsigned requested_bindid = requested_core * nhyperthreads;

		/* Look at the remaining PUs to be bound to */
		for (i = 0; i < workers_nbindid; i++)
		{
			if (topology->workers_bindid[i] == requested_bindid &&
					(!config->currently_bound[i] ||
					 (config->currently_shared[i] && !(flags & STARPU_THREAD_ACTIVE)))
					)
			{
				/* the PU is available, or shareable with us, we use it ! */
				config->currently_bound[i] = 1;
				if (!(flags & STARPU_THREAD_ACTIVE))
					config->currently_shared[i] = 1;
				return requested_bindid;
			}
		}
	}

	if (!(flags & STARPU_THREAD_ACTIVE))
	{
		/* Try to find a shareable PU */
		for (i = 0; i < workers_nbindid; i++)
			if (config->currently_shared[i])
				return topology->workers_bindid[i];
	}

	/* Try to find an available PU from last used PU */
	for (i = config->current_bindid; i < workers_nbindid; i++)
		if (!config->currently_bound[i])
			/* Found a cpu ready for use, use it! */
			break;

	if (i == workers_nbindid)
	{
		/* Finished binding on all cpus, restart from start in
		 * case the user really wants overloading */
		memset(&config->currently_bound, 0, sizeof(config->currently_bound));
		i = 0;
	}

	STARPU_ASSERT(i < workers_nbindid);
	unsigned bindid = topology->workers_bindid[i];
	config->currently_bound[i] = 1;
	if (!(flags & STARPU_THREAD_ACTIVE))
		config->currently_shared[i] = 1;
	config->current_bindid = i;
	return bindid;
}

unsigned starpu_get_next_bindid(unsigned flags, unsigned *preferred, unsigned npreferred)
{
	return _starpu_get_next_bindid(_starpu_get_machine_config(), flags, preferred, npreferred);
}

unsigned _starpu_topology_get_nhwcpu(struct _starpu_machine_config *config)
{
	_starpu_init_topology(config);

	return config->topology.nhwworker[STARPU_CPU_WORKER][0];
}

unsigned _starpu_topology_get_nhwpu(struct _starpu_machine_config *config)
{
	_starpu_init_topology(config);

	return config->topology.nhwpus;
}

unsigned _starpu_topology_get_nhwnumanodes(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED)
{
	_starpu_init_topology(config);

	int res;
#if defined(STARPU_HAVE_HWLOC)
	if (numa_enabled == -1)
		numa_enabled = starpu_getenv_number_default("STARPU_USE_NUMA", 0);
	if (numa_enabled)
	{
		struct _starpu_machine_topology *topology = &config->topology ;
		int nnumanodes = hwloc_get_nbobjs_by_type(topology->hwtopology, HWLOC_OBJ_NUMANODE) ;
		res = nnumanodes > 0 ? nnumanodes : 1 ;
	}
	else
#endif
	{
		res = 1;
	}

	STARPU_ASSERT_MSG(res <= STARPU_MAXNUMANODES, "Number of NUMA nodes discovered %d is higher than maximum accepted %d ! Use configure option --enable-maxnumanodes=xxx to increase the maximum value of supported NUMA nodes.\n", res, STARPU_MAXNUMANODES);
	return res;
}

#if defined(STARPU_HAVE_HWLOC)
static unsigned _starpu_topology_get_core_binding(unsigned *binding, unsigned nbinding, hwloc_obj_t obj)
{
	unsigned found = 0;
	unsigned n;

	if (obj->type == HWLOC_OBJ_CORE)
	{
		*binding = obj->logical_index;
		found++;
	}

	for (n = 0; n < obj->arity; n++)
	{
		found += _starpu_topology_get_core_binding(binding + found, nbinding - found, obj->children[n]);
	}
	return found;
}
#endif

unsigned _starpu_topology_get_numa_core_binding(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED, const unsigned *numa_binding STARPU_ATTRIBUTE_UNUSED, unsigned nnuma STARPU_ATTRIBUTE_UNUSED, unsigned *binding STARPU_ATTRIBUTE_UNUSED, unsigned nbinding STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_HAVE_HWLOC)
	unsigned n;
	unsigned cur = 0;

	for (n = 0; n < nnuma; n++)
	{
		hwloc_obj_t obj = hwloc_get_obj_by_type(config->topology.hwtopology, HWLOC_OBJ_NUMANODE, numa_binding[n]);

#if HWLOC_API_VERSION >= 0x00020000
		/* Get the actual topology object */
		obj = obj->parent;
#endif
		cur += _starpu_topology_get_core_binding(binding + cur, nbinding - cur, obj);
		if (cur == nbinding)
			break;
	}
	return cur;
#else
	return 0;
#endif
}

#ifdef STARPU_HAVE_HWLOC
void _starpu_topology_filter(hwloc_topology_t topology)
{
#if HWLOC_API_VERSION >= 0x20000
	hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM);
#else
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM | HWLOC_TOPOLOGY_FLAG_WHOLE_IO);
#endif
#ifdef HAVE_HWLOC_TOPOLOGY_SET_COMPONENTS
/* Driver porters: adding your driver here is optional, it is just to avoid hwloc components which take a lot of time to start.  */
#  ifndef STARPU_USE_CUDA
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "cuda");
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "nvml");
#  endif
#  ifndef STARPU_USE_HIP
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "hip");
	/* TODO: check about rocclr, the equivalent of nvml*/
	//hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "rocm_smi");
#  endif
#  ifndef STARPU_USE_OPENCL
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "opencl");
#  endif
#endif
}
#endif

void _starpu_topology_check_ndevices(int *ndevices, unsigned nhwdevices, int overflow, unsigned max, const char *nname, const char *dname, const char *configurename)
{
	if (!*ndevices)
		return;

	STARPU_ASSERT_MSG(*ndevices >= -1, "%s can not be negative and different from -1 (is is %d)", nname, *ndevices);

	if (*ndevices == -1)
	{
		/* Nothing was specified, so let's choose ! */
		STARPU_ASSERT_MSG(nhwdevices <= max, "Oops, driver reported more than its own maximum");
		*ndevices = nhwdevices;
	}
	else
	{
		if (!overflow && *ndevices > (int) nhwdevices)
		{
			/* The user requires more devices than there is available */
			_STARPU_DISP("Warning: %d %s devices requested. Only %d available.\n", *ndevices, dname, nhwdevices);
			*ndevices = nhwdevices;
		}
		/* Let's make sure this value is OK. */
		if (*ndevices > (int) max)
		{
			_STARPU_DISP("Warning: %d %s devices requested. Only %d enabled. Use configure option --enable-%s=xxx to update the maximum value of supported %s devices.\n", *ndevices, dname, max, configurename, dname);
			*ndevices = max;
		}
	}
}

void _starpu_topology_configure_workers(struct _starpu_machine_topology *topology,
					struct _starpu_machine_config *config,
					enum starpu_worker_archtype type,
					int devnum, int devid,
					int homogeneous, int worker_devid,
					unsigned nworker_per_device,
					unsigned ncores,
					struct _starpu_worker_set *worker_set,
					struct _starpu_worker_set *driver_worker_set)
{
	topology->nworker[type][devnum] = nworker_per_device;
	topology->devid[type][devnum] = devid;

	unsigned i;

	for (i = 0; i < nworker_per_device; i++)
	{
		if (topology->nworkers == STARPU_NMAXWORKERS)
			// We are full
			break;

		int worker_idx = topology->nworkers++;

		if (worker_set == ALLOC_WORKER_SET)
		{
			/* Just one worker in the set */
			_STARPU_CALLOC(config->workers[worker_idx].set, 1, sizeof(struct _starpu_worker_set));
			config->workers[worker_idx].set->workers = &config->workers[worker_idx];
			config->workers[worker_idx].set->nworkers = 1;
			if (type != STARPU_CPU_WORKER)
				_starpu_cpu_busy_cpu(1);
		}
		else
		{
			config->workers[worker_idx].set = worker_set;
			if ((!worker_set || worker_set->workers == &config->workers[worker_idx])
			 && (!driver_worker_set || driver_worker_set == worker_set)
			 && type != STARPU_CPU_WORKER)
				_starpu_cpu_busy_cpu(1);
		}

		config->workers[worker_idx].driver_worker_set = driver_worker_set;
		config->workers[worker_idx].arch = type;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = type;
		config->workers[worker_idx].perf_arch.devices[0].devid = homogeneous ? 0 : worker_devid ? (int) i : devid;
		config->workers[worker_idx].perf_arch.devices[0].ncores = ncores;
		config->workers[worker_idx].devid = worker_devid ? (int) i : devid;
		config->workers[worker_idx].devnum = worker_devid ? (int) i : devnum;
		config->workers[worker_idx].subworkerid = worker_devid ? 0 : i;
		config->workers[worker_idx].worker_mask = STARPU_WORKER_TO_MASK(type);
		config->worker_mask |= STARPU_WORKER_TO_MASK(type);
	}
}

#ifdef STARPU_HAVE_HWLOC
static unsigned _starpu_topology_count_ngpus(hwloc_obj_t obj)
{
	struct _starpu_hwloc_userdata *data = obj->userdata;
	unsigned n = data->ngpus;
	unsigned i;

	for (i = 0; i < obj->arity; i++)
		n += _starpu_topology_count_ngpus(obj->children[i]);

	data->ngpus = n;
//#ifdef STARPU_VERBOSE
//	{
//		char name[64];
//		hwloc_obj_type_snprintf(name, sizeof(name), obj, 0);
//		_STARPU_DEBUG("hwloc obj %s has %u GPUs below\n", name, n);
//	}
//#endif
	return n;
}
#endif

static int _starpu_init_machine_config(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED)
{
	int i;

	for (i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		config->workers[i].workerid = i;
		config->workers[i].set = NULL;
	}

	struct _starpu_machine_topology *topology = &config->topology;

	topology->nworkers = 0;
	topology->ncombinedworkers = 0;
	topology->nsched_ctxs = 0;

	_starpu_init_topology(config);

	_starpu_initialize_workers_bindid(config);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	_starpu_init_cuda_config(topology, config);
#endif

#if defined(STARPU_USE_HIP)
	_starpu_init_hip_config(topology, config);
#endif

/* We put the OpenCL section after the CUDA section: we rather use NVidia GPUs in CUDA mode than in OpenCL mode */
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	_starpu_init_opencl_config(topology, config);
#endif

#ifdef STARPU_USE_MAX_FPGA
	_starpu_init_max_fpga_config(topology, config);
#endif

#if defined(STARPU_USE_MPI_MASTER_SLAVE)
	_starpu_init_mpi_config(topology, config, &config->conf, no_mp_config);
#endif
#if defined(STARPU_USE_TCPIP_MASTER_SLAVE)
	_starpu_init_tcpip_config(topology, config, &config->conf, no_mp_config);
#endif

/* we put the CPU section after the accelerator : in case there was an
 * accelerator found, we devote one cpu */
#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
	_starpu_init_cpu_config(topology, config);
#endif

	if (topology->nworkers == 0)
	{
		_STARPU_DEBUG("No worker found, aborting ...\n");
		return -ENODEV;
	}
	return 0;
}

void _starpu_destroy_machine_config(struct _starpu_machine_config *config)
{
	_starpu_close_debug_logfile();

	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &config->workers[worker];
		int bindid = workerarg->bindid;
		free(workerarg->perf_arch.devices);
#ifdef STARPU_HAVE_HWLOC
		hwloc_bitmap_free(workerarg->hwloc_cpu_set);
		if (bindid != -1)
		{
			hwloc_obj_t worker_obj = hwloc_get_obj_by_depth(config->topology.hwtopology,
									config->pu_depth,
									bindid);
			struct _starpu_hwloc_userdata *data = worker_obj->userdata;
			if (data->worker_list)
			{
				_starpu_worker_list_delete(data->worker_list);
				data->worker_list = NULL;
			}
		}
#endif
		if (bindid != -1)
		{
			free(config->bindid_workers[bindid].workerids);
			config->bindid_workers[bindid].workerids = NULL;
		}
	}
	free(config->bindid_workers);
	config->bindid_workers = NULL;
	config->nbindid = 0;
	unsigned combined_worker_id;
	for(combined_worker_id=0 ; combined_worker_id < config->topology.ncombinedworkers ; combined_worker_id++)
	{
		struct _starpu_combined_worker *combined_worker = &config->combined_workers[combined_worker_id];
#ifdef STARPU_HAVE_HWLOC
		hwloc_bitmap_free(combined_worker->hwloc_cpu_set);
#endif
		free(combined_worker->perf_arch.devices);
	}

#ifdef STARPU_HAVE_HWLOC
	_starpu_deallocate_topology_userdata(hwloc_get_root_obj(config->topology.hwtopology));
	hwloc_topology_destroy(config->topology.hwtopology);
#endif

	topology_is_initialized = 0;

	_starpu_devices_gpu_clean();

	int i;
	for (i=0; i<STARPU_NARCH; i++)
		_starpu_may_bind_automatically[i] = 0;
}

int _starpu_bind_thread_on_cpu(int cpuid STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, const char *name STARPU_ATTRIBUTE_UNUSED)
{
	int ret = 0;
#ifdef STARPU_SIMGRID
	return ret;
#else
	if (nobind > 0)
		return ret;
	if (cpuid < 0)
		return ret;

#ifdef STARPU_HAVE_HWLOC
	const struct hwloc_topology_support *support;
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	_starpu_init_topology(config);

	if (workerid != STARPU_NOWORKERID && cpuid < STARPU_MAXCPUS)
	{
/* TODO: mutex... */
		int previous = cpu_worker[cpuid];
		/* We would like the PU to be available, or we are perhaps fine to share it */
		if (!(previous == STARPU_NOWORKERID ||
		      (previous == STARPU_NONACTIVETHREAD && workerid == STARPU_NONACTIVETHREAD) ||
		      (previous >= 0 && previous == workerid) ||
		      (name && cpu_name[cpuid] && !strcmp(name, cpu_name[cpuid]))))
		{
			char hostname[65];
			gethostname(hostname, sizeof(hostname));

			if (previous == STARPU_ACTIVETHREAD)
				_STARPU_DISP("[%s] Warning: active thread %s was already bound to PU %d\n", hostname, cpu_name[cpuid], cpuid);
			else if (previous == STARPU_NONACTIVETHREAD)
				_STARPU_DISP("[%s] Warning: non-active thread %s was already bound to PU %d\n", hostname, cpu_name[cpuid], cpuid);
			else
				_STARPU_DISP("[%s] Warning: worker %d was already bound to PU %d\n", hostname, previous, cpuid);

			if (workerid == STARPU_ACTIVETHREAD)
				_STARPU_DISP("and we were told to also bind active thread %s to it.\n", name);
			else if (previous == STARPU_NONACTIVETHREAD)
				_STARPU_DISP("and we were told to also bind non-active thread %s to it.\n", name);
			else
				_STARPU_DISP("and we were told to also bind worker %d to it.\n", workerid);

			_STARPU_DISP("This will strongly degrade performance.\n");

			if (workerid >= 0)
				/* This shouldn't happen for workers */
				_STARPU_DISP("[%s] Maybe check starpu_machine_display's output to determine what wrong binding happened. Hwloc reported a total of %d cores and %d threads, and to use %d threads from logical %d, perhaps there is misdetection between hwloc, the kernel and the BIOS, or an administrative allocation issue from e.g. the job scheduler?\n", hostname, config->topology.nhwworker[STARPU_CPU_WORKER][0], config->topology.nhwpus, config->topology.nusedpus, config->topology.firstusedpu);
			ret = -1;
		}
		else
		{
			cpu_worker[cpuid] = workerid;
			if (name)
			{
				if (cpu_name[cpuid])
					free(cpu_name[cpuid]);
				cpu_name[cpuid] = strdup(name);
			}
		}
	}

	support = hwloc_topology_get_support(config->topology.hwtopology);
	if (support->cpubind->set_thisthread_cpubind)
	{
		hwloc_obj_t obj = hwloc_get_obj_by_depth(config->topology.hwtopology, config->pu_depth, cpuid);
		hwloc_bitmap_t set = obj->cpuset;
		int res;

		hwloc_bitmap_singlify(set);
		res = hwloc_set_cpubind(config->topology.hwtopology, set, HWLOC_CPUBIND_THREAD);
		if (res)
		{
			perror("hwloc_set_cpubind");
			STARPU_ABORT();
		}
	}

#elif defined(HAVE_PTHREAD_SETAFFINITY_NP) && defined(__linux__)
	int res;
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(cpuid, &aff_mask);

	starpu_pthread_t self = starpu_pthread_self();

	res = pthread_setaffinity_np(self, sizeof(aff_mask), &aff_mask);
	if (res)
	{
		const char *msg = strerror(res);
		_STARPU_MSG("pthread_setaffinity_np: %s\n", msg);
		STARPU_ABORT();
	}

#elif defined(_WIN32)
	DWORD mask = 1 << cpuid;
	if (!SetThreadAffinityMask(GetCurrentThread(), mask))
	{
		_STARPU_ERROR("SetThreadMaskAffinity(%lx) failed\n", mask);
	}
#else
#warning no CPU binding support
#endif
#endif
	return ret;
}

int
starpu_bind_thread_on(int cpuid, unsigned flags, const char *name)
{
	int workerid;
	STARPU_ASSERT_MSG(name, "starpu_bind_thread_on must be provided with a name");
	starpu_pthread_setname(name);
	if (flags & STARPU_THREAD_ACTIVE)
		workerid = STARPU_ACTIVETHREAD;
	else
		workerid = STARPU_NONACTIVETHREAD;
	return _starpu_bind_thread_on_cpu(cpuid, workerid, name);
}

void _starpu_bind_thread_on_cpus(struct _starpu_combined_worker *combined_worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	return;
#endif
#ifdef STARPU_HAVE_HWLOC
	const struct hwloc_topology_support *support;
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	_starpu_init_topology(config);

	support = hwloc_topology_get_support(config->topology.hwtopology);
	if (support->cpubind->set_thisthread_cpubind)
	{
		hwloc_bitmap_t set = combined_worker->hwloc_cpu_set;
		int ret;

		ret = hwloc_set_cpubind(config->topology.hwtopology, set, HWLOC_CPUBIND_THREAD);
		if (ret)
		{
			perror("binding thread");
			STARPU_ABORT();
		}
	}
#else
#ifdef __GLIBC__
	sched_setaffinity(0,sizeof(combined_worker->cpu_set),&combined_worker->cpu_set);
#else
#  warning no parallel worker CPU binding support
#endif
#endif
}

static size_t _starpu_cpu_get_global_mem_size(int nodeid, struct _starpu_machine_config *config)
{
	size_t global_mem;
	starpu_ssize_t limit = -1;

#if defined(STARPU_HAVE_HWLOC)
	struct _starpu_machine_topology *topology = &config->topology;

	STARPU_ASSERT(numa_enabled != -1);
	if (numa_enabled)
	{
		int depth_node = hwloc_get_type_depth(topology->hwtopology, HWLOC_OBJ_NUMANODE);

		if (depth_node == HWLOC_TYPE_DEPTH_UNKNOWN)
		{
#if HWLOC_API_VERSION >= 0x00020000
			global_mem = hwloc_get_root_obj(topology->hwtopology)->total_memory;
#else
			global_mem = hwloc_get_root_obj(topology->hwtopology)->memory.total_memory;
#endif
		}
		else
		{
			char name[32];
			hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, depth_node, nodeid);
#if HWLOC_API_VERSION >= 0x00020000
			global_mem = obj->attr->numanode.local_memory;
#else
			global_mem = obj->memory.local_memory;
#endif
			snprintf(name, sizeof(name), "STARPU_LIMIT_CPU_NUMA_%d_MEM", obj->os_index);
			limit = starpu_getenv_number(name);
		}
	}
	else
	{
		/* Do not limit ourself to a single NUMA node */
#if HWLOC_API_VERSION >= 0x00020000
		global_mem = hwloc_get_root_obj(topology->hwtopology)->total_memory;
#else
		global_mem = hwloc_get_root_obj(topology->hwtopology)->memory.total_memory;
#endif
	}

#else /* STARPU_HAVE_HWLOC */
#ifdef STARPU_DEVEL
#  warning TODO: use sysinfo when available to get global size
#endif
	global_mem = 0;
#endif

	if (limit == -1)
		limit = starpu_getenv_number("STARPU_LIMIT_CPU_NUMA_MEM");

	if (limit == -1)
	{
		limit = starpu_getenv_number("STARPU_LIMIT_CPU_MEM");
		if (limit != -1 && numa_enabled)
		{
			_STARPU_DISP("NUMA is enabled and STARPU_LIMIT_CPU_MEM is set to %luMB. Assuming that it should be distributed over the %d NUMA node(s). You probably want to use STARPU_LIMIT_CPU_NUMA_MEM instead.\n", (long) limit, _starpu_topology_get_nhwnumanodes(config));
			limit /= _starpu_topology_get_nhwnumanodes(config);
		}
	}

	/* Don't eat all memory for ourself */
	global_mem *= 0.9;

	if (limit < 0)
		// No limit is defined, we return the global memory size
		return global_mem;
	else if (global_mem && (size_t)limit * 1024*1024 > global_mem)
	{
		if (numa_enabled)
			_STARPU_DISP("The requested limit %ldMB for NUMA node %d is higher that available memory %luMB, using the latter\n", (unsigned long) limit, nodeid, (unsigned long) global_mem / (1024*1024));
		else
			_STARPU_DISP("The requested limit %ldMB is higher that available memory %luMB, using the latter\n", (long) limit, (unsigned long) global_mem / (1024*1024));
		return global_mem;
	}
	else
		// We limit the memory
		return limit*1024*1024;
}

//TODO : Check SIMGRID
static void _starpu_init_numa_node(struct _starpu_machine_config *config)
{
	nb_numa_nodes = 0;

	unsigned i;
	for (i = 0; i < STARPU_MAXNUMANODES; i++)
	{
		numa_memory_nodes_to_hwloclogid[i] = STARPU_NUMA_UNINITIALIZED;
		numa_memory_nodes_to_physicalid[i] = STARPU_NUMA_UNINITIALIZED;
	}

#ifdef STARPU_SIMGRID
	char name[16];
	starpu_sg_host_t host;
#endif

	numa_enabled = starpu_getenv_number_default("STARPU_USE_NUMA", 0);
	/* NUMA mode activated */
	if (numa_enabled)
	{
		/* Take all NUMA nodes used by CPU workers */
		unsigned worker;
		for (worker = 0; worker < config->topology.nworkers; worker++)
		{
			struct _starpu_worker *workerarg = &config->workers[worker];
			if (workerarg->arch == STARPU_CPU_WORKER)
			{
				int numa_logical_id = _starpu_get_logical_numa_node_worker(worker);

				/* Convert logical id to StarPU id to check if this NUMA node is already saved or not */
				int numa_starpu_id = starpu_memory_nodes_numa_hwloclogid_to_id(numa_logical_id);

				/* This shouldn't happen */
				if (numa_starpu_id == -1 && nb_numa_nodes == STARPU_MAXNUMANODES)
				{
					_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nhwnumanodes(config), STARPU_MAXNUMANODES);
					STARPU_ABORT();
				}

				if (numa_starpu_id == -1)
				{
					int devid = numa_logical_id == STARPU_NUMA_MAIN_RAM ? 0 : numa_logical_id;
					int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, devid);
					_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(devid, config));
					STARPU_ASSERT_MSG(memnode < STARPU_MAXNUMANODES, "Wrong Memory Node : %d (only %d available)", memnode, STARPU_MAXNUMANODES);
					_starpu_memory_node_set_mapped(memnode);
					numa_memory_nodes_to_hwloclogid[memnode] = numa_logical_id;
					int numa_physical_id = _starpu_get_physical_numa_node_worker(worker);
					numa_memory_nodes_to_physicalid[memnode] = numa_physical_id;
					nb_numa_nodes++;
#ifdef STARPU_SIMGRID
					snprintf(name, sizeof(name), "RAM%d", memnode);
					host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memnode, host);
#endif
				}
			}
		}

		/* If we found NUMA nodes from CPU workers, it's good */
		if (nb_numa_nodes != 0)
			return;

		_STARPU_DISP("No NUMA nodes found when checking CPU workers...\n");

#ifdef STARPU_HAVE_HWLOC
		_STARPU_DISP("Take NUMA nodes attached to GPU devices...\n");

		for (i = 0; i < STARPU_NARCH; i++)
		{
			if (!starpu_driver_info[i].get_hwloc_obj)
				continue;

			unsigned j;

			for (j = 0; j < config->topology.ndevices[i]; j++)
			{
				hwloc_obj_t obj = starpu_driver_info[i].get_hwloc_obj(&config->topology,
						config->topology.devid[i][j]);

				if (obj)
					obj = numa_get_obj(obj);
				/* Hwloc cannot recognize some devices */
				if (!obj)
					continue;
				int numa_starpu_id = starpu_memory_nodes_numa_hwloclogid_to_id(obj->logical_index);

				/* This shouldn't happen */
				if (numa_starpu_id == -1 && nb_numa_nodes == STARPU_MAXNUMANODES)
				{
					_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nhwnumanodes(config), STARPU_MAXNUMANODES);
					STARPU_ABORT();
				}

				if (numa_starpu_id == -1)
				{
					int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, obj->logical_index);
					_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(obj->logical_index, config));
					STARPU_ASSERT_MSG(memnode < STARPU_MAXNUMANODES, "Wrong Memory Node : %d (only %d available)", memnode, STARPU_MAXNUMANODES);
					_starpu_memory_node_set_mapped(memnode);
					numa_memory_nodes_to_hwloclogid[memnode] = obj->logical_index;
					numa_memory_nodes_to_physicalid[memnode] = obj->os_index;
					nb_numa_nodes++;
#ifdef STARPU_SIMGRID
					snprintf(name, sizeof(name), "RAM%d", memnode);
					host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memnode, host);
#endif
				}
			}
		}
#endif
	}

#ifdef STARPU_HAVE_HWLOC
	//Found NUMA nodes from CUDA nodes
	if (nb_numa_nodes != 0)
		return;

	/* In case, we do not find any NUMA nodes when checking NUMA nodes attached to GPUs, we take all of them */
	if (numa_enabled)
		_STARPU_DISP("No NUMA nodes found when checking GPUs devices...\n");
#endif

	if (numa_enabled)
		_STARPU_DISP("Finally, take all NUMA nodes available... \n");

	unsigned nnuma = _starpu_topology_get_nhwnumanodes(config);
	if (nnuma > STARPU_MAXNUMANODES)
	{
		_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nhwnumanodes(config), STARPU_MAXNUMANODES);
		nnuma = STARPU_MAXNUMANODES;
	}

	unsigned numa;
	for (numa = 0; numa < nnuma; numa++)
	{
		unsigned numa_logical_id;
		unsigned numa_physical_id;
#if defined(STARPU_HAVE_HWLOC)
		hwloc_obj_t obj = hwloc_get_obj_by_type(config->topology.hwtopology, HWLOC_OBJ_NUMANODE, numa);
		if (obj)
		{
			numa_logical_id = obj->logical_index;
			numa_physical_id = obj->os_index;
		}
		else
#endif
		{
			numa_logical_id = 0;
			numa_physical_id = 0;
		}
		int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, numa_logical_id);
		STARPU_ASSERT(memnode < STARPU_MAXNUMANODES);
		_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(numa_logical_id, config));
		_starpu_memory_node_set_mapped(memnode);

		numa_memory_nodes_to_hwloclogid[memnode] = numa_logical_id;
		numa_memory_nodes_to_physicalid[memnode] = numa_physical_id;
		nb_numa_nodes++;

		if (numa == 0)
			STARPU_ASSERT_MSG(memnode == STARPU_MAIN_RAM, "Wrong Memory Node : %d (expected %d) \n", memnode, STARPU_MAIN_RAM);
		STARPU_ASSERT_MSG(memnode < STARPU_MAXNUMANODES, "Wrong Memory Node : %d (only %d available) \n", memnode, STARPU_MAXNUMANODES);

#ifdef STARPU_SIMGRID
		if (nnuma > 1)
		{
			snprintf(name, sizeof(name), "RAM%d", memnode);
			host = _starpu_simgrid_get_host_by_name(name);
		}
		else
		{
			/* In this case, nnuma has only one node */
			host = _starpu_simgrid_get_host_by_name("RAM");
		}

		STARPU_ASSERT(host);
		_starpu_simgrid_memory_node_set_host(memnode, host);
#endif
	}

	STARPU_ASSERT_MSG(nb_numa_nodes > 0, "No NUMA node found... We need at least one memory node !\n");
}

static void _starpu_init_numa_bus()
{
	unsigned i, j;
	for (i = 0; i < nb_numa_nodes; i++)
		for (j = 0; j < nb_numa_nodes; j++)
			if (i != j)
				numa_bus_id[i*nb_numa_nodes+j] = _starpu_register_bus(i, j);
}

#if defined(STARPU_HAVE_HWLOC) && !defined(STARPU_SIMGRID)
static int _starpu_find_pu_driving_numa_from(hwloc_obj_t root, unsigned node)
{
	unsigned i;
	int found = 0;

	if (!root->arity)
	{
		if (root->type == HWLOC_OBJ_PU)
		{
			struct _starpu_hwloc_userdata *userdata = root->userdata;
			if (userdata->pu_worker)
			{
				/* Cool, found a worker! */
				_STARPU_DEBUG("found PU %d to drive memory node %d\n", userdata->pu_worker->bindid, node);
				_starpu_worker_drives_memory_node(userdata->pu_worker, node);
				found = 1;
			}
		}
	}
	for (i = 0; i < root->arity; i++)
	{
		if (_starpu_find_pu_driving_numa_from(root->children[i], node))
			found = 1;
	}
	return found;
}

/* Look upward to find a level containing the given NUMA node and workers to drive it */
static int _starpu_find_pu_driving_numa_up(hwloc_obj_t root, unsigned node)
{
	if (_starpu_find_pu_driving_numa_from(root, node))
		/* Ok, we already managed to find drivers */
		return 1;
	if (!root->parent)
		/* And no parent!? nobody can drive this... */
		return 0;
	/* Try from parent */
	return _starpu_find_pu_driving_numa_up(root->parent, node);
}
#endif

static void _starpu_init_workers_binding_and_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED)
{
	/* We will store all the busid of the different (src, dst)
	 * combinations in a matrix which we initialize here. */
	_starpu_initialize_busid_matrix();

	unsigned bindid;

	for (bindid = 0; bindid < config->nbindid; bindid++)
	{
		free(config->bindid_workers[bindid].workerids);
		config->bindid_workers[bindid].workerids = NULL;
		config->bindid_workers[bindid].nworkers = 0;
	}

	/* First determine the CPU binding */
	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &config->workers[worker];
		unsigned devid STARPU_ATTRIBUTE_UNUSED = workerarg->devid;

		/* select the worker binding */
		starpu_driver_info[workerarg->arch].init_worker_binding(config, no_mp_config, workerarg);

		_STARPU_DEBUG("worker %u type %d devid %u bound to cpu %d\n", worker, workerarg->arch, devid, workerarg->bindid);

#ifdef __GLIBC__
		if (workerarg->bindid != -1)
		{
			/* Save the initial cpuset */
			CPU_ZERO(&workerarg->cpu_set);
			CPU_SET(workerarg->bindid, &workerarg->cpu_set);
		}
#endif /* __GLIBC__ */

#ifdef STARPU_HAVE_HWLOC
		if (workerarg->bindid == -1)
		{
			workerarg->hwloc_cpu_set = hwloc_bitmap_alloc();
			workerarg->hwloc_obj = NULL;
		}
		else
		{
			/* Put the worker descriptor in the userdata field of the
			 * hwloc object describing the CPU */
			hwloc_obj_t worker_obj = hwloc_get_obj_by_depth(config->topology.hwtopology,
									config->pu_depth,
									workerarg->bindid);
			struct _starpu_hwloc_userdata *data = worker_obj->userdata;
			if (data->worker_list == NULL)
				data->worker_list = _starpu_worker_list_new();
			_starpu_worker_list_push_front(data->worker_list, workerarg);

			/* Clear the cpu set and set the cpu */
			workerarg->hwloc_cpu_set = hwloc_bitmap_dup(worker_obj->cpuset);
			workerarg->hwloc_obj = worker_obj;
		}
#endif
		if (workerarg->bindid != -1)
		{
			bindid = workerarg->bindid;
			unsigned old_nbindid = config->nbindid;
			if (bindid >= old_nbindid)
			{
				/* More room needed */
				if (!old_nbindid)
					config->nbindid = STARPU_NMAXWORKERS;
				else
					config->nbindid = 2 * old_nbindid;
				if (bindid >= config->nbindid)
				{
					config->nbindid = bindid+1;
				}
				_STARPU_REALLOC(config->bindid_workers, config->nbindid * sizeof(config->bindid_workers[0]));
				memset(&config->bindid_workers[old_nbindid], 0, (config->nbindid - old_nbindid) * sizeof(config->bindid_workers[0]));
			}
			/* Add slot for this worker */
			/* Don't care about amortizing the cost, there are usually very few workers sharing the same bindid */
			config->bindid_workers[bindid].nworkers++;
			_STARPU_REALLOC(config->bindid_workers[bindid].workerids, config->bindid_workers[bindid].nworkers * sizeof(config->bindid_workers[bindid].workerids[0]));
			config->bindid_workers[bindid].workerids[config->bindid_workers[bindid].nworkers-1] = worker;
		}
	}

	/* Then initialize NUMA nodes accordingly */
	_starpu_init_numa_node(config);
	_starpu_init_numa_bus();

	/* Eventually initialize accelerators memory nodes */
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &config->workers[worker];
		unsigned devid STARPU_ATTRIBUTE_UNUSED = workerarg->devid;

		/* select the memory node that contains worker's memory */
		starpu_driver_info[workerarg->arch].init_worker_memory(config, no_mp_config, workerarg);

		_STARPU_DEBUG("worker %u type %d devid %u STARPU memory node %u\n", worker, workerarg->arch, devid, workerarg->memory_node);
	}

#if defined(STARPU_HAVE_HWLOC) && !defined(STARPU_SIMGRID)
	/* If some NUMA nodes don't have drivers, attribute some */
	unsigned node, nnodes = starpu_memory_nodes_get_count();;
	for (node = 0; node < nnodes; node++)
	{
		if (starpu_node_get_kind(node) != STARPU_CPU_RAM)
			/* Only RAM nodes can be processed by any CPU */
			continue;
		for (worker = 0; worker < config->topology.nworkers; worker++)
		{
			if (_starpu_worker_drives_memory[worker][node])
				break;
		}
		if (worker < config->topology.nworkers)
			/* Already somebody driving it */
			continue;

		/* Nobody driving this node! Attribute some */
		_STARPU_DEBUG("nobody drives memory node %d\n", node);
		hwloc_obj_t numa_node_obj = hwloc_get_obj_by_type(config->topology.hwtopology, HWLOC_OBJ_NUMANODE, starpu_memory_nodes_numa_id_to_hwloclogid(node));
		int ret = _starpu_find_pu_driving_numa_up(numa_node_obj, node);
		STARPU_ASSERT_MSG(ret, "oops, didn't find any worker to drive memory node %d!?", node);
	}
#endif

#ifdef STARPU_SIMGRID
	_starpu_simgrid_count_ngpus();
#else
#ifdef STARPU_HAVE_HWLOC
	_starpu_topology_count_ngpus(hwloc_get_root_obj(config->topology.hwtopology));
#endif
#endif
}

int _starpu_build_topology(struct _starpu_machine_config *config, int no_mp_config)
{
	int ret;
	unsigned i;
	enum starpu_worker_archtype type;

	/* First determine which devices we will use */
	ret = _starpu_init_machine_config(config, no_mp_config);
	if (ret)
		return ret;

	/* for the data management library */
	_starpu_memory_nodes_init();
	_starpu_datastats_init();

	/* Now determine CPU binding and memory nodes */
	_starpu_init_workers_binding_and_memory(config, no_mp_config);

	_starpu_mem_chunk_init_last();

	for (type = 0; type < STARPU_NARCH; type++)
		config->arch_nodeid[type] = -1;

	for (i = 0; i < starpu_worker_get_count(); i++)
	{
		type = starpu_worker_get_type(i);
		if (config->arch_nodeid[type] == -1)
			config->arch_nodeid[type] = starpu_worker_get_memory_node(i);
		else if (config->arch_nodeid[type] != (int) starpu_worker_get_memory_node(i))
			config->arch_nodeid[type] = -2;
	}

	return 0;
}

void _starpu_destroy_topology(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_USE_MPI_MASTER_SLAVE)
	_starpu_deinit_mpi_config(config);
#endif
#if defined(STARPU_USE_TCPIP_MASTER_SLAVE)
	_starpu_deinit_tcpip_config(config);
#endif

	/* cleanup StarPU internal data structures */
	_starpu_memory_nodes_deinit();

	_starpu_destroy_machine_config(config);

	_starpu_deinitialize_workers_bindid(config);
}

void starpu_topology_print(FILE *output)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned pu;
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	unsigned ncombinedworkers = topology->ncombinedworkers;
	unsigned nthreads_per_core = topology->nhwpus / topology->nhwworker[STARPU_CPU_WORKER][0];

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_t topo = topology->hwtopology;
	hwloc_obj_t pu_obj;
	hwloc_obj_t last_numa_obj = (void*) -1, numa_obj;
	hwloc_obj_t last_package_obj = (void*) -1, package_obj;
#endif

	for (pu = 0; pu < topology->nhwpus; pu++)
	{
#ifdef STARPU_HAVE_HWLOC
		pu_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, pu);
		numa_obj = numa_get_obj(pu_obj);
		if (numa_obj != last_numa_obj)
		{
			if (numa_obj)
				fprintf(output, "numa %2u", numa_obj->logical_index);
			else
				fprintf(output, "No numa");
			last_numa_obj = numa_obj;
		}
		fprintf(output, "\t");
		package_obj = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_SOCKET, pu_obj);
		if (package_obj != last_package_obj)
		{
			if (package_obj)
				fprintf(output, "pack %2u", package_obj->logical_index);
			else
				fprintf(output, "no pack");
			last_package_obj = package_obj;
		}
		fprintf(output, "\t");
#endif
		if ((pu % nthreads_per_core) == 0)
			fprintf(output, "core %u", pu / nthreads_per_core);
		fprintf(output, "\tPU %u\t", pu);
		for (worker = 0;
		     worker < nworkers + ncombinedworkers;
		     worker++)
		{
			if (worker < nworkers)
			{
				struct _starpu_worker *workerarg = &config->workers[worker];

				if (workerarg->bindid == (int) pu)
				{
					char name[256];
					starpu_worker_get_name(worker, name, sizeof(name));
					fprintf(output, "%s\t", name);
				}
			}
			else
			{
				int worker_size, i;
				int *combined_workerid;
				starpu_combined_worker_get_description(worker, &worker_size, &combined_workerid);
				for (i = 0; i < worker_size; i++)
				{
					if (topology->workers_bindid[combined_workerid[i]] == pu)
						fprintf(output, "comb %u\t", worker-nworkers);
				}
			}
		}
		fprintf(output, "\n");
	}
}

int starpu_get_pu_os_index(unsigned logical_index)
{
#ifdef STARPU_HAVE_HWLOC
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;

	hwloc_topology_t topo = topology->hwtopology;
	hwloc_obj_t obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, logical_index);
	STARPU_ASSERT(obj);

	return obj->os_index;
#else
	return logical_index;
#endif
}

#ifdef STARPU_HAVE_HWLOC
hwloc_topology_t starpu_get_hwloc_topology(void)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	return config->topology.hwtopology;
}
#endif

unsigned _starpu_get_nhyperthreads()
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	return config->topology.nhwpus / config->topology.nhwworker[STARPU_CPU_WORKER][0];
}

long starpu_get_memory_location_bitmap(void* ptr, size_t size)
{
	if (ptr == NULL || size == 0)
	{
		return -1;
	}

#ifdef HAVE_HWLOC_GET_AREA_MEMLOCATION // implies STARPU_HAVE_HWLOC
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;

	hwloc_bitmap_t set = hwloc_bitmap_alloc();
	int ret = hwloc_get_area_memlocation(topology->hwtopology, ptr, size, set, HWLOC_MEMBIND_BYNODESET);
	if (ret != 0)
	{
		hwloc_bitmap_free(set);
		return -1;
	}

	if (hwloc_bitmap_iszero(set) || hwloc_bitmap_isfull(set))
	{
		// If the page isn't allocated yet, the bitmap is empty:
		hwloc_bitmap_free(set);
		return -1;
	}

	/* We could maybe use starpu_bitmap, but that seems a little bit
	 * overkill and it would make recording it in traces harder. */
	long ret_bitmap = 0;
	unsigned i = 0;
	hwloc_bitmap_foreach_begin(i, set)
	{
		hwloc_obj_t numa_node = hwloc_get_numanode_obj_by_os_index(topology->hwtopology, i);
		if (numa_node)
		{
			ret_bitmap |= (1 << numa_node->logical_index);
		}
		else
		{
			// We can't find a matching NUMA node, this can happen on machine without NUMA node
			hwloc_bitmap_free(set);
			return -1;
		}
	}
	hwloc_bitmap_foreach_end();

	hwloc_bitmap_free(set);
	return ret_bitmap;
#else
	/* we could use move_pages(), but please, rather use hwloc (version >= 1.11.3)! */
	return -1;
#endif
}
