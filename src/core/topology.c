/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <core/workers.h>
#include <core/debug.h>
#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/mic/driver_mic_source.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_common.h>
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

#if defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
#include <hwloc/cuda.h>
#endif

#if defined(STARPU_HAVE_HWLOC) && defined(STARPU_USE_OPENCL)
#include <hwloc/opencl.h>
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
static int _starpu_get_logical_numa_node_worker(unsigned workerid);

#define STARPU_NUMA_UNINITIALIZED (-2)
#define STARPU_NUMA_MAIN_RAM (-1)

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID) || defined(STARPU_USE_MPI_MASTER_SLAVE)

struct handle_entry
{
	UT_hash_handle hh;
	unsigned gpuid;
};

#  if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
/* Entry in the `devices_using_cuda' hash table.  */
static struct handle_entry *devices_using_cuda;
#  endif

static unsigned may_bind_automatically[STARPU_NARCH] = { 0 };

#endif // defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
static struct _starpu_worker_set cuda_worker_set[STARPU_MAXCUDADEVS];
#endif
#ifdef STARPU_USE_MIC
static struct _starpu_worker_set mic_worker_set[STARPU_MAXMICDEVS];
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
struct _starpu_worker_set mpi_worker_set[STARPU_MAXMPIDEVS];
#endif

int starpu_memory_nodes_get_numa_count(void)
{
	return nb_numa_nodes;
}

#if defined(STARPU_HAVE_HWLOC)
static hwloc_obj_t numa_get_obj(hwloc_obj_t obj)
{
#if HWLOC_API_VERSION >= 0x00020000
	while (obj->memory_first_child == NULL)
	{
		obj = obj->parent;
		if (!obj)
			return NULL;
	}
	return obj->memory_first_child;
#else
	while (obj->type != HWLOC_OBJ_NUMANODE)
	{
		obj = obj->parent;

		/* If we don't find a "node" obj before the root, this means
		 * hwloc does not know whether there are numa nodes or not, so
		 * we should not use a per-node sampling in that case. */
		if (!obj)
			return NULL;
	}
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

/* This returns the exact NUMA node next to a worker */
static int _starpu_get_logical_numa_node_worker(unsigned workerid)
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

		hwloc_obj_t obj;
		switch(worker->arch)
		{
			case STARPU_CPU_WORKER:
				obj = hwloc_get_obj_by_type(topology->hwtopology, HWLOC_OBJ_PU, worker->bindid) ;
				break;
#ifndef STARPU_SIMGRID
#if defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
			case STARPU_CUDA_WORKER:
				obj = hwloc_cuda_get_device_osdev_by_index(topology->hwtopology, worker->devid);
				break;
#endif
#endif
			default:
				return 0;
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
			switch (d->type)
			{
#ifdef STARPU_USE_CPU
			case STARPU_CPU_WORKER:
				if (worker->devid == d->id.cpu_id)
					return worker;
				break;
#endif
#ifdef STARPU_USE_OPENCL
			case STARPU_OPENCL_WORKER:
			{
				cl_device_id device;
				starpu_opencl_get_device(worker->devid, &device);
				if (device == d->id.opencl_id)
					return worker;
				break;
			}
#endif
#ifdef STARPU_USE_CUDA
			case STARPU_CUDA_WORKER:
			{
				if (worker->devid == d->id.cuda_id)
					return worker;
				break;

			}
#endif

			default:
				(void) worker;
				_STARPU_DEBUG("Invalid device type\n");
				return NULL;
			}
		}
	}

	return NULL;
}


/*
 * Discover the topology of the machine
 */

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID) || defined(STARPU_USE_MPI_MASTER_SLAVE)
static void _starpu_initialize_workers_deviceid(int *explicit_workers_gpuid,
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
	 * according to the STARPU_WORKERS_CUDAID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	strval = starpu_getenv(varname);
	if (strval)
	{
		/* STARPU_WORKERS_CUDAID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round
		 * robin fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1
		 * 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the STARPU_WORKERS_CUDAID
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
		may_bind_automatically[type] = 1;
	}
}
#endif

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
static void _starpu_initialize_workers_cuda_gpuid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

        _starpu_initialize_workers_deviceid(uconf->use_explicit_workers_cuda_gpuid == 0
					    ? NULL
					    : (int *)uconf->workers_cuda_gpuid,
					    &(config->current_cuda_gpuid),
					    (int *)topology->workers_cuda_gpuid,
					    "STARPU_WORKERS_CUDAID",
					    topology->nhwcudagpus,
					    STARPU_CUDA_WORKER);
}

static inline int _starpu_get_next_cuda_gpuid(struct _starpu_machine_config *config)
{
	unsigned i = ((config->current_cuda_gpuid++) % config->topology.ncudagpus);

	return (int)config->topology.workers_cuda_gpuid[i];
}
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
static void _starpu_initialize_workers_opencl_gpuid(struct _starpu_machine_config*config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

        _starpu_initialize_workers_deviceid(uconf->use_explicit_workers_opencl_gpuid == 0
					    ? NULL
					    : (int *)uconf->workers_opencl_gpuid,
					    &(config->current_opencl_gpuid),
					    (int *)topology->workers_opencl_gpuid,
					    "STARPU_WORKERS_OPENCLID",
					    topology->nhwopenclgpus,
					    STARPU_OPENCL_WORKER);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
        // Detect devices which are already used with CUDA
        {
                unsigned tmp[STARPU_NMAXWORKERS];
                unsigned nb=0;
                int i;
                for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
		{
			struct handle_entry *entry;
			int devid = config->topology.workers_opencl_gpuid[i];

			HASH_FIND_INT(devices_using_cuda, &devid, entry);
			if (entry == NULL)
			{
                                tmp[nb] = topology->workers_opencl_gpuid[i];
                                nb++;
                        }
                }
                for (i=nb ; i<STARPU_NMAXWORKERS ; i++)
			tmp[i] = -1;
                memcpy(topology->workers_opencl_gpuid, tmp, sizeof(unsigned)*STARPU_NMAXWORKERS);
        }
#endif /* STARPU_USE_CUDA */
        {
                // Detect identical devices
		struct handle_entry *devices_already_used = NULL;
                unsigned tmp[STARPU_NMAXWORKERS];
                unsigned nb=0;
                int i;

                for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
		{
			int devid = topology->workers_opencl_gpuid[i];
			struct handle_entry *entry;
			HASH_FIND_INT(devices_already_used, &devid, entry);
			if (entry == NULL)
			{
				struct handle_entry *entry2;
				_STARPU_MALLOC(entry2, sizeof(*entry2));
				entry2->gpuid = devid;
				HASH_ADD_INT(devices_already_used, gpuid,
					     entry2);
                                tmp[nb] = devid;
                                nb ++;
                        }
                }
		struct handle_entry *entry=NULL, *tempo=NULL;
		HASH_ITER(hh, devices_already_used, entry, tempo)
		{
			HASH_DEL(devices_already_used, entry);
			free(entry);
		}
                for (i=nb ; i<STARPU_NMAXWORKERS ; i++)
			tmp[i] = -1;
                memcpy(topology->workers_opencl_gpuid, tmp, sizeof(unsigned)*STARPU_NMAXWORKERS);
        }
}

static inline int _starpu_get_next_opencl_gpuid(struct _starpu_machine_config *config)
{
	unsigned i = ((config->current_opencl_gpuid++) % config->topology.nopenclgpus);

	return (int)config->topology.workers_opencl_gpuid[i];
}
#endif

#if 0
#if defined(STARPU_USE_MIC) || defined(STARPU_SIMGRID)
static void _starpu_initialize_workers_mic_deviceid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

	_starpu_initialize_workers_deviceid(uconf->use_explicit_workers_mic_deviceid == 0
					    ? NULL
					    : (int *)config->user_conf->workers_mic_deviceid,
					    &(config->current_mic_deviceid),
					    (int *)topology->workers_mic_deviceid,
					    "STARPU_WORKERS_MICID",
					    topology->nhwmiccores,
					    STARPU_MIC_WORKER);
}
#endif
#endif

#if 0
#ifdef STARPU_USE_MIC
static inline int _starpu_get_next_mic_deviceid(struct _starpu_machine_config *config)
{
	unsigned i = ((config->current_mic_deviceid++) % config->topology.nmicdevices);

	return (int)config->topology.workers_mic_deviceid[i];
}
#endif
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
static inline int _starpu_get_next_mpi_deviceid(struct _starpu_machine_config *config)
{
	unsigned i = ((config->current_mpi_deviceid++) % config->topology.nmpidevices);

	return (int)config->topology.workers_mpi_ms_deviceid[i];
}

static void _starpu_init_mpi_topology(struct _starpu_machine_config *config, long mpi_idx)
{
	/* Discover the topology of the mpi node identifier by MPI_IDX. That
	 * means, make this StarPU instance aware of the number of cores available
	 * on this MPI device. Update the `nhwmpicores' topology field
	 * accordingly. */

	struct _starpu_machine_topology *topology = &config->topology;

	int nbcores;
	_starpu_src_common_sink_nbcores(_starpu_mpi_ms_nodes[mpi_idx], &nbcores);
	topology->nhwmpicores[mpi_idx] = nbcores;
}

#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#ifdef STARPU_USE_MIC
static void _starpu_init_mic_topology(struct _starpu_machine_config *config, long mic_idx)
{
	/* Discover the topology of the mic node identifier by MIC_IDX. That
	 * means, make this StarPU instance aware of the number of cores available
	 * on this MIC device. Update the `nhwmiccores' topology field
	 * accordingly. */

	struct _starpu_machine_topology *topology = &config->topology;

	int nbcores;
	_starpu_src_common_sink_nbcores(_starpu_mic_nodes[mic_idx], &nbcores);
	topology->nhwmiccores[mic_idx] = nbcores;
}

static int _starpu_init_mic_node(struct _starpu_machine_config *config, int mic_idx,
				 COIENGINE *coi_handle, COIPROCESS *coi_process)
{
	/* Initialize the MIC node of index MIC_IDX. */

	struct starpu_conf *user_conf = &config->conf;

	char ***argv = _starpu_get_argv();
	const char *suffixes[] = {"-mic", "_mic", NULL};

	/* Environment variables to send to the Sink, it informs it what kind
	 * of node it is (architecture and type) as there is no way to discover
	 * it itself */
	char mic_idx_env[32];
	snprintf(mic_idx_env, sizeof(mic_idx_env), "_STARPU_MIC_DEVID=%d", mic_idx);

	/* XXX: this is currently necessary so that the remote process does not
	 * segfault. */
	char nb_mic_env[32];
	snprintf(nb_mic_env, sizeof(nb_mic_env), "_STARPU_MIC_NB=%d", 2);

	const char *mic_sink_env[] = {"STARPU_SINK=STARPU_MIC", mic_idx_env, nb_mic_env, NULL};

	char mic_sink_program_path[1024];
	/* Let's get the helper program to run on the MIC device */
	int mic_file_found = _starpu_src_common_locate_file(mic_sink_program_path,
							    sizeof(mic_sink_program_path),
							    starpu_getenv("STARPU_MIC_SINK_PROGRAM_NAME"),
							    starpu_getenv("STARPU_MIC_SINK_PROGRAM_PATH"),
							    user_conf->mic_sink_program_path,
							    (argv ? (*argv)[0] : NULL),
							    suffixes);

	if (0 != mic_file_found)
	{
		_STARPU_MSG("No MIC program specified, use the environment\n"
			    "variable STARPU_MIC_SINK_PROGRAM_NAME or the environment\n"
			    "or the field 'starpu_conf.mic_sink_program_path'\n"
			    "to define it.\n");

		return -1;
	}

	COIRESULT res;
	/* Let's get the handle which let us manage the remote MIC device */
	res = COIEngineGetHandle(COI_ISA_MIC, mic_idx, coi_handle);
	if (STARPU_UNLIKELY(res != COI_SUCCESS))
		STARPU_MIC_SRC_REPORT_COI_ERROR(res);

	/* We launch the helper on the MIC device, which will wait for us
	 * to give it work to do.
	 * As we will communicate further with the device throught scif we
	 * don't need to keep the process pointer */
	res = COIProcessCreateFromFile(*coi_handle, mic_sink_program_path, 0, NULL, 0,
				       mic_sink_env, 1, NULL, 0, NULL,
				       coi_process);
	if (STARPU_UNLIKELY(res != COI_SUCCESS))
		STARPU_MIC_SRC_REPORT_COI_ERROR(res);

	/* Let's create the node structure, we'll communicate with the peer
	 * through scif thanks to it */
	_starpu_mic_nodes[mic_idx] =
		_starpu_mp_common_node_create(STARPU_NODE_MIC_SOURCE, mic_idx);

	return 0;
}
#endif

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

	nobind = starpu_get_env_number("STARPU_WORKERS_NOBIND");

	topology->nhwcpus = 0;
	topology->nhwpus = 0;

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_init(&topology->hwtopology);
	char *hwloc_input = starpu_getenv("STARPU_HWLOC_INPUT");
	if (hwloc_input && hwloc_input[0])
	{
		int err = hwloc_topology_set_xml(topology->hwtopology, hwloc_input);
		if (err < 0) _STARPU_DISP("Could not load hwloc input %s\n", hwloc_input);
	}

	_starpu_topology_filter(topology->hwtopology);
	hwloc_topology_load(topology->hwtopology);

	if (starpu_get_env_number_default("STARPU_WORKERS_GETBIND", 0))
	{
		/* Respect the existing binding */
		hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();

		int ret = hwloc_get_cpubind(topology->hwtopology, cpuset, HWLOC_CPUBIND_THREAD);
		if (ret)
			_STARPU_DISP("Warning: could not get current CPU binding: %s\n", strerror(errno));
		else
		{
			ret = hwloc_topology_restrict(topology->hwtopology, cpuset, 0);
			if (ret)
				_STARPU_DISP("Warning: could not restrict hwloc to cpuset: %s\n", strerror(errno));
		}
		hwloc_bitmap_free(cpuset);
	}

	_starpu_allocate_topology_userdata(hwloc_get_root_obj(topology->hwtopology));
#endif
#endif

#ifdef STARPU_SIMGRID
	config->topology.nhwcpus = config->topology.nhwpus = _starpu_simgrid_get_nbhosts("CPU");
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

	topology->nhwcpus = hwloc_get_nbobjs_by_depth(topology->hwtopology, config->cpu_depth);
	topology->nhwpus = hwloc_get_nbobjs_by_depth(topology->hwtopology, config->pu_depth);

#elif defined(HAVE_SYSCONF)
	/* Discover the CPUs relying on the sysconf(3) function and fills
	 * CONFIG accordingly. */

	config->topology.nhwcpus = config->topology.nhwpus = sysconf(_SC_NPROCESSORS_ONLN);

#elif defined(_WIN32)
	/* Discover the CPUs on Cygwin and MinGW systems. */

	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	config->topology.nhwcpus = config->topology.nhwpus = sysinfo.dwNumberOfProcessors;
#else
#warning no way to know number of cores, assuming 1
	config->topology.nhwcpus = config->topology.nhwpus = 1;
#endif

	if (config->conf.ncuda != 0)
		_starpu_cuda_discover_devices(config);
	if (config->conf.nopencl != 0)
		_starpu_opencl_discover_devices(config);
#ifdef STARPU_USE_MPI_MASTER_SLAVE
        config->topology.nhwmpi = _starpu_mpi_src_get_device_count();
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

	config->current_bindid = 0;

	/* conf->workers_bindid indicates the successive logical PU identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	strval = starpu_getenv("STARPU_WORKERS_CPUID");
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
					topology->workers_bindid[i] = (unsigned)(val % topology->nhwpus);
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
							endval = topology->nhwpus-1;
							if (*strval)
								strval++;
						}
						for (val++; val <= endval && i < STARPU_NMAXWORKERS-1; val++)
						{
							i++;
							topology->workers_bindid[i] = (unsigned)(val % topology->nhwpus);
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
	}
	else if (config->conf.use_explicit_workers_bindid)
	{
		/* we use the explicit value from the user */
		memcpy(topology->workers_bindid,
			config->conf.workers_bindid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else
	{
		int nth_per_core = starpu_get_env_number_default("STARPU_NTHREADS_PER_CORE", 1);
		int k;
		int nbindids=0;
		int nhyperthreads = topology->nhwpus / topology->nhwcpus;
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
			topology->workers_bindid[nbindids++] = (unsigned)(i % topology->nhwpus);
			k++;
			i++;
		}
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

/* This function gets the identifier of the next core on which to bind a
 * worker. In case a list of preferred cores was specified (logical indexes),
 * we look for a an available core among the list if possible, otherwise a
 * round-robin policy is used. */
static inline unsigned _starpu_get_next_bindid(struct _starpu_machine_config *config, unsigned flags,
					       unsigned *preferred_binding, unsigned npreferred)
{
	struct _starpu_machine_topology *topology = &config->topology;

	STARPU_ASSERT_MSG(topology_is_initialized, "The StarPU core is not initialized yet, have you called starpu_init?");

	unsigned current_preferred;
	unsigned nhyperthreads = topology->nhwpus / topology->nhwcpus;
	unsigned ncores = topology->nhwpus / nhyperthreads;
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

		/* Look at the remaining cores to be bound to */
		for (i = 0; i < ncores; i++)
		{
			if (topology->workers_bindid[i] == requested_bindid &&
					(!config->currently_bound[i] ||
					 (config->currently_shared[i] && !(flags & STARPU_THREAD_ACTIVE)))
					)
			{
				/* the cpu is available, or shareable with us, we use it ! */
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
		for (i = 0; i < ncores; i++)
			if (config->currently_shared[i])
				return topology->workers_bindid[i];
	}

	/* Try to find an available PU from last used PU */
	for (i = config->current_bindid; i < ncores; i++)
		if (!config->currently_bound[i])
			/* Found a cpu ready for use, use it! */
			break;

	if (i == ncores)
	{
		/* Finished binding on all cpus, restart from start in
		 * case the user really wants overloading */
		memset(&config->currently_bound, 0, sizeof(config->currently_bound));
		i = 0;
	}

	STARPU_ASSERT(i < ncores);
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
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	return config->topology.nhwcpus;
}

unsigned _starpu_topology_get_nhwpu(struct _starpu_machine_config *config)
{
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	return config->topology.nhwpus;
}

unsigned _starpu_topology_get_nnumanodes(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
        _starpu_init_topology(config);

	int res;
#if defined(STARPU_HAVE_HWLOC)
	if (numa_enabled == -1)
		numa_enabled = starpu_get_env_number_default("STARPU_USE_NUMA", 0);
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

#ifdef STARPU_HAVE_HWLOC
void _starpu_topology_filter(hwloc_topology_t topology)
{
#if HWLOC_API_VERSION >= 0x20000
	hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM);
#else
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM | HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES);
#endif
#ifdef HAVE_HWLOC_TOPOLOGY_SET_COMPONENTS
#  ifndef STARPU_USE_CUDA
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "cuda");
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "nvml");
#  endif
#  ifndef STARPU_USE_OPENCL
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "opencl");
#  endif
#endif
}
#endif

#ifdef STARPU_USE_MIC
static void _starpu_init_mic_config(struct _starpu_machine_config *config,
				    struct starpu_conf *user_conf,
				    unsigned mic_idx)
{
	// Configure the MIC device of index MIC_IDX.

	struct _starpu_machine_topology *topology = &config->topology;

	topology->nhwmiccores[mic_idx] = 0;

	_starpu_init_mic_topology(config, mic_idx);

	int nmiccores;
	nmiccores = starpu_get_env_number("STARPU_NMICTHREADS");

	STARPU_ASSERT_MSG(nmiccores >= -1, "nmiccores can not be negative and different from -1 (is is %d)", nmiccores);
	if (nmiccores == -1)
	{
		/* Nothing was specified, so let's use the number of
		 * detected mic cores. ! */
		nmiccores = topology->nhwmiccores[mic_idx];
	}
	else
	{
		if ((unsigned) nmiccores > topology->nhwmiccores[mic_idx])
		{
			/* The user requires more MIC cores than there is available */
			_STARPU_MSG("# Warning: %d MIC cores requested. Only %u available.\n", nmiccores, topology->nhwmiccores[mic_idx]);
			nmiccores = topology->nhwmiccores[mic_idx];
		}
	}

	topology->nmiccores[mic_idx] = nmiccores;
	STARPU_ASSERT_MSG(topology->nmiccores[mic_idx] + topology->nworkers <= STARPU_NMAXWORKERS,
			  "topology->nmiccores[mic_idx(%u)] (%u) + topology->nworkers (%u) <= STARPU_NMAXWORKERS (%d)",
			  mic_idx, topology->nmiccores[mic_idx], topology->nworkers, STARPU_NMAXWORKERS);

	/* _starpu_initialize_workers_mic_deviceid (config); */

	mic_worker_set[mic_idx].workers = &config->workers[topology->nworkers];
	mic_worker_set[mic_idx].nworkers = topology->nmiccores[mic_idx];
	unsigned miccore_id;
	for (miccore_id = 0; miccore_id < topology->nmiccores[mic_idx]; miccore_id++)
	{
		int worker_idx = topology->nworkers + miccore_id;
		config->workers[worker_idx].set = &mic_worker_set[mic_idx];
		config->workers[worker_idx].arch = STARPU_MIC_WORKER;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = STARPU_MIC_WORKER;
		config->workers[worker_idx].perf_arch.devices[0].devid = mic_idx;
		config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
		config->workers[worker_idx].devid = mic_idx;
		config->workers[worker_idx].subworkerid = miccore_id;
		config->workers[worker_idx].worker_mask = STARPU_MIC;
		config->worker_mask |= STARPU_MIC;
	}
	_starpu_mic_nodes[mic_idx]->baseworkerid = topology->nworkers;

	topology->nworkers += topology->nmiccores[mic_idx];
}

static COIENGINE mic_handles[STARPU_MAXMICDEVS];
COIPROCESS _starpu_mic_process[STARPU_MAXMICDEVS];
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
static void _starpu_init_mpi_config(struct _starpu_machine_config *config,
				    struct starpu_conf *user_conf,
				    unsigned mpi_idx)
{
        struct _starpu_machine_topology *topology = &config->topology;

        topology->nhwmpicores[mpi_idx] = 0;

        _starpu_init_mpi_topology(config, mpi_idx);

        int nmpicores;
        nmpicores = starpu_get_env_number("STARPU_NMPIMSTHREADS");

        if (nmpicores == -1)
        {
                /* Nothing was specified, so let's use the number of
                 * detected mpi cores. ! */
                nmpicores = topology->nhwmpicores[mpi_idx];
        }
        else
        {
                if ((unsigned) nmpicores > topology->nhwmpicores[mpi_idx])
                {
                        /* The user requires more MPI cores than there is available */
                        _STARPU_MSG("# Warning: %d MPI cores requested. Only %u available.\n",
				    nmpicores, topology->nhwmpicores[mpi_idx]);
                        nmpicores = topology->nhwmpicores[mpi_idx];
                }
        }

        topology->nmpicores[mpi_idx] = nmpicores;
        STARPU_ASSERT_MSG(topology->nmpicores[mpi_idx] + topology->nworkers <= STARPU_NMAXWORKERS,
                        "topology->nmpicores[mpi_idx(%u)] (%u) + topology->nworkers (%u) <= STARPU_NMAXWORKERS (%d)",
                        mpi_idx, topology->nmpicores[mpi_idx], topology->nworkers, STARPU_NMAXWORKERS);

        mpi_worker_set[mpi_idx].workers = &config->workers[topology->nworkers];
        mpi_worker_set[mpi_idx].nworkers = topology->nmpicores[mpi_idx];
        unsigned mpicore_id;
        for (mpicore_id = 0; mpicore_id < topology->nmpicores[mpi_idx]; mpicore_id++)
        {
                int worker_idx = topology->nworkers + mpicore_id;
                config->workers[worker_idx].set = &mpi_worker_set[mpi_idx];
                config->workers[worker_idx].arch = STARPU_MPI_MS_WORKER;
                _STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
                config->workers[worker_idx].perf_arch.ndevices = 1;
                config->workers[worker_idx].perf_arch.devices[0].type = STARPU_MPI_MS_WORKER;
                config->workers[worker_idx].perf_arch.devices[0].devid = mpi_idx;
                config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
                config->workers[worker_idx].devid = mpi_idx;
                config->workers[worker_idx].subworkerid = mpicore_id;
                config->workers[worker_idx].worker_mask = STARPU_MPI_MS;
                config->worker_mask |= STARPU_MPI_MS;
        }
	_starpu_mpi_ms_nodes[mpi_idx]->baseworkerid = topology->nworkers;

        topology->nworkers += topology->nmpicores[mpi_idx];
}
#endif

#if defined(STARPU_USE_MIC) || defined(STARPU_USE_MPI_MASTER_SLAVE)
static void _starpu_init_mp_config(struct _starpu_machine_config *config,
				   struct starpu_conf *user_conf, int no_mp_config)
{
	/* Discover and configure the mp topology. That means:
	 * - discover the number of mp nodes;
	 * - initialize each discovered node;
	 * - discover the local topology (number of PUs/devices) of each node;
	 * - configure the workers accordingly.
	 */

#ifdef STARPU_USE_MIC
	if (!no_mp_config)
	{
		struct _starpu_machine_topology *topology = &config->topology;

		/* Discover and initialize the number of MIC nodes through the mp
		 * infrastructure. */
		unsigned nhwmicdevices = _starpu_mic_src_get_device_count();

		int reqmicdevices = starpu_get_env_number("STARPU_NMIC");
		if (reqmicdevices == -1 && user_conf)
			reqmicdevices = user_conf->nmic;
		if (reqmicdevices == -1)
			/* Nothing was specified, so let's use the number of
			 * detected mic devices. ! */
			reqmicdevices = nhwmicdevices;

		STARPU_ASSERT_MSG(reqmicdevices >= -1, "nmic can not be negative and different from -1 (is is %d)", reqmicdevices);
		if (reqmicdevices != -1)
		{
			if ((unsigned) reqmicdevices > nhwmicdevices)
			{
				/* The user requires more MIC devices than there is available */
				_STARPU_MSG("# Warning: %d MIC devices requested. Only %u available.\n", reqmicdevices, nhwmicdevices);
				reqmicdevices = nhwmicdevices;
			}
		}

		topology->nmicdevices = 0;
		unsigned i;
		for (i = 0; i < (unsigned) reqmicdevices; i++)
			if (0 == _starpu_init_mic_node(config, i, &mic_handles[i], &_starpu_mic_process[i]))
				topology->nmicdevices++;

		for (i = 0; i < topology->nmicdevices; i++)
			_starpu_init_mic_config(config, user_conf, i);
	}
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	{
		struct _starpu_machine_topology *topology = &config->topology;

		/* Discover and initialize the number of MPI nodes through the mp
		 * infrastructure. */
		unsigned nhwmpidevices = _starpu_mpi_src_get_device_count();

		int reqmpidevices = starpu_get_env_number("STARPU_NMPI_MS");
		if (reqmpidevices == -1 && user_conf)
			reqmpidevices = user_conf->nmpi_ms;
		if (reqmpidevices == -1)
			/* Nothing was specified, so let's use the number of
			 * detected mpi devices. ! */
			reqmpidevices = nhwmpidevices;

		if (reqmpidevices != -1)
		{
			if ((unsigned) reqmpidevices > nhwmpidevices)
			{
				/* The user requires more MPI devices than there is available */
				_STARPU_MSG("# Warning: %d MPI Master-Slave devices requested. Only %u available.\n",
					    reqmpidevices, nhwmpidevices);
				reqmpidevices = nhwmpidevices;
			}
		}

		topology->nmpidevices = reqmpidevices;

		/* if user don't want to use MPI slaves, we close the slave processes */
		if (no_mp_config && topology->nmpidevices == 0)
		{
			_starpu_mpi_common_mp_deinit();
			exit(0);
		}

		if (!no_mp_config)
		{
			unsigned i;
			for (i = 0; i < topology->nmpidevices; i++)
				_starpu_mpi_ms_nodes[i] = _starpu_mp_common_node_create(STARPU_NODE_MPI_SOURCE, i);

			for (i = 0; i < topology->nmpidevices; i++)
				_starpu_init_mpi_config(config, user_conf, i);
		}
	}
#endif
}
#endif

#ifdef STARPU_USE_MIC
static void _starpu_deinit_mic_node(unsigned mic_idx)
{
	_starpu_mp_common_send_command(_starpu_mic_nodes[mic_idx], STARPU_MP_COMMAND_EXIT, NULL, 0);

	COIProcessDestroy(_starpu_mic_process[mic_idx], -1, 0, NULL, NULL);

	_starpu_mp_common_node_destroy(_starpu_mic_nodes[mic_idx]);
}
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
static void _starpu_deinit_mpi_node(int devid)
{
        _starpu_mp_common_send_command(_starpu_mpi_ms_nodes[devid], STARPU_MP_COMMAND_EXIT, NULL, 0);

        _starpu_mp_common_node_destroy(_starpu_mpi_ms_nodes[devid]);
}
#endif


#if defined(STARPU_USE_MIC) || defined(STARPU_USE_MPI_MASTER_SLAVE)
static void _starpu_deinit_mp_config(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned i;

#ifdef STARPU_USE_MIC
	for (i = 0; i < topology->nmicdevices; i++)
		_starpu_deinit_mic_node(i);
	_starpu_mic_clear_kernels();
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	for (i = 0; i < topology->nmpidevices; i++)
		_starpu_deinit_mpi_node(i);
#endif
}
#endif

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

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	_starpu_initialize_workers_bindid(config);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	for (i = 0; i < (int) (sizeof(cuda_worker_set)/sizeof(cuda_worker_set[0])); i++)
		cuda_worker_set[i].workers = NULL;
#endif
#ifdef STARPU_USE_MIC
	for (i = 0; i < (int) (sizeof(mic_worker_set)/sizeof(mic_worker_set[0])); i++)
		mic_worker_set[i].workers = NULL;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	for (i = 0; i < (int) (sizeof(mpi_worker_set)/sizeof(mpi_worker_set[0])); i++)
		mpi_worker_set[i].workers = NULL;
#endif

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	int ncuda = config->conf.ncuda;
	int nworker_per_cuda = starpu_get_env_number_default("STARPU_NWORKER_PER_CUDA", 1);

	STARPU_ASSERT_MSG(nworker_per_cuda > 0, "STARPU_NWORKER_PER_CUDA has to be > 0");
	STARPU_ASSERT_MSG(nworker_per_cuda < STARPU_NMAXWORKERS, "STARPU_NWORKER_PER_CUDA (%d) cannot be higher than STARPU_NMAXWORKERS (%d)\n", nworker_per_cuda, STARPU_NMAXWORKERS);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	if (nworker_per_cuda > 1)
	{
		_STARPU_DISP("Warning: reducing STARPU_NWORKER_PER_CUDA to 1 because blocking drivers are enabled\n");
		nworker_per_cuda = 1;
	}
#endif

	if (ncuda != 0)
	{
		/* The user did not disable CUDA. We need to initialize CUDA
 		 * early to count the number of devices */
		_starpu_init_cuda();
		int nb_devices = _starpu_get_cuda_device_count();

		STARPU_ASSERT_MSG(ncuda >= -1, "ncuda can not be negative and different from -1 (is is %d)", ncuda);
		if (ncuda == -1)
		{
			/* Nothing was specified, so let's choose ! */
			ncuda = nb_devices;
		}
		else
		{
			if (ncuda > nb_devices)
			{
				/* The user requires more CUDA devices than
				 * there is available */
				_STARPU_DISP("Warning: %d CUDA devices requested. Only %d available.\n", ncuda, nb_devices);
				ncuda = nb_devices;
			}
		}
	}

	/* Now we know how many CUDA devices will be used */
	topology->ncudagpus = ncuda;
	topology->nworkerpercuda = nworker_per_cuda;
	STARPU_ASSERT(topology->ncudagpus <= STARPU_MAXCUDADEVS);

	_starpu_initialize_workers_cuda_gpuid(config);

	/* allow having one worker per stream */
	topology->cuda_th_per_stream = starpu_get_env_number_default("STARPU_CUDA_THREAD_PER_WORKER", -1);
	topology->cuda_th_per_dev = starpu_get_env_number_default("STARPU_CUDA_THREAD_PER_DEV", -1);

	STARPU_ASSERT_MSG(!(topology->cuda_th_per_stream == 1 && topology->cuda_th_per_dev != -1), "It does not make sense to set both STARPU_CUDA_THREAD_PER_WORKER to 1 and to set STARPU_CUDA_THREAD_PER_DEV, please choose either per worker or per device or none");

	/* per device by default */
	if (topology->cuda_th_per_dev == -1)
	{
		if (topology->cuda_th_per_stream == 1)
			topology->cuda_th_per_dev = 0;
		else
			topology->cuda_th_per_dev = 1;
	}
	/* Not per stream by default */
	if (topology->cuda_th_per_stream == -1)
	{
		topology->cuda_th_per_stream = 0;
	}

	if (!topology->cuda_th_per_dev)
	{
		cuda_worker_set[0].workers = &config->workers[topology->nworkers];
		cuda_worker_set[0].nworkers = topology->ncudagpus * nworker_per_cuda;
	}

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < topology->ncudagpus; cudagpu++)
	{
		int devid = _starpu_get_next_cuda_gpuid(config);
		int worker_idx0 = topology->nworkers + cudagpu * nworker_per_cuda;
		struct _starpu_worker_set *worker_set;

		if (topology->cuda_th_per_dev)
		{
			worker_set = &cuda_worker_set[devid];
			worker_set->workers = &config->workers[worker_idx0];
			worker_set->nworkers = nworker_per_cuda;
		}
		else
		{
			/* Same worker set for all devices */
			worker_set = &cuda_worker_set[0];
		}

		for (i = 0; i < nworker_per_cuda; i++)
		{
			int worker_idx = worker_idx0 + i;
			if(topology->cuda_th_per_stream)
			{
				/* Just one worker in the set */
				_STARPU_CALLOC(config->workers[worker_idx].set, 1, sizeof(struct _starpu_worker_set));
				config->workers[worker_idx].set->workers = &config->workers[worker_idx];
				config->workers[worker_idx].set->nworkers = 1;
			}
			else
				config->workers[worker_idx].set = worker_set;

			config->workers[worker_idx].arch = STARPU_CUDA_WORKER;
			_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
			config->workers[worker_idx].perf_arch.ndevices = 1;
			config->workers[worker_idx].perf_arch.devices[0].type = STARPU_CUDA_WORKER;
			config->workers[worker_idx].perf_arch.devices[0].devid = devid;
			// TODO: fix perfmodels etc.
			//config->workers[worker_idx].perf_arch.ncore = nworker_per_cuda - 1;
			config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
			config->workers[worker_idx].devid = devid;
			config->workers[worker_idx].subworkerid = i;
			config->workers[worker_idx].worker_mask = STARPU_CUDA;
			config->worker_mask |= STARPU_CUDA;

			struct handle_entry *entry;
			HASH_FIND_INT(devices_using_cuda, &devid, entry);
			if (!entry)
			{
				_STARPU_MALLOC(entry, sizeof(*entry));
				entry->gpuid = devid;
				HASH_ADD_INT(devices_using_cuda, gpuid, entry);
			}
		}

#ifndef STARPU_SIMGRID
#if defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
		{
			hwloc_obj_t obj = hwloc_cuda_get_device_osdev_by_index(topology->hwtopology, devid);
			if (obj)
			{
				struct _starpu_hwloc_userdata *data = obj->userdata;
				data->ngpus++;
			}
			else
			{
				_STARPU_DEBUG("Warning: could not find location of CUDA%u, do you have the hwloc CUDA plugin installed?\n", devid);
			}
		}
#endif
#endif
        }

	topology->nworkers += topology->ncudagpus * nworker_per_cuda;
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	int nopencl = config->conf.nopencl;

	if (nopencl != 0)
	{
		/* The user did not disable OPENCL. We need to initialize
 		 * OpenCL early to count the number of devices */
		_starpu_opencl_init();
		int nb_devices;
		nb_devices = _starpu_opencl_get_device_count();

		STARPU_ASSERT_MSG(nopencl >= -1, "nopencl can not be negative and different from -1 (is is %d)", nopencl);
		if (nopencl == -1)
		{
			/* Nothing was specified, so let's choose ! */
			nopencl = nb_devices;
			if (nopencl > STARPU_MAXOPENCLDEVS)
			{
				_STARPU_DISP("Warning: %d OpenCL devices available. Only %d enabled. Use configure option --enable-maxopencldadev=xxx to update the maximum value of supported OpenCL devices.\n", nb_devices, STARPU_MAXOPENCLDEVS);
				nopencl = STARPU_MAXOPENCLDEVS;
			}
		}
		else
		{
			/* Let's make sure this value is OK. */
			if (nopencl > nb_devices)
			{
				/* The user requires more OpenCL devices than
				 * there is available */
				_STARPU_DISP("Warning: %d OpenCL devices requested. Only %d available.\n", nopencl, nb_devices);
				nopencl = nb_devices;
			}
			/* Let's make sure this value is OK. */
			if (nopencl > STARPU_MAXOPENCLDEVS)
			{
				_STARPU_DISP("Warning: %d OpenCL devices requested. Only %d enabled. Use configure option --enable-maxopencldev=xxx to update the maximum value of supported OpenCL devices.\n", nopencl, STARPU_MAXOPENCLDEVS);
				nopencl = STARPU_MAXOPENCLDEVS;
			}
		}
	}

	topology->nopenclgpus = nopencl;
	STARPU_ASSERT(topology->nopenclgpus + topology->nworkers <= STARPU_NMAXWORKERS);

	_starpu_initialize_workers_opencl_gpuid(config);

	unsigned openclgpu;
	for (openclgpu = 0; openclgpu < topology->nopenclgpus; openclgpu++)
	{
		int worker_idx = topology->nworkers + openclgpu;
		int devid = _starpu_get_next_opencl_gpuid(config);
		if (devid == -1)
		{
			// There is no more devices left
			topology->nopenclgpus = openclgpu;
			break;
		}
		config->workers[worker_idx].arch = STARPU_OPENCL_WORKER;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = STARPU_OPENCL_WORKER;
		config->workers[worker_idx].perf_arch.devices[0].devid = devid;
		config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
		config->workers[worker_idx].subworkerid = 0;
		config->workers[worker_idx].devid = devid;
		config->workers[worker_idx].worker_mask = STARPU_OPENCL;
		config->worker_mask |= STARPU_OPENCL;
	}

	topology->nworkers += topology->nopenclgpus;
#endif

#if defined(STARPU_USE_MIC) || defined(STARPU_USE_MPI_MASTER_SLAVE)
	    _starpu_init_mp_config(config, &config->conf, no_mp_config);
#endif

/* we put the CPU section after the accelerator : in case there was an
 * accelerator found, we devote one cpu */
#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
	int ncpu = config->conf.ncpus;

	if (ncpu != 0)
	{
		STARPU_ASSERT_MSG(ncpu >= -1, "ncpus can not be negative and different from -1 (is is %d)", ncpu);
		if (ncpu == -1)
		{
			unsigned mic_busy_cpus = 0;
			int j = 0;
			for (j = 0; j < STARPU_MAXMICDEVS; j++)
				mic_busy_cpus += (topology->nmiccores[j] ? 1 : 0);

			unsigned mpi_ms_busy_cpus = 0;
#ifdef STARPU_USE_MPI_MASTER_SLAVE
#ifdef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
			for (j = 0; j < STARPU_MAXMPIDEVS; j++)
				mpi_ms_busy_cpus += (topology->nmpicores[j] ? 1 : 0);
#else
			mpi_ms_busy_cpus = 1; /* we launch one thread to control all slaves */
#endif
#endif /* STARPU_USE_MPI_MASTER_SLAVE */
			unsigned cuda_busy_cpus = 0;
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
			cuda_busy_cpus =
				topology->cuda_th_per_dev == 0 && topology->cuda_th_per_stream == 0 ? (topology->ncudagpus ? 1 : 0) :
				topology->cuda_th_per_stream ? (nworker_per_cuda * topology->ncudagpus) : topology->ncudagpus;
#endif
			unsigned already_busy_cpus = mpi_ms_busy_cpus + mic_busy_cpus
				+ cuda_busy_cpus
				+ topology->nopenclgpus;

			long avail_cpus = (long) topology->nhwcpus - (long) already_busy_cpus;
			if (avail_cpus < 0)
				avail_cpus = 0;
			int nth_per_core = starpu_get_env_number_default("STARPU_NTHREADS_PER_CORE", 1);
			avail_cpus *= nth_per_core;

			ncpu = avail_cpus;
		}

		if (ncpu > STARPU_MAXCPUS)
		{
			_STARPU_DISP("Warning: %d CPU cores requested. Only %d enabled. Use configure option --enable-maxcpus=xxx to update the maximum value of supported CPU devices.\n", ncpu, STARPU_MAXCPUS);
			ncpu = STARPU_MAXCPUS;
		}

		if (config->conf.reserve_ncpus > 0)
		{
			if (ncpu < config->conf.reserve_ncpus)
			{
				_STARPU_DISP("Warning: %d CPU cores were requested to be reserved, but only %d were available,\n", config->conf.reserve_ncpus, ncpu);
				ncpu = 0;
			}
			else
			{
				ncpu -= config->conf.reserve_ncpus;
			}
		}

	}

	topology->ncpus = ncpu;
	STARPU_ASSERT(topology->ncpus + topology->nworkers <= STARPU_NMAXWORKERS);

	unsigned cpu;
	unsigned homogeneous = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_CPU", 1);
	for (cpu = 0; cpu < topology->ncpus; cpu++)
	{
		int worker_idx = topology->nworkers + cpu;
		config->workers[worker_idx].arch = STARPU_CPU_WORKER;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices,  sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = STARPU_CPU_WORKER;
		config->workers[worker_idx].perf_arch.devices[0].devid = homogeneous ? 0 : cpu;
		config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
		config->workers[worker_idx].subworkerid = 0;
		config->workers[worker_idx].devid = cpu;
		config->workers[worker_idx].worker_mask = STARPU_CPU;
		config->worker_mask |= STARPU_CPU;
	}

	topology->nworkers += topology->ncpus;
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
#ifdef STARPU_USE_CUDA
	struct handle_entry *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, devices_using_cuda, entry, tmp)
	{
		HASH_DEL(devices_using_cuda, entry);
		free(entry);
	}
	devices_using_cuda = NULL;
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	int i;
	for (i=0; i<STARPU_NARCH; i++)
		may_bind_automatically[i] = 0;
#endif
}

int _starpu_bind_thread_on_cpu(int cpuid STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, const char *name)
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

#ifdef STARPU_USE_OPENCL
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#ifdef STARPU_USE_CUDA
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	if (workerid != STARPU_NOWORKERID && cpuid < STARPU_MAXCPUS)
	{
/* TODO: mutex... */
		int previous = cpu_worker[cpuid];
		/* We would like the PU to be available, or we are perhaps fine to share it */
		if ( !(  previous == STARPU_NOWORKERID ||
			 (previous == STARPU_NONACTIVETHREAD && workerid == STARPU_NONACTIVETHREAD) ||
			 (previous >= 0 && previous == workerid) ||
			 (name && cpu_name[cpuid] && !strcmp(name, cpu_name[cpuid])) ) )
		{
			if (previous == STARPU_ACTIVETHREAD)
				_STARPU_DISP("Warning: active thread %s was already bound to PU %d\n", cpu_name[cpuid], cpuid);
			else if (previous == STARPU_NONACTIVETHREAD)
				_STARPU_DISP("Warning: non-active thread %s was already bound to PU %d\n", cpu_name[cpuid], cpuid);
			else
				_STARPU_DISP("Warning: worker %d was already bound to PU %d\n", previous, cpuid);

			if (workerid == STARPU_ACTIVETHREAD)
				_STARPU_DISP("and we were told to also bind active thread %s to it.\n", name);
			else if (previous == STARPU_NONACTIVETHREAD)
				_STARPU_DISP("and we were told to also bind non-active thread %s to it.\n", name);
			else
				_STARPU_DISP("and we were told to also bind worker %d to it.\n", workerid);

			_STARPU_DISP("This will strongly degrade performance.\n");

			if (workerid >= 0)
				/* This shouldn't happen for workers */
				_STARPU_DISP("Maybe check starpu_machine_display's output to determine what wrong binding happened. Hwloc reported %d cores and %d threads, perhaps there is misdetection between hwloc, the kernel and the BIOS, or an administrative allocation issue from e.g. the job scheduler?\n", config->topology.nhwcpus, config->topology.nhwpus);
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

#ifdef STARPU_USE_OPENC
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#ifdef STARPU_USE_CUDA
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
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

static void _starpu_init_binding_cpu(struct _starpu_machine_config *config)
{
	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &config->workers[worker];

		switch (workerarg->arch)
		{
			case STARPU_CPU_WORKER:
			{
				/* Dedicate a cpu core to that worker */
				workerarg->bindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, NULL, 0);
				break;
			}
			default:
				/* Do nothing */
				break;
		}


	}
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
			limit = starpu_get_env_number(name);
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
		limit = starpu_get_env_number("STARPU_LIMIT_CPU_NUMA_MEM");

	if (limit == -1)
	{
		limit = starpu_get_env_number("STARPU_LIMIT_CPU_MEM");
		if (limit != -1 && numa_enabled)
		{
			_STARPU_DISP("NUMA is enabled and STARPU_LIMIT_CPU_MEM is set to %luMB. Assuming that it should be distributed over the %d NUMA node(s). You probably want to use STARPU_LIMIT_CPU_NUMA_MEM instead.\n", (long) limit, _starpu_topology_get_nnumanodes(config));
			limit /= _starpu_topology_get_nnumanodes(config);
		}
	}

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

	numa_enabled = starpu_get_env_number_default("STARPU_USE_NUMA", 0);
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
					_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nnumanodes(config), STARPU_MAXNUMANODES);
					STARPU_ABORT();
				}

				if (numa_starpu_id == -1)
				{
					int devid = numa_logical_id == STARPU_NUMA_MAIN_RAM ? 0 : numa_logical_id;
					int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, devid, &_starpu_driver_cpu_node_ops);
					_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(devid, config));
					STARPU_ASSERT_MSG(memnode < STARPU_MAXNUMANODES, "Wrong Memory Node : %d (only %d available)", memnode, STARPU_MAXNUMANODES);
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

#if (defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)) && defined(STARPU_HAVE_HWLOC)
		_STARPU_DISP("Take NUMA nodes attached to CUDA and OpenCL devices...\n");
#endif

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_HWLOC)
		for (i = 0; i < config->topology.ncudagpus; i++)
		{
			hwloc_obj_t obj = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, i);
			if (obj)
				obj = numa_get_obj(obj);
			/* Hwloc cannot recognize some devices */
			if (!obj)
				continue;
			int numa_starpu_id = starpu_memory_nodes_numa_hwloclogid_to_id(obj->logical_index);

			/* This shouldn't happen */
			if (numa_starpu_id == -1 && nb_numa_nodes == STARPU_MAXNUMANODES)
			{
				_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nnumanodes(config), STARPU_MAXNUMANODES);
				STARPU_ABORT();
			}

			if (numa_starpu_id == -1)
			{
				int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, obj->logical_index, &_starpu_driver_cpu_node_ops);
				_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(obj->logical_index, config));
				STARPU_ASSERT_MSG(memnode < STARPU_MAXNUMANODES, "Wrong Memory Node : %d (only %d available)", memnode, STARPU_MAXNUMANODES);
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
#endif
#if defined(STARPU_USE_OPENCL) && defined(STARPU_HAVE_HWLOC)
		if (config->topology.nopenclgpus > 0)
		{
			cl_int err;
			cl_platform_id platform_id[_STARPU_OPENCL_PLATFORM_MAX];
			cl_uint nb_platforms;
			unsigned platform;
			unsigned nb_opencl_devices = 0, num = 0;

			err = clGetPlatformIDs(_STARPU_OPENCL_PLATFORM_MAX, platform_id, &nb_platforms);
			if (STARPU_UNLIKELY(err != CL_SUCCESS))
				nb_platforms=0;

			cl_device_type device_type = CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_ACCELERATOR;
			if (starpu_get_env_number("STARPU_OPENCL_ON_CPUS") > 0)
				device_type |= CL_DEVICE_TYPE_CPU;
			if (starpu_get_env_number("STARPU_OPENCL_ONLY_ON_CPUS") > 0)
				device_type = CL_DEVICE_TYPE_CPU;

			for (platform = 0; platform < nb_platforms ; platform++)
			{
				err = clGetDeviceIDs(platform_id[platform], device_type, 0, NULL, &num);
				if (err != CL_SUCCESS)
					num = 0;
				nb_opencl_devices += num;

				for (i = 0; i < num; i++)
				{
					hwloc_obj_t obj = hwloc_opencl_get_device_osdev_by_index(config->topology.hwtopology, platform, i);
					if (obj)
						obj = numa_get_obj(obj);
					/* Hwloc cannot recognize some devices */
					if (!obj)
						continue;
					int numa_starpu_id = starpu_memory_nodes_numa_hwloclogid_to_id(obj->logical_index);

					/* This shouldn't happen */
					if (numa_starpu_id == -1 && nb_numa_nodes == STARPU_MAXNUMANODES)
					{
						_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nnumanodes(config), STARPU_MAXNUMANODES);
						STARPU_ABORT();
					}

					if (numa_starpu_id == -1)
					{
						int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, obj->logical_index, &_starpu_driver_cpu_node_ops);
						_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(obj->logical_index, config));
						STARPU_ASSERT_MSG(memnode < STARPU_MAXNUMANODES, "Wrong Memory Node : %d (only %d available)", memnode, STARPU_MAXNUMANODES);
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
		}
#endif
	}

#if (defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)) && defined(STARPU_HAVE_HWLOC)
	//Found NUMA nodes from CUDA nodes
	if (nb_numa_nodes != 0)
		return;

	/* In case, we do not find any NUMA nodes when checking NUMA nodes attached to GPUs, we take all of them */
	if (numa_enabled)
		_STARPU_DISP("No NUMA nodes found when checking GPUs devices...\n");
#endif

	if (numa_enabled)
		_STARPU_DISP("Finally, take all NUMA nodes available... \n");

	unsigned nnuma = _starpu_topology_get_nnumanodes(config);
	if (nnuma > STARPU_MAXNUMANODES)
	{
		_STARPU_MSG("Warning: %u NUMA nodes available. Only %u enabled. Use configure option --enable-maxnumanodes=xxx to update the maximum value of supported NUMA nodes.\n", _starpu_topology_get_nnumanodes(config), STARPU_MAXNUMANODES);
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
		int memnode = _starpu_memory_node_register(STARPU_CPU_RAM, numa_logical_id, &_starpu_driver_cpu_node_ops);
		_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cpu_get_global_mem_size(numa_logical_id, config));

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

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	unsigned cuda_init[STARPU_MAXCUDADEVS] = { };
	unsigned cuda_memory_nodes[STARPU_MAXCUDADEVS];
	unsigned cuda_bindid[STARPU_MAXCUDADEVS];
	int cuda_globalbindid = -1;
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	unsigned opencl_init[STARPU_MAXOPENCLDEVS] = { };
	unsigned opencl_memory_nodes[STARPU_MAXOPENCLDEVS];
	unsigned opencl_bindid[STARPU_MAXOPENCLDEVS];
#endif
#ifdef STARPU_USE_MIC
	unsigned mic_init[STARPU_MAXMICDEVS] = { };
	unsigned mic_memory_nodes[STARPU_MAXMICDEVS];
	unsigned mic_bindid[STARPU_MAXMICDEVS];
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	unsigned mpi_init[STARPU_MAXMPIDEVS] = { };
	unsigned mpi_memory_nodes[STARPU_MAXMPIDEVS];
	unsigned mpi_bindid[STARPU_MAXMPIDEVS];
#endif

	unsigned bindid;

	for (bindid = 0; bindid < config->nbindid; bindid++)
	{
		free(config->bindid_workers[bindid].workerids);
		config->bindid_workers[bindid].workerids = NULL;
		config->bindid_workers[bindid].nworkers = 0;
	}

	/* Init CPU binding before NUMA nodes, because we use it to discover NUMA nodes */
	_starpu_init_binding_cpu(config);

	/* Initialize NUMA nodes */
	_starpu_init_numa_node(config);
	_starpu_init_numa_bus();

	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		unsigned memory_node = -1;
		struct _starpu_worker *workerarg = &config->workers[worker];
		unsigned devid STARPU_ATTRIBUTE_UNUSED = workerarg->devid;

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_USE_MIC) || defined(STARPU_SIMGRID) || defined(STARPU_USE_MPI_MASTER_SLAVE)
		/* Perhaps the worker has some "favourite" bindings  */
		unsigned *preferred_binding = NULL;
		unsigned npreferred = 0;
#endif

		/* select the memory node that contains worker's memory */
		switch (workerarg->arch)
		{
			case STARPU_CPU_WORKER:
			{
				int numa_logical_id = _starpu_get_logical_numa_node_worker(worker);
				int numa_starpu_id =  starpu_memory_nodes_numa_hwloclogid_to_id(numa_logical_id);
				if (numa_starpu_id < 0 || numa_starpu_id >= STARPU_MAXNUMANODES)
					numa_starpu_id = STARPU_MAIN_RAM;

#if defined(STARPU_HAVE_HWLOC) && !defined(STARPU_SIMGRID)
				hwloc_obj_t pu_obj = hwloc_get_obj_by_type(config->topology.hwtopology, HWLOC_OBJ_PU, workerarg->bindid);
				struct _starpu_hwloc_userdata *userdata = pu_obj->userdata;
				userdata->pu_worker = workerarg;
#endif

				workerarg->numa_memory_node = memory_node = numa_starpu_id;

				_starpu_memory_node_add_nworkers(memory_node);

				_starpu_worker_drives_memory_node(workerarg, numa_starpu_id);
				break;
			}
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
			case STARPU_CUDA_WORKER:
			{
				unsigned numa;
#ifndef STARPU_SIMGRID
				if (may_bind_automatically[STARPU_CUDA_WORKER])
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_cuda_affinity_vector(devid);
					npreferred = config->topology.nhwpus;
				}
#endif /* SIMGRID */
				if (cuda_init[devid])
				{
					memory_node = cuda_memory_nodes[devid];
					if (config->topology.cuda_th_per_stream == 0)
						workerarg->bindid = cuda_bindid[devid];
					else
						workerarg->bindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
				}
				else
				{
					cuda_init[devid] = 1;
					if (config->topology.cuda_th_per_dev == 0 && config->topology.cuda_th_per_stream == 0)
					{
						if (cuda_globalbindid == -1)
							cuda_globalbindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
						workerarg->bindid = cuda_bindid[devid] = cuda_globalbindid;
					}
					else
						workerarg->bindid = cuda_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
					memory_node = cuda_memory_nodes[devid] = _starpu_memory_node_register(STARPU_CUDA_RAM, devid, &_starpu_driver_cuda_node_ops);

					for (numa = 0; numa < nb_numa_nodes; numa++)
					{
						_starpu_cuda_bus_ids[numa][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(numa, memory_node);
						_starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][numa] = _starpu_register_bus(memory_node, numa);
					}
#ifdef STARPU_SIMGRID
					const char* cuda_memcpy_peer;
					char name[16];
					snprintf(name, sizeof(name), "CUDA%u", devid);
					starpu_sg_host_t host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memory_node, host);
#  ifdef STARPU_HAVE_SIMGRID_ACTOR_H
					cuda_memcpy_peer = sg_host_get_property_value(host, "memcpy_peer");
#  else
					cuda_memcpy_peer = MSG_host_get_property_value(host, "memcpy_peer");
#  endif
#endif /* SIMGRID */
					if (
#ifdef STARPU_SIMGRID
						cuda_memcpy_peer && atoll(cuda_memcpy_peer)
#elif defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
						1
#else /* MEMCPY_PEER */
						0
#endif /* MEMCPY_PEER */
					   )
					{
						unsigned worker2;
						for (worker2 = 0; worker2 < worker; worker2++)
						{
							struct _starpu_worker *workerarg2 = &config->workers[worker2];
							int devid2 = workerarg2->devid;
							if (workerarg2->arch == STARPU_CUDA_WORKER)
							{
								unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
								_starpu_cuda_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node2, memory_node);
								_starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node, memory_node2);
#ifndef STARPU_SIMGRID
#if defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
								{
									hwloc_obj_t obj, obj2, ancestor;
									obj = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, devid);
									obj2 = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, devid2);
									ancestor = hwloc_get_common_ancestor_obj(config->topology.hwtopology, obj, obj2);
									if (ancestor)
									{
										struct _starpu_hwloc_userdata *data = ancestor->userdata;
#ifdef STARPU_VERBOSE
										{
											char name[64];
											hwloc_obj_type_snprintf(name, sizeof(name), ancestor, 0);
											_STARPU_DEBUG("CUDA%u and CUDA%u are linked through %s, along %u GPUs\n", devid, devid2, name, data->ngpus);
										}
#endif
										starpu_bus_set_ngpus(_starpu_cuda_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES], data->ngpus);
										starpu_bus_set_ngpus(_starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES], data->ngpus);
									}
								}
#endif
#endif
							}
						}
					}
				}
				_starpu_memory_node_add_nworkers(memory_node);

				//This worker can manage transfers on NUMA nodes
				for (numa = 0; numa < nb_numa_nodes; numa++)
						_starpu_worker_drives_memory_node(&workerarg->set->workers[0], numa);

				_starpu_worker_drives_memory_node(&workerarg->set->workers[0], memory_node);
				break;
			}
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
		        case STARPU_OPENCL_WORKER:
			{
				unsigned numa;
#ifndef STARPU_SIMGRID
				if (may_bind_automatically[STARPU_OPENCL_WORKER])
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_opencl_affinity_vector(devid);
					npreferred = config->topology.nhwpus;
				}
#endif /* SIMGRID */
				if (opencl_init[devid])
				{
					memory_node = opencl_memory_nodes[devid];
#ifndef STARPU_SIMGRID
					workerarg->bindid = opencl_bindid[devid];
#endif /* SIMGRID */
				}
				else
				{
					opencl_init[devid] = 1;
					workerarg->bindid = opencl_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
					memory_node = opencl_memory_nodes[devid] = _starpu_memory_node_register(STARPU_OPENCL_RAM, devid, &_starpu_driver_opencl_node_ops);

					for (numa = 0; numa < nb_numa_nodes; numa++)
					{
						_starpu_register_bus(numa, memory_node);
						_starpu_register_bus(memory_node, numa);
					}
#ifdef STARPU_SIMGRID
					char name[16];
					snprintf(name, sizeof(name), "OpenCL%u", devid);
					starpu_sg_host_t host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memory_node, host);
#endif /* SIMGRID */
				}
				_starpu_memory_node_add_nworkers(memory_node);

				//This worker can manage transfers on NUMA nodes
				for (numa = 0; numa < nb_numa_nodes; numa++)
						_starpu_worker_drives_memory_node(workerarg, numa);

				_starpu_worker_drives_memory_node(workerarg, memory_node);
				break;
			}
#endif

#ifdef STARPU_USE_MIC
		        case STARPU_MIC_WORKER:
			{
				unsigned numa;
				if (mic_init[devid])
				{
					memory_node = mic_memory_nodes[devid];
				}
				else
				{
					mic_init[devid] = 1;
					/* TODO */
					//if (may_bind_automatically)
					//{
					//	/* StarPU is allowed to bind threads automatically */
						//	preferred_binding = _starpu_get_mic_affinity_vector(devid);
					//	npreferred = config->topology.nhwpus;
					//}
					mic_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
					memory_node = mic_memory_nodes[devid] = _starpu_memory_node_register(STARPU_MIC_RAM, devid, &_starpu_driver_mic_node_ops);

					for (numa = 0; numa < nb_numa_nodes; numa++)
					{
						_starpu_register_bus(numa, memory_node);
						_starpu_register_bus(memory_node, numa);
					}

				}
				workerarg->bindid = mic_bindid[devid];
				_starpu_memory_node_add_nworkers(memory_node);

				//This worker can manage transfers on NUMA nodes
				for (numa = 0; numa < nb_numa_nodes; numa++)
						_starpu_worker_drives_memory_node(&workerarg->set->workers[0], numa);

				_starpu_worker_drives_memory_node(&workerarg->set->workers[0], memory_node);
				break;
			}
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_MPI_MASTER_SLAVE
			case STARPU_MPI_MS_WORKER:
			{
				unsigned numa;
				if (mpi_init[devid])
				{
					memory_node = mpi_memory_nodes[devid];
				}
				else
				{
					mpi_init[devid] = 1;
					mpi_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
					memory_node = mpi_memory_nodes[devid] = _starpu_memory_node_register(STARPU_MPI_MS_RAM, devid, &_starpu_driver_mpi_node_ops);

					for (numa = 0; numa < nb_numa_nodes; numa++)
					{
						_starpu_register_bus(numa, memory_node);
						_starpu_register_bus(memory_node, numa);
					}

				}
				//This worker can manage transfers on NUMA nodes
				for (numa = 0; numa < nb_numa_nodes; numa++)
						_starpu_worker_drives_memory_node(&workerarg->set->workers[0], numa);

				_starpu_worker_drives_memory_node(&workerarg->set->workers[0], memory_node);
#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
                                /* MPI driver thread can manage all slave memories if we disable the MPI multiple thread */
                                unsigned findworker;
                                for (findworker = 0; findworker < worker; findworker++)
                                {
                                        struct _starpu_worker *findworkerarg = &config->workers[findworker];
                                        if (findworkerarg->arch == STARPU_MPI_MS_WORKER)
                                        {
                                                _starpu_worker_drives_memory_node(workerarg, findworkerarg->memory_node);
                                                _starpu_worker_drives_memory_node(findworkerarg, memory_node);
                                        }
                                }
#endif

				workerarg->bindid = mpi_bindid[devid];
				_starpu_memory_node_add_nworkers(memory_node);
				break;
			}
#endif /* STARPU_USE_MPI_MASTER_SLAVE */


			default:
				STARPU_ABORT();
		}

		workerarg->memory_node = memory_node;

		_STARPU_DEBUG("worker %u type %d devid %u bound to cpu %d, STARPU memory node %u\n", worker, workerarg->arch, devid, workerarg->bindid, memory_node);

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

	ret = _starpu_init_machine_config(config, no_mp_config);
	if (ret)
		return ret;

	/* for the data management library */
	_starpu_memory_nodes_init();
	_starpu_datastats_init();

	_starpu_init_workers_binding_and_memory(config, no_mp_config);

	_starpu_mem_chunk_init_last();

	config->cpus_nodeid = -1;
	config->cuda_nodeid = -1;
	config->opencl_nodeid = -1;
	config->mic_nodeid = -1;
        config->mpi_nodeid = -1;
	for (i = 0; i < starpu_worker_get_count(); i++)
	{
		switch (starpu_worker_get_type(i))
		{
			case STARPU_CPU_WORKER:
				if (config->cpus_nodeid == -1)
					config->cpus_nodeid = starpu_worker_get_memory_node(i);
				else if (config->cpus_nodeid != (int) starpu_worker_get_memory_node(i))
					config->cpus_nodeid = -2;
				break;
			case STARPU_CUDA_WORKER:
				if (config->cuda_nodeid == -1)
					config->cuda_nodeid = starpu_worker_get_memory_node(i);
				else if (config->cuda_nodeid != (int) starpu_worker_get_memory_node(i))
					config->cuda_nodeid = -2;
				break;
			case STARPU_OPENCL_WORKER:
				if (config->opencl_nodeid == -1)
					config->opencl_nodeid = starpu_worker_get_memory_node(i);
				else if (config->opencl_nodeid != (int) starpu_worker_get_memory_node(i))
					config->opencl_nodeid = -2;
				break;
			case STARPU_MIC_WORKER:
				if (config->mic_nodeid == -1)
					config->mic_nodeid = starpu_worker_get_memory_node(i);
				else if (config->mic_nodeid != (int) starpu_worker_get_memory_node(i))
					config->mic_nodeid = -2;
				break;
			case STARPU_MPI_MS_WORKER:
				if (config->mpi_nodeid == -1)
					config->mpi_nodeid = starpu_worker_get_memory_node(i);
				else if (config->mpi_nodeid != (int) starpu_worker_get_memory_node(i))
					config->mpi_nodeid = -2;
				break;
			case STARPU_ANY_WORKER:
				STARPU_ASSERT(0);
		}
	}

	return 0;
}

void _starpu_destroy_topology(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_USE_MIC) || defined(STARPU_USE_MPI_MASTER_SLAVE)
	_starpu_deinit_mp_config(config);
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
	unsigned nthreads_per_core = topology->nhwpus / topology->nhwcpus;

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_t topo = topology->hwtopology;
	hwloc_obj_t pu_obj;
	hwloc_obj_t last_numa_obj = NULL, numa_obj;
	hwloc_obj_t last_package_obj = NULL, package_obj;
#endif

	for (pu = 0; pu < topology->nhwpus; pu++)
	{
#ifdef STARPU_HAVE_HWLOC
		pu_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, pu);
		numa_obj = numa_get_obj(pu_obj);
		if (numa_obj != last_numa_obj)
		{
			fprintf(output, "numa %u", numa_obj->logical_index);
			last_numa_obj = numa_obj;
		}
		fprintf(output, "\t");
		package_obj = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_SOCKET, pu_obj);
		if (package_obj != last_package_obj)
		{
			fprintf(output, "pack %u", package_obj->logical_index);
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

	return hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, logical_index)->os_index;
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
