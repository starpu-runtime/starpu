/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __TOPOLOGY_H__
#define __TOPOLOGY_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <common/list.h>
#include <common/fxt.h>
#include <common/uthash.h>

#pragma GCC visibility push(hidden)

struct _starpu_machine_config;

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
/** This is allocated for each hwloc object */
struct _starpu_hwloc_userdata
{
	 /** List of workers running on this obj */
	struct _starpu_worker_list *worker_list;
	 /** Number of GPUs sharing this PCI link */
	unsigned ngpus;
	/** Worker running this PU */
	struct _starpu_worker *pu_worker;
};
#endif
#endif

struct _starpu_worker_set;
struct _starpu_machine_topology;

/** Detect the number of memory nodes and where to bind the different workers. */
int _starpu_build_topology(struct _starpu_machine_config *config, int no_mp_config);

/** Fill workers_gpuid with ids, either commit from explicit_workers_gpuid or from the environment variable \p named varname */
void _starpu_initialize_workers_deviceid(int *explicit_workers_gpuid,
					 int *current, int *workers_gpuid,
					 const char *varname, unsigned nhwgpus,
					 enum starpu_worker_archtype type);

/** Get the next devid for architecture \p type */
int _starpu_get_next_devid(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config, enum starpu_worker_archtype arch);

/** Check that \p *ndevices is not larger than \p nhwdevices (unless overflow is 1), and is not larger than \p max.
 * Cap it otherwise, and advise using the configurename ./configure option in the \p max case. */
void _starpu_topology_check_ndevices(int *ndevices, unsigned nhwdevices, int overflow, unsigned max, const char *nname, const char *dname, const char *configurename);

/** Configures the topology according to the desired worker distribution on the device.
 * - homogeneous tells to use devid 0 for the perfmodel (all devices have the same performance)
 * - worker_devid tells to set a devid per worker, and subworkerid to 0, rather
 * than sharing the devid and giving a different subworkerid to each worker.
 */

/** Request to allocate a worker set for each worker */
#define ALLOC_WORKER_SET ((struct _starpu_worker_set*) -1)

/** Request to set a different perfmodel devid per worker */
#define DEVID_PER_WORKER -2

void _starpu_topology_configure_workers(struct _starpu_machine_topology *topology,
					struct _starpu_machine_config *config,
					enum starpu_worker_archtype type,
					int devnum, int devid,
					int homogeneous, int worker_devid,
					unsigned nworker_per_device,
					unsigned ncores,
					struct _starpu_worker_set *worker_set,
					struct _starpu_worker_set *driver_worker_set);

extern unsigned _starpu_may_bind_automatically[STARPU_NARCH];

/** This function gets the identifier of the next core on which to bind a
 * worker. In case a list of preferred cores was specified (logical indexes),
 * we look for a an available core among the list if possible, otherwise a
 * round-robin policy is used. */
unsigned _starpu_get_next_bindid(struct _starpu_machine_config *config, unsigned flags,
				 unsigned *preferred_binding, unsigned npreferred);

/** Should be called instead of _starpu_destroy_topology when _starpu_build_topology returns a non zero value. */
void _starpu_destroy_machine_config(struct _starpu_machine_config *config);

/** Destroy all resources used to store the topology of the machine. */
void _starpu_destroy_topology(struct _starpu_machine_config *config);

/** returns the number of physical cpus */
unsigned _starpu_topology_get_nhwcpu(struct _starpu_machine_config *config);

/** returns the number of NUMA nodes */
unsigned _starpu_topology_get_nnumanodes(struct _starpu_machine_config *config);

/** given a list of numa nodes (logical indexes) \p numa_binding, fill \p binding with the corresponding cores (logical indexes) */
unsigned _starpu_topology_get_numa_core_binding(struct _starpu_machine_config *config, const unsigned *numa_binding, unsigned nnuma, unsigned *binding, unsigned nbinding);

int starpu_memory_nodes_numa_hwloclogid_to_id(int logid);

/* This returns the exact NUMA node next to a worker */
int _starpu_get_logical_numa_node_worker(unsigned workerid);

/** returns the number of hyperthreads per core */
unsigned _starpu_get_nhyperthreads() STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

#ifdef STARPU_HAVE_HWLOC
/** Small convenient function to filter hwloc topology depending on HWLOC API version */
void _starpu_topology_filter(hwloc_topology_t topology);
#endif

#define STARPU_NOWORKERID -1
#define STARPU_ACTIVETHREAD -2
#define STARPU_NONACTIVETHREAD -2
/** Bind the current thread on the CPU logically identified by "cpuid". The
 * logical ordering of the processors is either that of hwloc (if available),
 * or the ordering exposed by the OS. */
int _starpu_bind_thread_on_cpu(int cpuid, int workerid, const char *name);

struct _starpu_combined_worker;
/** Bind the current thread on the set of CPUs for the given combined worker. */
void _starpu_bind_thread_on_cpus(struct _starpu_combined_worker *combined_worker);

struct _starpu_worker *_starpu_get_worker_from_driver(struct starpu_driver *d);

unsigned starpu_memory_nodes_get_numa_count(void) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;
int starpu_memory_nodes_numa_id_to_hwloclogid(unsigned id);

/** Get the memory node for data number i when task is to be executed on memory node \p target_node. Returns -1 if the data does not need to be loaded. */
int _starpu_task_data_get_node_on_node(struct starpu_task *task, unsigned index, unsigned target_node);
/** Get the memory node for data number i when task is to be executed on worker \p worker. Returns -1 if the data does not need to be loaded. */
int _starpu_task_data_get_node_on_worker(struct starpu_task *task, unsigned index, unsigned worker);

#pragma GCC visibility pop

#endif // __TOPOLOGY_H__
