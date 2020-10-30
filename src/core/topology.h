/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/** Detect the number of memory nodes and where to bind the different workers. */
int _starpu_build_topology(struct _starpu_machine_config *config, int no_mp_config);

/** Should be called instead of _starpu_destroy_topology when _starpu_build_topology returns a non zero value. */
void _starpu_destroy_machine_config(struct _starpu_machine_config *config);

/** Destroy all resources used to store the topology of the machine. */
void _starpu_destroy_topology(struct _starpu_machine_config *config);

/** returns the number of physical cpus */
unsigned _starpu_topology_get_nhwcpu(struct _starpu_machine_config *config);

/** returns the number of logical cpus */
unsigned _starpu_topology_get_nhwpu(struct _starpu_machine_config *config);

/** returns the number of NUMA nodes */
unsigned _starpu_topology_get_nnumanodes(struct _starpu_machine_config *config);

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

int starpu_memory_nodes_get_numa_count(void);
int starpu_memory_nodes_numa_id_to_hwloclogid(unsigned id);

/** Get the memory node for data number i when task is to be executed on memory node target_node */
int _starpu_task_data_get_node_on_node(struct starpu_task *task, unsigned index, unsigned target_node);
int _starpu_task_data_get_node_on_worker(struct starpu_task *task, unsigned index, unsigned worker);

#endif // __TOPOLOGY_H__
