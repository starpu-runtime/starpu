/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPURM_PRIVATE_H
#define __STARPURM_PRIVATE_H

/** @file */

enum e_state
{
	state_uninitialized = 0,
	state_init
};

enum e_starpurm_unit_type
{
	starpurm_unit_cpu    = 0,
	starpurm_unit_opencl = 1,
	starpurm_unit_cuda   = 2,
	starpurm_unit_mic    = 3,
	starpurm_unit_ntypes = 4
};

struct s_starpurm
{
	/** Machine topology as detected by hwloc. */
	hwloc_topology_t topology;

	/** Current upper bound on the number of CPU cores selectable for computing with the runtime system. */
	unsigned max_ncpus;

	/** Number of currently selected CPU workers */
	unsigned selected_ncpus;

	/** Number of currently selected workers (CPU+devices) */
	unsigned selected_nworkers;

	/** Initialization state of the RM instance. */
	int state;

	/** Boolean indicating the state of the dynamic resource sharing layer.
	 *
	 * !0 indicates that dynamic resource sharing is enabled.
	 * 0 indicates that dynamic resource sharing is disabled.
	 */
	int dynamic_resource_sharing;

	/** Id of the StarPU's sched_ctx used by the RM instance. */
	unsigned sched_ctx_id;

	/** Number of unit types supported by this RM instance. */
	int unit_ntypes;

	/** Number of unitss available for each type. */
	int *nunits_by_type;

	/** Number of units. */
	int nunits;

	/** Offset of unit numbering for each type. */
	int *unit_offsets_by_type;

	/** Array of units. */
	struct s_starpurm_unit *units;

	/** Cpuset of all the StarPU's workers (CPU+devices. */
	hwloc_cpuset_t global_cpuset;

	/** Cpuset of all StarPU CPU workers. */
	hwloc_cpuset_t all_cpu_workers_cpuset;

	/** Cpuset of all StarPU OpenCL workers. */
	hwloc_cpuset_t all_opencl_device_workers_cpuset;

	/** Cpuset of all StarPU CUDA workers. */
	hwloc_cpuset_t all_cuda_device_workers_cpuset;

	/** Cpuset of all StarPU MIC workers. */
	hwloc_cpuset_t all_mic_device_workers_cpuset;

	/** Cpuset of all StarPU device workers. */
	hwloc_cpuset_t all_device_workers_cpuset;

	/** Cpuset of all selected workers (CPU+devices). */
	hwloc_cpuset_t selected_cpuset;

	/** Cpuset mask of initially owned cpuset or full if not used. */
	hwloc_cpuset_t initially_owned_cpuset_mask;

	/** maximum value among worker ids */
	int max_worker_id;

	/** worker id to unit id table */
	int *worker_unit_ids;

	/** Temporary contexts accounting. */
	unsigned int max_temporary_ctxs;
	unsigned int avail_temporary_ctxs;
	pthread_mutex_t temporary_ctxs_mutex;
	pthread_cond_t temporary_ctxs_cond;

	/** Global StarPU pause state */
	int starpu_in_pause;

	/** Event list. */
	pthread_t event_thread;
	pthread_mutex_t event_list_mutex;
	pthread_cond_t event_list_cond;
	pthread_cond_t event_processing_cond;
	int event_processing_enabled;
	int event_processing_ended;
	struct s_starpurm_event *event_list_head;
	struct s_starpurm_event *event_list_tail;
};


#ifdef STARPURM_HAVE_DLB
void starpurm_dlb_init(struct s_starpurm *rm);
void starpurm_dlb_exit(void);
int starpurm_dlb_notify_starpu_worker_mask_going_to_sleep(const hwloc_cpuset_t hwloc_workers_cpuset);
int starpurm_dlb_notify_starpu_worker_mask_waking_up(const hwloc_cpuset_t hwloc_workers_cpuset);
#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
void starpurm_enqueue_event_cpu_unit_available(int cpuid);
#endif
#endif

#endif /* __STARPURM_PRIVATE_H */
