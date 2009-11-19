/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __WORKERS_H__
#define __WORKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <common/config.h>
#include <pthread.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <core/jobs.h>
#include <core/perfmodel/perfmodel.h>
#include <core/policies/sched_policy.h>
#include <core/topology.h>

#include <starpu.h>

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_GORDON
#include <drivers/gordon/driver_gordon.h>
#endif

#include <drivers/core/driver_core.h>

#include <datawizard/datawizard.h>

#define CORE_ALPHA	1.0f
#define CUDA_ALPHA	13.33f
#define GORDON_ALPHA	6.0f /* XXX this is a random value ... */

#ifdef DATA_STATS
#define BENCHMARK_COMM	1
#else
#define BENCHMARK_COMM	0
#endif

struct worker_s {
	struct machine_config_s *config;
        pthread_mutex_t mutex;
	enum starpu_archtype arch; /* what is the type of worker ? */
	uint32_t worker_mask; /* what is the type of worker ? */
	enum starpu_perf_archtype perf_arch; /* in case there are different models of the same arch */
	pthread_t worker_thread; /* the thread which runs the worker */
	int id; /* which core/gpu/etc is controlled by the workker ? */
	int bindid; /* which core is the driver bound to ? */
	int workerid; /* uniquely identify the worker among all processing units types */
        pthread_cond_t ready_cond; /* indicate when the worker is ready */
	unsigned memory_node; /* which memory node is associated that worker to ? */
	struct jobq_s *jobq; /* in which queue will that worker get/put tasks ? */
	struct job_list_s *local_jobs; /* this queue contains tasks that have been explicitely submitted to that queue */
	pthread_mutex_t local_jobs_mutex; /* protect the local_jobs list */
	struct worker_set_s *set; /* in case this worker belongs to a set */
	struct job_list_s *terminated_jobs; /* list of pending jobs which were executed */
	unsigned worker_is_running;
	unsigned worker_is_initialized;
	char name[32];
};

/* in case a single CPU worker may control multiple 
 * accelerators (eg. Gordon for n SPUs) */
struct worker_set_s {
        pthread_mutex_t mutex;
	pthread_t worker_thread; /* the thread which runs the worker */
	unsigned nworkers;
	unsigned joined; /* only one thread may call pthread_join*/
	void *retval;
	struct worker_s *workers;
        pthread_cond_t ready_cond; /* indicate when the set is ready */
	unsigned set_is_initialized;
};

struct machine_config_s {
	unsigned nworkers;

#ifdef HAVE_HWLOC
	hwloc_topology_t hwtopology;
	int core_depth;
#endif

	unsigned nhwcores;

	unsigned ncores;
	unsigned ncudagpus;
	unsigned ngordon_spus;

	/* Where to bind workers ? */
	int current_bindid;
	unsigned workers_bindid[STARPU_NMAXWORKERS];
	
	/* Which GPU(s) do we use ? */
	int current_gpuid;
	unsigned workers_gpuid[STARPU_NMAXWORKERS];
	
	struct worker_s workers[STARPU_NMAXWORKERS];
	uint32_t worker_mask;

	struct starpu_topo_obj_t *topology;

	/* in case the user gives an explicit configuration, this is only valid
	 * during starpu_init. */
	struct starpu_conf *user_conf;

	/* this flag is set until the runtime is stopped */
	unsigned running;
};

void display_general_stats(void);

unsigned machine_is_running(void);

inline uint32_t worker_exists(uint32_t task_mask);
inline uint32_t may_submit_cuda_task(void);
inline uint32_t may_submit_core_task(void);
inline uint32_t worker_may_execute_task(unsigned workerid, uint32_t where);

void bind_thread_on_cpu(struct machine_config_s *config, unsigned coreid);

inline void lock_all_queues_attached_to_node(unsigned node);
inline void unlock_all_queues_attached_to_node(unsigned node);
inline void broadcast_all_queues_attached_to_node(unsigned node);

void set_local_worker_key(struct worker_s *worker);

struct worker_s *get_worker_struct(unsigned id);

#endif // __WORKERS_H__
