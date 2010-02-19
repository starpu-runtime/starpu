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
#include <core/errorcheck.h>

#include <starpu.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef STARPU_USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef STARPU_USE_GORDON
#include <drivers/gordon/driver_gordon.h>
#endif

#include <drivers/cpu/driver_cpu.h>

#include <datawizard/datawizard.h>

#define STARPU_CPU_ALPHA	1.0f
#define STARPU_CUDA_ALPHA	13.33f
#define STARPU_GORDON_ALPHA	6.0f /* XXX this is a random value ... */

#ifdef STARPU_DATA_STATS
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
	int id; /* which cpu/gpu/etc is controlled by the workker ? */
	int bindid; /* which cpu is the driver bound to ? */
	int workerid; /* uniquely identify the worker among all processing units types */
        pthread_cond_t ready_cond; /* indicate when the worker is ready */
	unsigned memory_node; /* which memory node is associated that worker to ? */
	struct jobq_s *jobq; /* in which queue will that worker get/put tasks ? */
	struct starpu_job_list_s *local_jobs; /* this queue contains tasks that have been explicitely submitted to that queue */
	pthread_mutex_t local_jobs_mutex; /* protect the local_jobs list */
	struct worker_set_s *set; /* in case this worker belongs to a set */
	struct starpu_job_list_s *terminated_jobs; /* list of pending jobs which were executed */
	unsigned worker_is_running;
	unsigned worker_is_initialized;
	worker_status status; /* what is the worker doing now ? (eg. CALLBACK) */
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

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_t hwtopology;
	int cpu_depth;
#endif

	unsigned nhwcpus;

	unsigned ncpus;
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

unsigned _starpu_machine_is_running(void);

inline uint32_t _starpu_worker_exists(uint32_t task_mask);
inline uint32_t may_submit_cuda_task(void);
inline uint32_t may_submit_cpu_task(void);
inline uint32_t _starpu_worker_may_execute_task(unsigned workerid, uint32_t where);
unsigned _starpu_worker_can_block(unsigned memnode);

inline void _starpu_lock_all_queues_attached_to_node(unsigned node);
inline void _starpu_unlock_all_queues_attached_to_node(unsigned node);
inline void _starpu_broadcast_all_queues_attached_to_node(unsigned node);

void _starpu_set_local_worker_key(struct worker_s *worker);
struct worker_s *_starpu_get_local_worker_key(void);

struct worker_s *_starpu_get_worker_struct(unsigned id);

struct machine_config_s *_starpu_get_machine_config(void);

/* TODO move */
unsigned _starpu_execute_registered_progression_hooks(void);

#endif // __WORKERS_H__
