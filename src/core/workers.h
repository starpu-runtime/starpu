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

#include <starpu.h>

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

#define NMAXWORKERS	16

#ifdef DATA_STATS
#define BENCHMARK_COMM	1
#else
#define BENCHMARK_COMM	0
#endif

enum archtype {
	CORE_WORKER,
	CUDA_WORKER,
	GORDON_WORKER
};

struct worker_s {
        pthread_mutex_t mutex;
	enum archtype arch; /* what is the type of worker ? */
	enum starpu_perf_archtype perf_arch; /* in case there are different models of the same arch */
	pthread_t worker_thread; /* the thread which runs the worker */
	int id; /* which core/gpu/etc is controlled by the workker ? */
	int bindid; /* which core is the driver bound to ? */
        pthread_cond_t ready_cond; /* indicate when the worker is ready */
	unsigned memory_node; /* which memory node is associated that worker to ? */
	struct jobq_s *jobq; /* in which queue will that worker get/put tasks ? */
	struct worker_set_s *set; /* in case this worker belongs to a set */
	struct job_list_s *terminated_jobs; /* list of pending jobs which were executed */
	unsigned worker_is_running;
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
};

struct machine_config_s {
	unsigned nworkers;

	struct worker_s workers[NMAXWORKERS];

	/* this flag is set until the runtime is stopped */
	unsigned running;
};

void terminate_workers(struct machine_config_s *config);
void kill_all_workers(struct machine_config_s *config);
void display_general_stats(void);

unsigned machine_is_running(void);

inline uint32_t worker_exists(uint32_t task_mask);
inline uint32_t may_submit_cuda_task(void);
inline uint32_t may_submit_core_task(void);


#endif // __WORKERS_H__
