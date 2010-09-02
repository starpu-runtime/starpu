/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#ifndef __STARPU_SCHEDULER_H__
#define __STARPU_SCHEDULER_H__

#include <starpu.h>
#include <starpu_config.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

struct starpu_task;

struct starpu_machine_topology_s {
	unsigned nworkers;

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_t hwtopology;
#else
	/* We maintain ABI compatibility with and without hwloc */
	void *dummy;
#endif

	unsigned nhwcpus;
        unsigned nhwcudagpus;
        unsigned nhwopenclgpus;

	unsigned ncpus;
	unsigned ncudagpus;
	unsigned nopenclgpus;
	unsigned ngordon_spus;

	/* Where to bind workers ? */
	unsigned workers_bindid[STARPU_NMAXWORKERS];
	
	/* Which GPU(s) do we use for CUDA ? */
	unsigned workers_cuda_gpuid[STARPU_NMAXWORKERS];

	/* Which GPU(s) do we use for OpenCL ? */
	unsigned workers_opencl_gpuid[STARPU_NMAXWORKERS];
};

struct starpu_sched_policy_s {
	/* create all the queues */
	void (*init_sched)(struct starpu_machine_topology_s *, struct starpu_sched_policy_s *);

	/* cleanup method at termination */
	void (*deinit_sched)(struct starpu_machine_topology_s *, struct starpu_sched_policy_s *);

	/* some methods to manipulate the previous queue */
	int (*push_task)(struct starpu_task *);
	int (*push_prio_task)(struct starpu_task *);
	struct starpu_task *(*pop_task)(void);

	 /* Remove all available tasks from the scheduler (tasks are chained by
	  * the means of the prev and next fields of the starpu_task
	  * structure). */
	struct starpu_task *(*pop_every_task)(uint32_t where);

	/* name of the policy (optionnal) */
	const char *policy_name;

	/* description of the policy (optionnal) */
	const char *policy_description;
};

void starpu_worker_set_sched_condition(int workerid, pthread_cond_t *sched_cond, pthread_mutex_t *sched_mutex);

#endif // __STARPU_SCHEDULER_H__
