/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#ifndef __STARPU_SCHEDULER_H__
#define __STARPU_SCHEDULER_H__

#include <starpu.h>
#include <starpu_config.h>
#include <pthread.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

struct starpu_task;

struct starpu_machine_topology_s {
	unsigned nworkers;

	unsigned ncombinedworkers;

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

/* This structure contains all the methods that implement a scheduling policy.
 * An application may specify which scheduling strategy in the "sched_policy"
 * field of the starpu_conf structure passed to the starpu_init function. */
struct starpu_sched_policy_s {
	/* Initialize the scheduling policy. */
	void (*init_sched)(struct starpu_machine_topology_s *, struct starpu_sched_policy_s *);

	/* Cleanup the scheduling policy. */
	void (*deinit_sched)(struct starpu_machine_topology_s *, struct starpu_sched_policy_s *);

	/* Insert a task into the scheduler. */
	int (*push_task)(struct starpu_task *);

	/* Insert a priority task into the scheduler. */
	int (*push_prio_task)(struct starpu_task *);

	/* Get a task from the scheduler. The mutex associated to the worker is
	 * already taken when this method is called. */
	struct starpu_task *(*pop_task)(void);

	 /* Remove all available tasks from the scheduler (tasks are chained by
	  * the means of the prev and next fields of the starpu_task
	  * structure). The mutex associated to the worker is already taken
	  * when this method is called. */
	struct starpu_task *(*pop_every_task)(void);

	/* This method is called every time a task has been executed. (optionnal) */
	void (*post_exec_hook)(struct starpu_task *);

	/* Name of the policy (optionnal) */
	const char *policy_name;

	/* Description of the policy (optionnal) */
	const char *policy_description;
};

/* When there is no available task for a worker, StarPU blocks this worker on a
condition variable. This function specifies which condition variable (and the
associated mutex) should be used to block (and to wake up) a worker. Note that
multiple workers may use the same condition variable. For instance, in the case
of a scheduling strategy with a single task queue, the same condition variable
would be used to block and wake up all workers.  The initialization method of a
scheduling strategy (init_sched) must call this function once per worker. */
void starpu_worker_set_sched_condition(int workerid, pthread_cond_t *sched_cond, pthread_mutex_t *sched_mutex);

/* Provided for legacy reasons */
#define STARPU_MIN_PRIO		(starpu_sched_get_min_priority())
#define STARPU_MAX_PRIO		(starpu_sched_get_max_priority())

/* By convention, the default priority level should be 0 so that we can
 * statically allocate tasks with a default priority. */
#define STARPU_DEFAULT_PRIO	0

int starpu_sched_get_min_priority(void);
int starpu_sched_get_max_priority(void);

void starpu_sched_set_min_priority(int min_prio);
void starpu_sched_set_max_priority(int max_prio);

int starpu_combined_worker_assign_workerid(int nworkers, int workerid_array[]);

/* Return the current date */
double starpu_timing_now(void);

/* Check if the worker specified by workerid can execute the codelet. */
int starpu_worker_may_execute_task(unsigned workerid, struct starpu_task *task);
int starpu_combined_worker_may_execute_task(unsigned workerid, struct starpu_task *task);

/* Whether STARPU_PREFETCH was set */
int starpu_get_prefetch_flag(void);
/* Prefetch data for a given task on a given node */
int starpu_prefetch_task_input_on_node(struct starpu_task *task, uint32_t node);

/* Initialize combined workers */
void _starpu_sched_find_worker_combinations(struct starpu_machine_topology_s *topology);

int starpu_combined_worker_get_description(int workerid, int *worker_size, int **combined_workerid);

#endif // __STARPU_SCHEDULER_H__
