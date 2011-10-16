/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2011  INRIA
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

#if ! defined(_MSC_VER)
#  include <pthread.h>
#endif

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

struct starpu_task;

struct starpu_machine_topology_s {
	unsigned nworkers;

	unsigned ncombinedworkers;

	unsigned nsched_ctxs;
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
	void (*init_sched)(unsigned ctx_id);

	/* Initialize the scheduling policy only for certain workers. */
	void (*init_sched_for_workers)(unsigned ctx_id, int *workerids, unsigned nworkers);

	/* Cleanup the scheduling policy. */
	void (*deinit_sched)(unsigned ctx_id);

	/* Insert a task into the scheduler. */
        int (*push_task)(struct starpu_task *, unsigned ctx_id);
	/* Notify the scheduler that a task was directly pushed to the worker
	 * without going through the scheduler. This method is called when a
	 * task is explicitely assigned to a worker. This method therefore
	 * permits to keep the timing state of the scheduler coherent even
	 * when StarPU bypasses the scheduling strategy. */
	void (*push_task_notify)(struct starpu_task *, int workerid, unsigned ctx_id);


	/* Get a task from the scheduler. The mutex associated to the worker is
	 * already taken when this method is called. */
	struct starpu_task *(*pop_task)(unsigned ctx_id);

	 /* Remove all available tasks from the scheduler (tasks are chained by
	  * the means of the prev and next fields of the starpu_task
	  * structure). The mutex associated to the worker is already taken
	  * when this method is called. */
	struct starpu_task *(*pop_every_task)(unsigned ctx_id);

	/* This method is called every time a task has been executed. (optionnal) */
	void (*post_exec_hook)(struct starpu_task *, unsigned ctx_id);

	/* Name of the policy (optionnal) */
	const char *policy_name;

	/* Description of the policy (optionnal) */
	const char *policy_description;
};

struct starpu_sched_ctx_hypervisor_criteria {
	void (*update_current_idle_time)(unsigned sched_ctx, int worker, double idle_time, unsigned current_nprocs);
	void (*update_current_working_time)(unsigned sched_ctx, int worker, double working_time, unsigned current_nprocs);
};

#ifdef STARPU_BUILD_SCHED_CTX_HYPERVISOR
unsigned starpu_create_sched_ctx_with_criteria(const char *policy_name, int *workerids_ctx, int nworkers_ctx, const char *sched_name, struct starpu_sched_ctx_hypervisor_criteria *criteria);
#endif //STARPU_BUILD_SCHED_CTX_HYPERVISOR

unsigned starpu_create_sched_ctx(const char *policy_name, int *workerids_ctx, int nworkers_ctx, const char *sched_name);

void starpu_delete_sched_ctx(unsigned sched_ctx_id, unsigned inheritor_sched_ctx_id);

void starpu_add_workers_to_sched_ctx(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx);

void starpu_remove_workers_from_sched_ctx(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx);

int* starpu_get_workers_of_ctx(unsigned sched_ctx, int *nworkers);

void starpu_require_resize(unsigned sched_ctx, int *workers_to_be_moved, unsigned nworkers_to_be_moved);


/* When there is no available task for a worker, StarPU blocks this worker on a
condition variable. This function specifies which condition variable (and the
associated mutex) should be used to block (and to wake up) a worker. Note that
multiple workers may use the same condition variable. For instance, in the case
of a scheduling strategy with a single task queue, the same condition variable
would be used to block and wake up all workers.  The initialization method of a
scheduling strategy (init_sched) must call this function once per worker. */
#if !defined(_MSC_VER)
void starpu_worker_set_sched_condition(int workerid, pthread_cond_t *sched_cond, pthread_mutex_t *sched_mutex);
#endif

/* Check if the worker specified by workerid can execute the codelet. */
int starpu_worker_may_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

/* The scheduling policy may put tasks directly into a worker's local queue so
 * that it is not always necessary to create its own queue when the local queue
 * is sufficient. If "back" not null, the task is put at the back of the queue
 * where the worker will pop tasks first. Setting "back" to 0 therefore ensures
 * a FIFO ordering. */
int starpu_push_local_task(int workerid, struct starpu_task *task, int back);

/*
 *	Priorities
 */

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

/*
 *	Parallel tasks
 */

/* Register a new combined worker and get its identifier */
int starpu_combined_worker_assign_workerid(int nworkers, int workerid_array[]);
/* Initialize combined workers */
void _starpu_sched_find_worker_combinations(struct starpu_machine_topology_s *topology);
/* Get the description of a combined worker */
int starpu_combined_worker_get_description(int workerid, int *worker_size, int **combined_workerid);
/* Variant of starpu_worker_may_execute_task compatible with combined workers */
int starpu_combined_worker_may_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

/*
 *	Data prefetching
 */

/* Whether STARPU_PREFETCH was set */
int starpu_get_prefetch_flag(void);
/* Prefetch data for a given task on a given node */
int starpu_prefetch_task_input_on_node(struct starpu_task *task, uint32_t node);

/*
 *	Performance predictions
 */

/* Return the current date */
double starpu_timing_now(void);
/* Returns expected task duration in µs */
double starpu_task_expected_length(struct starpu_task *task, enum starpu_perf_archtype arch, unsigned nimpl);
/* Returns an estimated speedup factor relative to CPU speed */
double starpu_worker_get_relative_speedup(enum starpu_perf_archtype perf_archtype);
/* Returns expected data transfer time in µs */
double starpu_task_expected_data_transfer_time(uint32_t memory_node, struct starpu_task *task);
/* Predict the transfer time (in µs) to move a handle to a memory node */
double starpu_data_expected_transfer_time(starpu_data_handle handle, unsigned memory_node, starpu_access_mode mode);
/* Returns expected power consumption in J */
double starpu_task_expected_power(struct starpu_task *task, enum starpu_perf_archtype arch, unsigned nimpl);

/* Waits until all the tasks of a worker, already submitted, have been executed */
int starpu_wait_for_all_tasks_of_worker(int workerid);

/* Waits until all the tasks of a bunch of workers have been executed */
int starpu_wait_for_all_tasks_of_workers(int *workerids_ctx, int nworkers_ctx);

#endif /* __STARPU_SCHEDULER_H__ */
