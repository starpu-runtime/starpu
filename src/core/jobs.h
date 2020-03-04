/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __JOBS_H__
#define __JOBS_H__

/** @file */

#include <starpu.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <common/timing.h>
#include <common/list.h>
#include <common/fxt.h>
#include <core/dependencies/tags.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>
#include <core/errorcheck.h>
#include <common/barrier.h>
#include <common/utils.h>
#include <common/list.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

struct _starpu_worker;

/** codelet function */
typedef void (*_starpu_cl_func_t)(void **, void *);

#define _STARPU_CPU_MAY_PERFORM(j)	((j)->task->where & STARPU_CPU)
#define _STARPU_CUDA_MAY_PERFORM(j)      ((j)->task->where & STARPU_CUDA)
#define _STARPU_OPENCL_MAY_PERFORM(j)	((j)->task->where & STARPU_OPENCL)
#define _STARPU_MIC_MAY_PERFORM(j)	((j)->task->where & STARPU_MIC)

struct _starpu_data_descr
{
	starpu_data_handle_t handle;
	enum starpu_data_access_mode mode;
	int node; /** This is the value actually chosen, only set by
		     _starpu_fetch_task_input for coherency with
		     __starpu_push_task_output */
	int index;

	int orderedindex; /** For this field the array is actually indexed by
			     parameter order, and this provides the ordered
			     index */
};

#ifdef STARPU_DEBUG
MULTILIST_CREATE_TYPE(_starpu_job, all_submitted)
#endif
/** A job is the internal representation of a task. */
struct _starpu_job
{
	/** Each job is attributed a unique id. */
	unsigned long job_id;

	/** The task associated to that job */
	struct starpu_task *task;

        /** A task that this will unlock quickly, e.g. we are the pre_sync part
         * of a data acquisition, and the caller promised that data release will
	 * happen immediately, so that the post_sync task will be started
         * immediately after. */
	struct _starpu_job *quick_next;

	/** These synchronization structures are used to wait for the job to be
	 * available or terminated for instance. */
	starpu_pthread_mutex_t sync_mutex;
	starpu_pthread_cond_t sync_cond;

	/** To avoid deadlocks, we reorder the different buffers accessed to by
	 * the task so that we always grab the rw-lock associated to the
	 * handles in the same order. */
	struct _starpu_data_descr ordered_buffers[STARPU_NMAXBUFS];
	struct _starpu_task_wrapper_dlist dep_slots[STARPU_NMAXBUFS];
	struct _starpu_data_descr *dyn_ordered_buffers;
	struct _starpu_task_wrapper_dlist *dyn_dep_slots;

	/** If a tag is associated to the job, this points to the internal data
	 * structure that describes the tag status. */
	struct _starpu_tag *tag;

	/** Maintain a list of all the completion groups that depend on the job.
	 * */
	struct _starpu_cg_list job_successors;

	/** Task whose termination depends on this task */
	struct starpu_task *end_rdep;

	/** For tasks with cl==NULL but submitted with explicit data dependency,
	 * the handle for this dependency, so as to remove the task from the
	 * last_writer/readers */
	starpu_data_handle_t implicit_dep_handle;
	struct _starpu_task_wrapper_dlist implicit_dep_slot;

	/** Indicates whether the task associated to that job has already been
	 * submitted to StarPU (1) or not (0) (using starpu_task_submit).
	 * Becomes and stays 2 when the task is submitted several times.
	 *
	 * Protected by j->sync_mutex.
	 */
	unsigned submitted:2;

	/** Indicates whether the task associated to this job is terminated or
	 * not.
	 *
	 * Protected by j->sync_mutex.
	 */
	unsigned terminated:2;

#ifdef STARPU_OPENMP
	/** Job is a continuation or a regular task. */
	unsigned continuation;

	/** If 0, the prepared continuation is not resubmitted automatically
	 * when going to sleep, if 1, the prepared continuation is immediately
	 * resubmitted when going to sleep. */
	unsigned continuation_resubmit;

	/** Callback function called when:
	 * - The continuation starpu task is ready to be submitted again if
	 *   continuation_resubmit = 0;
	 * - The continuation starpu task has just been re-submitted if
	 *   continuation_resubmit = 1. */
	void (*continuation_callback_on_sleep)(void *arg);
	void *continuation_callback_on_sleep_arg;

	void (*omp_cleanup_callback)(void *arg);
	void *omp_cleanup_callback_arg;

	/** Job has been stopped at least once. */
	unsigned discontinuous;

	/** Cumulated execution time for discontinuous jobs */
	struct timespec cumulated_ts;

	/** Cumulated energy consumption for discontinuous jobs */
	double cumulated_energy_consumed;
#endif

	/** The value of the footprint that identifies the job may be stored in
	 * this structure. */
	uint32_t footprint;
	unsigned footprint_is_computed:1;

	/** Should that task appear in the debug tools ? (eg. the DAG generated
	 * with dot) */
	unsigned exclude_from_dag:1;

	/** Is that task internal to StarPU? */
	unsigned internal:1;
	/** Did that task use sequential consistency for its data? */
	unsigned sequential_consistency:1;

	/** During the reduction of a handle, StarPU may have to submit tasks to
	 * perform the reduction itself: those task should not be stalled while
	 * other tasks are blocked until the handle has been properly reduced,
	 * so we need a flag to differentiate them from "normal" tasks. */
	unsigned reduction_task:1;

	/** The implementation associated to the job */
	unsigned nimpl;

	/** Number of workers executing that task (>1 if the task is parallel)
	 * */
	int task_size;

	/** In case we have assigned this job to a combined workerid */
	int combined_workerid;

	/** How many workers are currently running an alias of that job (for
	 * parallel tasks only). */
	int active_task_alias_count;

	struct bound_task *bound_task;

	/** Parallel workers may have to synchronize before/after the execution of a parallel task. */
	starpu_pthread_barrier_t before_work_barrier;
	starpu_pthread_barrier_t after_work_barrier;
	unsigned after_work_busy_barrier;

	struct _starpu_graph_node *graph_node;

#ifdef STARPU_DEBUG
	/** Linked-list of all jobs, for debugging */
	struct _starpu_job_multilist_all_submitted all_submitted;
#endif
};

#ifdef STARPU_DEBUG
MULTILIST_CREATE_INLINES(struct _starpu_job, _starpu_job, all_submitted)
#endif

void _starpu_job_init(void);
void _starpu_job_fini(void);

/** Create an internal struct _starpu_job *structure to encapsulate the task. */
struct _starpu_job* _starpu_job_create(struct starpu_task *task) STARPU_ATTRIBUTE_MALLOC;

/** Destroy the data structure associated to the job structure */
void _starpu_job_destroy(struct _starpu_job *j);

/** Test for the termination of the job */
int _starpu_job_finished(struct _starpu_job *j);

/** Wait for the termination of the job */
void _starpu_wait_job(struct _starpu_job *j);

#ifdef STARPU_OPENMP
/** Test for the termination of the job */
int _starpu_test_job_termination(struct _starpu_job *j);

/** Prepare the job for accepting new dependencies before becoming a continuation. */

void _starpu_job_prepare_for_continuation_ext(struct _starpu_job *j, unsigned continuation_resubmit,
		void (*continuation_callback_on_sleep)(void *arg), void *continuation_callback_on_sleep_arg);
void _starpu_job_prepare_for_continuation(struct _starpu_job *j);
void _starpu_job_set_omp_cleanup_callback(struct _starpu_job *j,
		void (*omp_cleanup_callback)(void *arg), void *omp_cleanup_callback_arg);
#endif

/** Specify that the task should not appear in the DAG generated by debug tools. */
void _starpu_exclude_task_from_dag(struct starpu_task *task);

/** try to submit job j, enqueue it if it's not schedulable yet. The job's sync mutex is supposed to be held already */
unsigned _starpu_enforce_deps_and_schedule(struct _starpu_job *j);
unsigned _starpu_enforce_deps_starting_from_task(struct _starpu_job *j);
#ifdef STARPU_OPENMP
/** When waking up a continuation, we only enforce new task dependencies */
unsigned _starpu_reenforce_task_deps_and_schedule(struct _starpu_job *j);
#endif
void _starpu_enforce_deps_notify_job_ready_soon(struct _starpu_job *j, _starpu_notify_job_start_data *data, int tag);

/** Called at the submission of the job */
void _starpu_handle_job_submission(struct _starpu_job *j);
/** This function must be called after the execution of a job, this triggers all
 * job's dependencies and perform the callback function if any. */
void _starpu_handle_job_termination(struct _starpu_job *j);

/** Get the sum of the size of the data accessed by the job. */
size_t _starpu_job_get_data_size(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, unsigned nimpl, struct _starpu_job *j);

/** Get a task from the local pool of tasks that were explicitly attributed to
 * that worker. */
struct starpu_task *_starpu_pop_local_task(struct _starpu_worker *worker);

/** Put a task into the pool of tasks that are explicitly attributed to the
 * specified worker. If "back" is set, the task is put at the back of the list.
 * Considering the tasks are popped from the back, this value should be 0 to
 * enforce a FIFO ordering. */
int _starpu_push_local_task(struct _starpu_worker *worker, struct starpu_task *task, int prio);

#define _STARPU_JOB_GET_ORDERED_BUFFER_INDEX(job, i) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers[i].index : job->ordered_buffers[i].index)
#define _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(job, i) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers[i].handle : job->ordered_buffers[i].handle)
#define _STARPU_JOB_GET_ORDERED_BUFFER_MODE(job, i) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers[i].mode : job->ordered_buffers[i].mode)
#define _STARPU_JOB_GET_ORDERED_BUFFER_NODE(job, i) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers[i].node : job->ordered_buffers[i].node)

#define _STARPU_JOB_SET_ORDERED_BUFFER_INDEX(job, __index, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i].index = (__index); else job->ordered_buffers[i].index = (__index);} while(0)
#define _STARPU_JOB_SET_ORDERED_BUFFER_HANDLE(job, __handle, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i].handle = (__handle); else job->ordered_buffers[i].handle = (__handle);} while(0)
#define _STARPU_JOB_SET_ORDERED_BUFFER_MODE(job, __mode, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i].mode = __mode; else job->ordered_buffers[i].mode = __mode;} while(0)
#define _STARPU_JOB_SET_ORDERED_BUFFER_NODE(job, __node, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i].node = __node; else job->ordered_buffers[i].node = __node;} while(0)

#define _STARPU_JOB_SET_ORDERED_BUFFER(job, buffer, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i] = buffer; else job->ordered_buffers[i] = buffer;} while(0)
#define _STARPU_JOB_GET_ORDERED_BUFFERS(job) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers : &job->ordered_buffers[0])

#define _STARPU_JOB_GET_DEP_SLOTS(job) (((job)->dyn_dep_slots) ? (job)->dyn_dep_slots : (job)->dep_slots)

#endif // __JOBS_H__
