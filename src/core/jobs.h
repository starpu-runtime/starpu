/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#include <starpu.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <common/config.h>
#include <common/timing.h>
#include <common/list.h>
#include <common/fxt.h>
#include <core/dependencies/tags.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>
#include <core/errorcheck.h>
#include <common/barrier.h>
#include <common/utils.h>
#include <common/thread.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

struct _starpu_worker;

/* codelet function */
typedef void (*_starpu_cl_func_t)(void **, void *);

#define _STARPU_CPU_MAY_PERFORM(j)	((j)->task->cl->where & STARPU_CPU)
#define _STARPU_CUDA_MAY_PERFORM(j)      ((j)->task->cl->where & STARPU_CUDA)
#define _STARPU_OPENCL_MAY_PERFORM(j)	((j)->task->cl->where & STARPU_OPENCL)
#define _STARPU_MIC_MAY_PERFORM(j)	((j)->task->cl->where & STARPU_MIC)
#define _STARPU_SCC_MAY_PERFORM(j)	((j)->task->cl->where & STARPU_SCC)

/* A job is the internal representation of a task. */
LIST_TYPE(_starpu_job,

	/* The implementation associated to the job */
	unsigned nimpl;

	/* The task associated to that job */
	struct starpu_task *task;

	/* These synchronization structures are used to wait for the job to be
	 * available or terminated for instance. */
	starpu_pthread_mutex_t sync_mutex;
	starpu_pthread_cond_t sync_cond;

	/* To avoid deadlocks, we reorder the different buffers accessed to by
	 * the task so that we always grab the rw-lock associated to the
	 * handles in the same order. */
	struct starpu_data_descr ordered_buffers[STARPU_NMAXBUFS];
	struct starpu_data_descr *dyn_ordered_buffers;

	/* If a tag is associated to the job, this points to the internal data
	 * structure that describes the tag status. */
	struct _starpu_tag *tag;

	/* Maintain a list of all the completion groups that depend on the job.
	 * */
	struct _starpu_cg_list job_successors;

	/* For tasks with cl==NULL but submitted with explicit data dependency,
	 * the handle for this dependency, so as to remove the task from the
	 * last_writer/readers */
	starpu_data_handle_t implicit_dep_handle;

	/* The value of the footprint that identifies the job may be stored in
	 * this structure. */
	unsigned footprint_is_computed;
	uint32_t footprint;

	/* Indicates whether the task associated to that job has already been
	 * submitted to StarPU (1) or not (0) (using starpu_task_submit).
	 * Becomes and stays 2 when the task is submitted several times.
	 */
	unsigned submitted;

	/* Indicates whether the task associated to this job is terminated or
	 * not. */
	unsigned terminated;

	/* Should that task appear in the debug tools ? (eg. the DAG generated
	 * with dot) */
        unsigned exclude_from_dag;

	/* Is that task internal to StarPU ? */
	unsigned internal;

	/* Each job is attributed a unique id. */
	unsigned long job_id;

	/* During the reduction of a handle, StarPU may have to submit tasks to
	 * perform the reduction itself: those task should not be stalled while
	 * other tasks are blocked until the handle has been properly reduced,
	 * so we need a flag to differentiate them from "normal" tasks. */
	unsigned reduction_task;

	/* Used by MIC driver to record codelet start time instead of using a
	 * local variable */
	struct timespec cl_start;

	/* A symbol name may be associated to the job directly for debug
	 * purposes (for instance if the codelet is NULL). */
        const char *model_name;

	struct bound_task *bound_task;

	/* Number of workers executing that task (>1 if the task is parallel)
	 * */
	int task_size;

	/* In case we have assigned this job to a combined workerid */
	int combined_workerid;

	/* How many workers are currently running an alias of that job (for
	 * parallel tasks only). */
	int active_task_alias_count;

	/* Parallel workers may have to synchronize before/after the execution of a parallel task. */
	_starpu_pthread_barrier_t before_work_barrier;
	_starpu_pthread_barrier_t after_work_barrier;
)

/* Create an internal struct _starpu_job *structure to encapsulate the task. */
struct _starpu_job* STARPU_ATTRIBUTE_MALLOC _starpu_job_create(struct starpu_task *task);

/* Destroy the data structure associated to the job structure */
void _starpu_job_destroy(struct _starpu_job *j);

/* Wait for the termination of the job */
void _starpu_wait_job(struct _starpu_job *j);

/* Specify that the task should not appear in the DAG generated by debug tools. */
void _starpu_exclude_task_from_dag(struct starpu_task *task);

/* try to submit job j, enqueue it if it's not schedulable yet. The job's sync mutex is supposed to be held already */
unsigned _starpu_enforce_deps_and_schedule(struct _starpu_job *j);
unsigned _starpu_enforce_deps_starting_from_task(struct _starpu_job *j);


/* This function must be called after the execution of a job, this triggers all
 * job's dependencies and perform the callback function if any. */
void _starpu_handle_job_termination(struct _starpu_job *j);

/* Get the sum of the size of the data accessed by the job. */
size_t _starpu_job_get_data_size(struct starpu_perfmodel *model, enum starpu_perfmodel_archtype arch, unsigned nimpl, struct _starpu_job *j);

/* Get a task from the local pool of tasks that were explicitly attributed to
 * that worker. */
struct starpu_task *_starpu_pop_local_task(struct _starpu_worker *worker);

/* Put a task into the pool of tasks that are explicitly attributed to the
 * specified worker. If "back" is set, the task is put at the back of the list.
 * Considering the tasks are popped from the back, this value should be 0 to
 * enforce a FIFO ordering. */
int _starpu_push_local_task(struct _starpu_worker *worker, struct starpu_task *task, int prio);

#define _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(job, i) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers[i].handle : job->ordered_buffers[i].handle)
#define _STARPU_JOB_GET_ORDERED_BUFFER_MODE(job, i) ((job->dyn_ordered_buffers) ? job->dyn_ordered_buffers[i].mode : job->ordered_buffers[i].mode)

#define _STARPU_JOB_SET_ORDERED_BUFFER_HANDLE(job, handle, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i].handle = (handle); else job->ordered_buffers[i].handle = (handle);} while(0)
#define _STARPU_JOB_SET_ORDERED_BUFFER_MODE(job, mode, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i].mode = mode; else job->ordered_buffers[i].mode = mode;} while(0)

#define _STARPU_JOB_SET_ORDERED_BUFFER(job, buffer, i) do { if (job->dyn_ordered_buffers) job->dyn_ordered_buffers[i] = buffer; else job->ordered_buffers[i] = buffer;} while(0)
#define _STARPU_JOB_GET_ORDERED_BUFFERS(job) (job->dyn_ordered_buffers) ? job->dyn_ordered_buffers : job->ordered_buffers

#endif // __JOBS_H__
