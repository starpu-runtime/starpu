/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
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

#ifndef __SCHED_CONTEXT_H__
#define __SCHED_CONTEXT_H__

/** @file */

#include <starpu.h>
#include <starpu_sched_ctx.h>
#include <starpu_sched_ctx_hypervisor.h>
#include <starpu_scheduler.h>
#include <common/config.h>
#include <common/barrier_counter.h>
#include <common/utils.h>
#include <profiling/profiling.h>
#include <semaphore.h>
#include <core/task.h>
#include "sched_ctx_list.h"

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#define NO_RESIZE -1
#define REQ_RESIZE 0
#define DO_RESIZE 1

#define STARPU_GLOBAL_SCHED_CTX 0
#define STARPU_NMAXSMS 13
struct _starpu_sched_ctx
{
	/** id of the context used in user mode*/
	unsigned id;

	/** boolean indicating whether the scheduling_ctx will be considered for scheduling (1) or not (0)*/
	unsigned do_schedule;

	/** name of context */
	const char *name;

	/** policy of the context */
	struct starpu_sched_policy *sched_policy;

	/** data necessary for the policy */
	void *policy_data;

	/** pointer for application use */
	void *user_data;

	struct starpu_worker_collection *workers;

	/** we keep an initial sched which we never delete */
	unsigned is_initial_sched;

	/** wait for the tasks submitted to the context to be executed */
	struct _starpu_barrier_counter tasks_barrier;

	/** wait for the tasks ready of the context to be executed */
	struct _starpu_barrier_counter ready_tasks_barrier;

	/** amount of ready flops in a context */
	double ready_flops;

	/** Iteration number, as advertised by application */
	long iterations[2];
	int iteration_level;

	/*ready tasks that couldn't be pushed because the ctx has no workers*/
	struct starpu_task_list empty_ctx_tasks;

	/*ready tasks that couldn't be pushed because the the window of tasks was already full*/
	struct starpu_task_list waiting_tasks;

	/** min CPUs to execute*/
	int min_ncpus;

	/** max CPUs to execute*/
	int max_ncpus;

	/** min GPUs to execute*/
	int min_ngpus;

	/** max GPUs to execute*/
	int max_ngpus;

	/** in case we delete the context leave resources to the inheritor*/
	unsigned inheritor;

	/** indicates whether the application finished submitting tasks
	   to this context*/
	unsigned finished_submit;

        /** By default we have a binary type of priority: either a task is a priority
         * task (level 1) or it is not (level 0). */
     	int min_priority;
	int max_priority;
     	int min_priority_is_set;
	int max_priority_is_set;

	/** hwloc tree structure of workers */
#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t hwloc_workers_set;
#endif

#ifdef STARPU_USE_SC_HYPERVISOR
	/** a structure containing a series of performance counters determining the resize procedure */
	struct starpu_sched_ctx_performance_counters *perf_counters;
#endif //STARPU_USE_SC_HYPERVISOR

	/** callback called when the context finished executed its submitted tasks */
	void (*close_callback)(unsigned sched_ctx_id, void* args);
	void *close_args;

	/** value placing the contexts in their hierarchy */
	unsigned hierarchy_level;

	/** if we execute non-StarPU code inside the context
	   we have a single master worker that stays awake,
	   if not master is -1 */
	int main_master;

	/** ctx nesting the current ctx */
	unsigned nesting_sched_ctx;

	/** perf model for the device comb of the ctx */
	struct starpu_perfmodel_arch perf_arch;

	/** For parallel workers, say whether it is viewed as sequential or not. This
		 is a helper for the prologue code. */
	unsigned parallel_view;

	/** for ctxs without policy: flag to indicate that we want to get
	   the threads to sleep in order to replace them with other threads or leave
	   them awake & use them in the parallel code*/
	unsigned awake_workers;

	/** function called when initializing the scheduler */
	void (*init_sched)(unsigned);

	int sub_ctxs[STARPU_NMAXWORKERS];
	int nsub_ctxs;

	/** nr of SMs assigned to this ctx if we partition gpus*/
	int nsms;
	int sms_start_idx;
	int sms_end_idx;

	int stream_worker;

	starpu_pthread_rwlock_t rwlock;
	starpu_pthread_t lock_write_owner;
};

/** per-worker list of deferred ctx_change ops */
LIST_TYPE(_starpu_ctx_change,
	int sched_ctx_id;
	int op;
	int nworkers_to_notify;
	int *workerids_to_notify;
	int nworkers_to_change;
	int *workerids_to_change;
);

struct _starpu_machine_config;

/** init sched_ctx_id of all contextes*/
void _starpu_init_all_sched_ctxs(struct _starpu_machine_config *config);

/** allocate all structures belonging to a context */
struct _starpu_sched_ctx*  _starpu_create_sched_ctx(struct starpu_sched_policy *policy, int *workerid, int nworkerids, unsigned is_init_sched, const char *sched_name,
						    int min_prio_set, int min_prio,
						    int max_prio_set, int max_prio, unsigned awake_workers, void (*sched_policy_init)(unsigned), void *user_data,
							int nsub_ctxs, int *sub_ctxs, int nsms);

/** delete all sched_ctx */
void _starpu_delete_all_sched_ctxs();

/** This function waits until all the tasks that were already submitted to a specific
 * context have been executed. */
int _starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id);

/** This function waits until at most n tasks are still submitted. */
int _starpu_wait_for_n_submitted_tasks_of_sched_ctx(unsigned sched_ctx_id, unsigned n);

/** In order to implement starpu_wait_for_all_tasks_of_ctx, we keep track of the number of
 * task currently submitted to the context */
void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);
void _starpu_increment_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);
int _starpu_get_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);
int _starpu_check_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);

void _starpu_decrement_nready_tasks_of_sched_ctx(unsigned sched_ctx_id, double ready_flops);
unsigned _starpu_increment_nready_tasks_of_sched_ctx(unsigned sched_ctx_id, double ready_flops, struct starpu_task *task);
int _starpu_wait_for_no_ready_of_sched_ctx(unsigned sched_ctx_id);

/** Return the corresponding index of the workerid in the ctx table */
int _starpu_get_index_in_ctx_of_workerid(unsigned sched_ctx, unsigned workerid);

/** Get the mutex corresponding to the global workerid */
starpu_pthread_mutex_t *_starpu_get_sched_mutex(struct _starpu_sched_ctx *sched_ctx, int worker);

/** Get workers belonging to a certain context, it returns the number of workers
 take care: no mutex taken, the list of workers might not be updated */
int _starpu_get_workers_of_sched_ctx(unsigned sched_ctx_id, int *pus, enum starpu_worker_archtype arch);

/** Let the worker know it does not belong to the context and that
   it should stop poping from it */
void _starpu_worker_gets_out_of_ctx(unsigned sched_ctx_id, struct _starpu_worker *worker);

/** Check if the worker belongs to another sched_ctx */
unsigned _starpu_worker_belongs_to_a_sched_ctx(int workerid, unsigned sched_ctx_id);

/** indicates wheather this worker should go to sleep or not
   (if it is the last one awake in a context he should better keep awake) */
unsigned _starpu_sched_ctx_last_worker_awake(struct _starpu_worker *worker);

/** If starpu_sched_ctx_set_context() has been called, returns the context
 * id set by its last call, or the id of the initial context */
unsigned _starpu_sched_ctx_get_current_context();

/** verify that some worker can execute a certain task */
int _starpu_workers_able_to_execute_task(struct starpu_task *task, struct _starpu_sched_ctx *sched_ctx);

void _starpu_fetch_tasks_from_empty_ctx_list(struct _starpu_sched_ctx *sched_ctx);

unsigned _starpu_sched_ctx_allow_hypervisor(unsigned sched_ctx_id);

struct starpu_perfmodel_arch * _starpu_sched_ctx_get_perf_archtype(unsigned sched_ctx);
#ifdef STARPU_USE_SC_HYPERVISOR
/** Notifies the hypervisor that a tasks was poped from the workers' list */
void _starpu_sched_ctx_post_exec_task_cb(int workerid, struct starpu_task *task, size_t data_size, uint32_t footprint);

#endif //STARPU_USE_SC_HYPERVISOR

void starpu_sched_ctx_add_combined_workers(int *combined_workers_to_add, unsigned n_combined_workers_to_add, unsigned sched_ctx_id);

/** if the worker is the master of a parallel context, and the job is meant to be executed on this parallel context, return a pointer to the context */
struct _starpu_sched_ctx *__starpu_sched_ctx_get_sched_ctx_for_worker_and_job(struct _starpu_worker *worker, struct _starpu_job *j);

#define _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(w,j) \
	(_starpu_get_nsched_ctxs() <= 1 ? _starpu_get_sched_ctx_struct(0) : __starpu_sched_ctx_get_sched_ctx_for_worker_and_job((w),(j)))

static inline struct _starpu_sched_ctx *_starpu_get_sched_ctx_struct(unsigned id);

static inline int _starpu_sched_ctx_check_write_locked(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return starpu_pthread_equal(sched_ctx->lock_write_owner, starpu_pthread_self());
}
#define STARPU_SCHED_CTX_CHECK_LOCK(sched_ctx_id) STARPU_ASSERT(_starpu_sched_ctx_check_write_locked((sched_ctx_id)))

static inline void _starpu_sched_ctx_lock_write(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_HG_DISABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_ASSERT(!starpu_pthread_equal(sched_ctx->lock_write_owner, starpu_pthread_self()));
	STARPU_HG_ENABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_PTHREAD_RWLOCK_WRLOCK(&sched_ctx->rwlock);
	sched_ctx->lock_write_owner = starpu_pthread_self();
}

static inline void _starpu_sched_ctx_unlock_write(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_HG_DISABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_ASSERT(starpu_pthread_equal(sched_ctx->lock_write_owner, starpu_pthread_self()));
	memset(&sched_ctx->lock_write_owner, 0, sizeof(sched_ctx->lock_write_owner));
	STARPU_HG_ENABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&sched_ctx->rwlock);
}

static inline void _starpu_sched_ctx_lock_read(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_HG_DISABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_ASSERT(!starpu_pthread_equal(sched_ctx->lock_write_owner, starpu_pthread_self()));
	STARPU_HG_ENABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&sched_ctx->rwlock);
}

static inline void _starpu_sched_ctx_unlock_read(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	STARPU_HG_DISABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_ASSERT(!starpu_pthread_equal(sched_ctx->lock_write_owner, starpu_pthread_self()));
	STARPU_HG_ENABLE_CHECKING(sched_ctx->lock_write_owner);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&sched_ctx->rwlock);
}

static inline unsigned _starpu_sched_ctx_worker_is_master_for_child_ctx(unsigned sched_ctx_id, unsigned workerid, struct starpu_task *task)
{
	unsigned child_sched_ctx = starpu_sched_ctx_worker_is_master_for_child_ctx(workerid, sched_ctx_id);
	if(child_sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{
		starpu_sched_ctx_move_task_to_ctx_locked(task, child_sched_ctx, 1);
		starpu_sched_ctx_revert_task_counters_ctx_locked(sched_ctx_id, task->flops);
		return 1;
	}
	return 0;
}

/** Go through the list of deferred ctx changes of the current worker and apply
 * any ctx change operation found until the list is empty */
void _starpu_worker_apply_deferred_ctx_changes(void);
#endif // __SCHED_CONTEXT_H__
