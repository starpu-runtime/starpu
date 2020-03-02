/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __STARPU_SCHEDULER_H__
#define __STARPU_SCHEDULER_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Scheduling_Policy Scheduling Policy
   @brief TODO. While StarPU comes with a variety of scheduling
   policies (see \ref TaskSchedulingPolicy), it may sometimes be
   desirable to implement custom policies to address specific
   problems. The API described below allows users to write their own
   scheduling policy.
   @{
*/

struct starpu_task;

/**
   Contain all the methods that implement a scheduling policy. An
   application may specify which scheduling strategy in the field
   starpu_conf::sched_policy passed to the function starpu_init().

   For each task going through the scheduler, the following methods
   get called in the given order:

   <ul>
   <li>starpu_sched_policy::submit_hook when the task is
   submitted</li>
   <li>starpu_sched_policy::push_task when the task becomes ready. The
   scheduler is here <b>given</b> the task</li>
   <li>starpu_sched_policy::pop_task when the worker is idle. The
   scheduler here <b>gives</b> back the task to the core. It must not
   access this task any more</li>
   <li>starpu_sched_policy::pre_exec_hook right before the worker
   actually starts the task computation (after transferring any
   missing data).</li>
   <li>starpu_sched_policy::post_exec_hook right after the worker
   actually completes the task computation.</li>
   </ul>

   For each task not going through the scheduler (because
   starpu_task::execute_on_a_specific_worker was set), these get
   called:

   <ul>
   <li>starpu_sched_policy::submit_hook when the task is
   submitted</li>
   <li>starpu_sched_policy::push_task_notify when the task becomes
   ready. This is just a notification, the scheduler does not have to
   do anything about the task.</li>
   <li>starpu_sched_policy::pre_exec_hook right before the worker
   actually starts the task computation (after transferring any
   missing data).</li>
   <li>starpu_sched_policy::post_exec_hook right after the worker
   actually completes the task computation.</li>
   </ul>
*/
struct starpu_sched_policy
{
	/**
	   Initialize the scheduling policy, called before any other
	   method.
	*/
	void (*init_sched)(unsigned sched_ctx_id);
	/**
	   Cleanup the scheduling policy
	*/
	void (*deinit_sched)(unsigned sched_ctx_id);

	/**
	   Insert a task into the scheduler, called when the task
	   becomes ready for execution. This must call
	   starpu_push_task_end() once it has effectively pushed the
	   task to a queue (to note the time when this was done in the
	   task), but before releasing mutexes (so that the task
	   hasn't been already taken by a worker).
	*/
	int (*push_task)(struct starpu_task *);

	double (*simulate_push_task)(struct starpu_task *);

	/**
	   Notify the scheduler that a task was pushed on a given
	   worker. This method is called when a task that was
	   explicitly assigned to a worker becomes ready and is about
	   to be executed by the worker. This method therefore permits
	   to keep the state of the scheduler coherent even when
	   StarPU bypasses the scheduling strategy.
	*/
	void (*push_task_notify)(struct starpu_task *, int workerid, int perf_workerid, unsigned sched_ctx_id);

	/**
	   Get a task from the scheduler.
	   If this method returns NULL, the worker will start
	   sleeping. If later on some task are pushed for this worker,
	   starpu_wake_worker() must be called to wake the worker so
	   it can call the pop_task() method again.
	   The mutex associated to the worker is already taken when
	   this method is called. This method may release it (e.g. for
	   scalability reasons when doing work stealing), but it must
	   acquire it again before taking the decision whether to
	   return a task or NULL, so the atomicity of deciding to
	   return NULL and making the worker actually sleep is
	   preserved. Otherwise in simgrid or blocking driver mode the
	   worker might start sleeping while a task has just been
	   pushed for it.
	   If this method is defined as <c>NULL</c>, the worker will
	   only execute tasks from its local queue. In this case, the
	   push_task method should use the starpu_push_local_task
	   method to assign tasks to the different workers.
	*/
	struct starpu_task *(*pop_task)(unsigned sched_ctx_id);

	/**
	   Remove all available tasks from the scheduler (tasks are
	   chained by the means of the field starpu_task::prev and
	   starpu_task::next). The mutex associated to the worker is
	   already taken when this method is called. This is currently
	   not used and can be discarded.
	*/
	struct starpu_task *(*pop_every_task)(unsigned sched_ctx_id);

	/**
	   Optional field. This method is called when a task is
	   submitted.
	*/
	void (*submit_hook)(struct starpu_task *task);

	/**
	   Optional field. This method is called every time a task is
	   starting.
	*/
	void (*pre_exec_hook)(struct starpu_task *, unsigned sched_ctx_id);

	/**
	   Optional field. This method is called every time a task has
	   been executed.
	*/
	void (*post_exec_hook)(struct starpu_task *, unsigned sched_ctx_id);

	/**
	   Optional field. This method is called when it is a good
	   time to start scheduling tasks. This is notably called when
	   the application calls starpu_task_wait_for_all() or
	   starpu_do_schedule() explicitly.
	*/
	void (*do_schedule)(unsigned sched_ctx_id);

	/**
	   Initialize scheduling structures corresponding to each
	   worker used by the policy.
	*/
	void (*add_workers)(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

	/**
	   Deinitialize scheduling structures corresponding to each
	   worker used by the policy.
	*/
	void (*remove_workers)(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

	/**
	   Optional field. Name of the policy.
	*/
	const char *policy_name;

	/**
	   Optional field. Human readable description of the policy.
	*/
	const char *policy_description;

	enum starpu_worker_collection_type worker_type;
};

/**
   Return an <c>NULL</c>-terminated array of all the predefined
   scheduling policies.
*/
struct starpu_sched_policy **starpu_sched_get_predefined_policies();

/**
   When there is no available task for a worker, StarPU blocks this
   worker on a condition variable. This function specifies which
   condition variable (and the associated mutex) should be used to
   block (and to wake up) a worker. Note that multiple workers may use
   the same condition variable. For instance, in the case of a
   scheduling strategy with a single task queue, the same condition
   variable would be used to block and wake up all workers.
*/
void starpu_worker_get_sched_condition(int workerid, starpu_pthread_mutex_t **sched_mutex, starpu_pthread_cond_t **sched_cond);

unsigned long starpu_task_get_job_id(struct starpu_task *task);

/**
   TODO: check if this is correct
   Return the current minimum priority level supported by the scheduling
   policy
*/
int starpu_sched_get_min_priority(void);

/**
   TODO: check if this is correct
   Return the current maximum priority level supported by the
   scheduling policy
*/
int starpu_sched_get_max_priority(void);

/**
   TODO: check if this is correct
   Define the minimum task priority level supported by the scheduling
   policy. The default minimum priority level is the same as the
   default priority level which is 0 by convention. The application
   may access that value by calling the function
   starpu_sched_get_min_priority(). This function should only be
   called from the initialization method of the scheduling policy, and
   should not be used directly from the application.
*/
int starpu_sched_set_min_priority(int min_prio);

/**
   TODO: check if this is correct
   Define the maximum priority level supported by the scheduling
   policy. The default maximum priority level is 1. The application
   may access that value by calling the function
   starpu_sched_get_max_priority(). This function should only be
   called from the initialization method of the scheduling policy, and
   should not be used directly from the application.
*/
int starpu_sched_set_max_priority(int max_prio);

/**
   Check if the worker specified by workerid can execute the codelet.
   Schedulers need to call it before assigning a task to a worker,
   otherwise the task may fail to execute.
*/
int starpu_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

/**
   Check if the worker specified by workerid can execute the codelet
   and return which implementation numbers can be used.
   Schedulers need to call it before assigning a task to a worker,
   otherwise the task may fail to execute.
   This should be preferred rather than calling
   starpu_worker_can_execute_task() for each and every implementation.
   It can also be used with <c>impl_mask == NULL</c> to check for at
   least one implementation without determining which.
*/
int starpu_worker_can_execute_task_impl(unsigned workerid, struct starpu_task *task, unsigned *impl_mask);

/**
   Check if the worker specified by workerid can execute the codelet
   and return the first implementation which can be used.
   Schedulers need to call it before assigning a task to a worker,
   otherwise the task may fail to execute. This should be preferred
   rather than calling starpu_worker_can_execute_task() for
   each and every implementation. It can also be used with
   <c>impl_mask == NULL</c> to check for at least one implementation
   without determining which.
*/
int starpu_worker_can_execute_task_first_impl(unsigned workerid, struct starpu_task *task, unsigned *nimpl);

/**
   The scheduling policy may put tasks directly into a worker’s local
   queue so that it is not always necessary to create its own queue
   when the local queue is sufficient. If \p back is not 0, \p task is
   put at the back of the queue where the worker will pop tasks first.
   Setting \p back to 0 therefore ensures a FIFO ordering.
*/
int starpu_push_local_task(int workerid, struct starpu_task *task, int back);

/**
   Must be called by a scheduler to notify that the given
   task has just been pushed.
*/
int starpu_push_task_end(struct starpu_task *task);

/**
   Whether \ref STARPU_PREFETCH was set
*/
int starpu_get_prefetch_flag(void);

/**
   Prefetch data for a given p task on a given p node with a given
   priority
*/
int starpu_prefetch_task_input_on_node_prio(struct starpu_task *task, unsigned node, int prio);

/**
   Prefetch data for a given p task on a given p node
*/
int starpu_prefetch_task_input_on_node(struct starpu_task *task, unsigned node);

/**
   Prefetch data for a given p task on a given p node when the bus is
   idle with a given priority
*/
int starpu_idle_prefetch_task_input_on_node_prio(struct starpu_task *task, unsigned node, int prio);

/**
   Prefetch data for a given p task on a given p node when the bus is
   idle
*/
int starpu_idle_prefetch_task_input_on_node(struct starpu_task *task, unsigned node);

/**
   Prefetch data for a given p task on a given p worker with a given
   priority
*/
int starpu_prefetch_task_input_for_prio(struct starpu_task *task, unsigned worker, int prio);

/**
   Prefetch data for a given p task on a given p worker
*/
int starpu_prefetch_task_input_for(struct starpu_task *task, unsigned worker);

/**
   Prefetch data for a given p task on a given p worker when the bus
   is idle with a given priority
*/
int starpu_idle_prefetch_task_input_for_prio(struct starpu_task *task, unsigned worker, int prio);

/**
   Prefetch data for a given p task on a given p worker when the bus
   is idle
*/
int starpu_idle_prefetch_task_input_for(struct starpu_task *task, unsigned worker);

/**
   Return the footprint for a given task, taking into account
   user-provided perfmodel footprint or size_base functions.
*/
uint32_t starpu_task_footprint(struct starpu_perfmodel *model, struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned nimpl);

/**
   Return the raw footprint for the data of a given task (without
   taking into account user-provided functions).
*/
uint32_t starpu_task_data_footprint(struct starpu_task *task);

/**
   Return expected task duration in micro-seconds.
*/
double starpu_task_expected_length(struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned nimpl);

/**
   Return an estimated speedup factor relative to CPU speed
*/
double starpu_worker_get_relative_speedup(struct starpu_perfmodel_arch *perf_arch);

/**
   Return expected data transfer time in micro-seconds for the given \p
   memory_node. Prefer using starpu_task_expected_data_transfer_time_for() which is
   more precise.
*/
double starpu_task_expected_data_transfer_time(unsigned memory_node, struct starpu_task *task);

/**
   Return expected data transfer time in micro-seconds for the given
   \p worker.
*/
double starpu_task_expected_data_transfer_time_for(struct starpu_task *task, unsigned worker);

/**
   Predict the transfer time (in micro-seconds) to move \p handle to a
   memory node
*/
double starpu_data_expected_transfer_time(starpu_data_handle_t handle, unsigned memory_node, enum starpu_data_access_mode mode);

/**
   Return expected energy consumption in J
*/
double starpu_task_expected_energy(struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned nimpl);

/**
   Return expected conversion time in ms (multiformat interface only)
*/
double starpu_task_expected_conversion_time(struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned nimpl);

typedef void (*starpu_notify_ready_soon_func)(void *data, struct starpu_task *task, double delay);

/**
   Register a callback to be called when it is determined when a task
   will be ready an estimated amount of time from now, because its
   last dependency has just started and we know how long it will take.
*/
void starpu_task_notify_ready_soon_register(starpu_notify_ready_soon_func f, void *data);

/**
   The scheduling policies indicates if the worker may pop tasks from
   the list of other workers or if there is a central list with task
   for all the workers
*/
void starpu_sched_ctx_worker_shares_tasks_lists(int workerid, int sched_ctx_id);

void starpu_sched_task_break(struct starpu_task *task);

/**
   @name Worker operations
   @{
*/

/**
   Wake up \p workerid while temporarily entering the current worker
   relax state if needed during the waiting process. Return 1 if \p
   workerid has been woken up or its state_keep_awake flag has been
   set to \c 1, and \c 0 otherwise (if \p workerid was not in the
   STATE_SLEEPING or in the STATE_SCHEDULING).
*/
int starpu_wake_worker_relax(int workerid);

/**
   Must be called to wake up a worker that is sleeping on the cond.
   Return 0 whenever the worker is not in a sleeping state or has the
   state_keep_awake flag on.
*/
int starpu_wake_worker_no_relax(int workerid);

/**
   Version of starpu_wake_worker_no_relax() which assumes that the
   sched mutex is locked
*/
int starpu_wake_worker_locked(int workerid);

/**
   Light version of starpu_wake_worker_relax() which, when possible,
   speculatively set keep_awake on the target worker without waiting
   for the worker to enter the relax state.
*/
int starpu_wake_worker_relax_light(int workerid);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHEDULER_H__ */
