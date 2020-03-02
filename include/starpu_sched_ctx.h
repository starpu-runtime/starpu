/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
 * Copyright (C) 2017       Arthur Chevalier
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

#ifndef __STARPU_SCHED_CTX_H__
#define __STARPU_SCHED_CTX_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Scheduling_Contexts Scheduling Contexts
   @brief StarPU permits on one hand grouping workers in combined
   workers in order to execute a parallel task and on the other hand
   grouping tasks in bundles that will be executed by a single
   specified worker.
   In contrast when we group workers in scheduling contexts we submit
   starpu tasks to them and we schedule them with the policy assigned
   to the context. Scheduling contexts can be created, deleted and
   modified dynamically.
   @{
*/

/**
   @name Scheduling Contexts Basic API
   @{
*/

/**
   Used when calling starpu_sched_ctx_create() to specify a
   name for a scheduling policy
*/
#define STARPU_SCHED_CTX_POLICY_NAME		 (1<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify a
   pointer to a scheduling policy
*/
#define STARPU_SCHED_CTX_POLICY_STRUCT		 (2<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify a
   minimum scheduler priority value.
*/
#define STARPU_SCHED_CTX_POLICY_MIN_PRIO	 (3<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify a
   maximum scheduler priority value.
*/
#define STARPU_SCHED_CTX_POLICY_MAX_PRIO	 (4<<16)

#define STARPU_SCHED_CTX_HIERARCHY_LEVEL         (5<<16)
#define STARPU_SCHED_CTX_NESTED                  (6<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify ???
*/
#define STARPU_SCHED_CTX_AWAKE_WORKERS           (7<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify a
   function pointer allowing to initialize the scheduling policy.
*/
#define STARPU_SCHED_CTX_POLICY_INIT             (8<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify a
   pointer to some user data related to the context being created.
*/
#define STARPU_SCHED_CTX_USER_DATA               (9<<16)

/**
   Used when calling starpu_sched_ctx_create() in order to create a
   context on the NVIDIA GPU to specify the number of SMs the context
   should have
*/
#define STARPU_SCHED_CTX_CUDA_NSMS               (10<<16)

/**
   Used when calling starpu_sched_ctx_create() to specify
   a list of sub contexts of the current context.
*/
#define STARPU_SCHED_CTX_SUB_CTXS                (11<<16)

/**
   Create a scheduling context with the given parameters
   (see below) and assign the workers in \p workerids_ctx to execute the
   tasks submitted to it. The return value represents the identifier of
   the context that has just been created. It will be further used to
   indicate the context the tasks will be submitted to. The return value
   should be at most ::STARPU_NMAX_SCHED_CTXS.

   The arguments following the name of the scheduling context can be of
   the following types:
   <ul>
   <li> ::STARPU_SCHED_CTX_POLICY_NAME, followed by the name of a
   predefined scheduling policy. Use an empty string to create the
   context with the default scheduling policy.
   </li>
   <li> ::STARPU_SCHED_CTX_POLICY_STRUCT, followed by a pointer to a
   custom scheduling policy (struct starpu_sched_policy *)
   </li>
   <li> ::STARPU_SCHED_CTX_POLICY_MIN_PRIO, followed by a integer
   representing the minimum priority value to be defined for the
   scheduling policy.
   </li>
   <li> ::STARPU_SCHED_CTX_POLICY_MAX_PRIO, followed by a integer
   representing the maximum priority value to be defined for the
   scheduling policy.
   </li>
   <li> ::STARPU_SCHED_CTX_POLICY_INIT, followed by a function pointer
   (ie. void init_sched(void)) allowing to initialize the scheduling policy.
   </li>
   <li> ::STARPU_SCHED_CTX_USER_DATA, followed by a pointer
   to a custom user data structure, to be retrieved by \ref starpu_sched_ctx_get_user_data().
   </li>
   </ul>
*/
unsigned starpu_sched_ctx_create(int *workerids_ctx, int nworkers_ctx, const char *sched_ctx_name, ...);

/**
   Create a context indicating an approximate interval of resources
*/
unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_ctx_name, int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus, unsigned allow_overlap);

/**
   Execute the callback whenever the last task of the context finished
   executing, it is called with the parameters \p sched_ctx and any
   other parameter needed by the application (packed in \p args)
*/
void starpu_sched_ctx_register_close_callback(unsigned sched_ctx_id, void (*close_callback)(unsigned sched_ctx_id, void* args), void *args);

/**
   Add dynamically the workers in \p workerids_ctx to the context \p
   sched_ctx_id. The last argument cannot be greater than
   ::STARPU_NMAX_SCHED_CTXS.
*/
void starpu_sched_ctx_add_workers(int *workerids_ctx, unsigned nworkers_ctx, unsigned sched_ctx_id);

/**
   Remove the workers in \p workerids_ctx from the context
   \p sched_ctx_id. The last argument cannot be greater than
   ::STARPU_NMAX_SCHED_CTXS.
*/
void starpu_sched_ctx_remove_workers(int *workerids_ctx, unsigned nworkers_ctx, unsigned sched_ctx_id);

/**
   Print on the file \p f the worker names belonging to the context \p
   sched_ctx_id
*/
void starpu_sched_ctx_display_workers(unsigned sched_ctx_id, FILE *f);

/**
   Delete scheduling context \p sched_ctx_id and transfer remaining
   workers to the inheritor scheduling context.
*/
void starpu_sched_ctx_delete(unsigned sched_ctx_id);

/**
   Indicate that the context \p inheritor will inherit the resources
   of the context \p sched_ctx_id when \p sched_ctx_id will be
   deleted.
*/
void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor);

unsigned starpu_sched_ctx_get_inheritor(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_hierarchy_level(unsigned sched_ctx_id);

/**
   Set the scheduling context the subsequent tasks will be submitted
   to
*/
void starpu_sched_ctx_set_context(unsigned *sched_ctx_id);

/**
   Return the scheduling context the tasks are currently submitted to,
   or ::STARPU_NMAX_SCHED_CTXS if no default context has been defined
   by calling the function starpu_sched_ctx_set_context().
*/
unsigned starpu_sched_ctx_get_context(void);

/**
   Stop submitting tasks from the empty context list until the next
   time the context has time to check the empty context list
*/
void starpu_sched_ctx_stop_task_submission(void);

/**
   Indicate starpu that the application finished submitting to this
   context in order to move the workers to the inheritor as soon as
   possible.
*/
void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);

/**
   Return the list of workers in the array \p workerids, the return
   value is the number of workers. The user should free the \p
   workerids table after finishing using it (it is allocated inside
   the function with the proper size)
*/
unsigned starpu_sched_ctx_get_workers_list(unsigned sched_ctx_id, int **workerids);

/**
   Return the list of workers in the array \p workerids, the return
   value is the number of workers. This list is provided in raw order,
   i.e. not sorted by tree or list order, and the user should not free
   the \p workerids table. This function is thus much less costly than
   starpu_sched_ctx_get_workers_list().
*/
unsigned starpu_sched_ctx_get_workers_list_raw(unsigned sched_ctx_id, int **workerids);

/**
   Return the number of workers managed by the specified context
   (Usually needed to verify if it manages any workers or if it should
   be blocked)
*/
unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id);

/**
   Return the number of workers shared by two contexts.
*/
unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);

/**
   Return 1 if the worker belongs to the context and 0 otherwise
*/
unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id);

unsigned starpu_sched_ctx_contains_type_of_worker(enum starpu_worker_archtype arch, unsigned sched_ctx_id);

/**
   Return the workerid if the worker belongs to the context and -1 otherwise.
   If the thread calling this function is not a worker the function returns -1
   as it calls the function starpu_worker_get_id().
*/
unsigned starpu_sched_ctx_worker_get_id(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_ctx_for_task(struct starpu_task *task);

/**
   Check if a worker is shared between several contexts
*/
unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid);

/**
   Return the user data pointer associated to the scheduling context.
*/
void *starpu_sched_ctx_get_user_data(unsigned sched_ctx_id);

void starpu_sched_ctx_set_user_data(unsigned sched_ctx_id, void* user_data);

/**
   Allocate the scheduling policy data (private information of the
   scheduler like queues, variables, additional condition variables)
   the context
*/
void starpu_sched_ctx_set_policy_data(unsigned sched_ctx_id, void *policy_data);

/**
   Return the scheduling policy data (private information of the
   scheduler) of the contexts previously assigned to.
*/
void *starpu_sched_ctx_get_policy_data(unsigned sched_ctx_id);

struct starpu_sched_policy *starpu_sched_ctx_get_sched_policy(unsigned sched_ctx_id);

/**
   Execute any parallel code on the workers of the sched_ctx (workers
   are blocked)
*/
void *starpu_sched_ctx_exec_parallel_code(void* (*func)(void*), void *param, unsigned sched_ctx_id);

int starpu_sched_ctx_get_nready_tasks(unsigned sched_ctx_id);

double starpu_sched_ctx_get_nready_flops(unsigned sched_ctx_id);

void starpu_sched_ctx_list_task_counters_increment(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_list_task_counters_decrement(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_list_task_counters_reset(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_list_task_counters_increment_all_ctx_locked(struct starpu_task *task, unsigned sched_ctx_id);

void starpu_sched_ctx_list_task_counters_decrement_all_ctx_locked(struct starpu_task *task, unsigned sched_ctx_id);

void starpu_sched_ctx_list_task_counters_reset_all(struct starpu_task *task, unsigned sched_ctx_id);

void starpu_sched_ctx_set_priority(int *workers, int nworkers, unsigned sched_ctx_id, unsigned priority);

unsigned starpu_sched_ctx_get_priority(int worker, unsigned sched_ctx_id);

void starpu_sched_ctx_get_available_cpuids(unsigned sched_ctx_id, int **cpuids, int *ncpuids);

void starpu_sched_ctx_bind_current_thread_to_cpuid(unsigned cpuid);

int starpu_sched_ctx_book_workers_for_task(unsigned sched_ctx_id, int *workerids, int nworkers);

void starpu_sched_ctx_unbook_workers_for_task(unsigned sched_ctx_id, int master);

/**
   Return the first context (child of sched_ctx_id) where the workerid
   is master
 */
unsigned starpu_sched_ctx_worker_is_master_for_child_ctx(int workerid, unsigned sched_ctx_id);

/**
   Return the context id of masterid if it master of a context. If
   not, return ::STARPU_NMAX_SCHED_CTXS.
*/
unsigned starpu_sched_ctx_master_get_context(int masterid);

void starpu_sched_ctx_revert_task_counters_ctx_locked(unsigned sched_ctx_id, double flops);

void starpu_sched_ctx_move_task_to_ctx_locked(struct starpu_task *task, unsigned sched_ctx, unsigned with_repush);

int starpu_sched_ctx_get_worker_rank(unsigned sched_ctx_id);

void (*starpu_sched_ctx_get_sched_policy_init(unsigned sched_ctx_id))(unsigned);

unsigned starpu_sched_ctx_has_starpu_scheduler(unsigned sched_ctx_id, unsigned *awake_workers);

int starpu_sched_ctx_get_stream_worker(unsigned sub_ctx);
int starpu_sched_ctx_get_nsms(unsigned sched_ctx);
void starpu_sched_ctx_get_sms_interval(int stream_workerid, int *start, int *end);

/** @} */

/**
   @name Scheduling Context Priorities
   @{
*/

/**
   Return the current minimum priority level supported by the
   scheduling policy of the given scheduler context.
*/
int starpu_sched_ctx_get_min_priority(unsigned sched_ctx_id);

/**
   Return the current maximum priority level supported by the
   scheduling policy of the given scheduler context.
*/
int starpu_sched_ctx_get_max_priority(unsigned sched_ctx_id);

/**
   Define the minimum task priority level supported by the scheduling
   policy of the given scheduler context. The default minimum priority
   level is the same as the default priority level which is 0 by
   convention. The application may access that value by calling the
   function starpu_sched_ctx_get_min_priority(). This function should
   only be called from the initialization method of the scheduling
   policy, and should not be used directly from the application.
*/
int starpu_sched_ctx_set_min_priority(unsigned sched_ctx_id, int min_prio);

/**
   Define the maximum priority level supported by the scheduling
   policy of the given scheduler context. The default maximum priority
   level is 1. The application may access that value by calling the
   starpu_sched_ctx_get_max_priority() function. This function should
   only be called from the initialization method of the scheduling
   policy, and should not be used directly from the application.
*/
int starpu_sched_ctx_set_max_priority(unsigned sched_ctx_id, int max_prio);

int starpu_sched_ctx_min_priority_is_set(unsigned sched_ctx_id);

int starpu_sched_ctx_max_priority_is_set(unsigned sched_ctx_id);

/**
   Provided for legacy reasons.
*/
#define STARPU_MIN_PRIO		(starpu_sched_get_min_priority())

/**
   Provided for legacy reasons.
*/
#define STARPU_MAX_PRIO		(starpu_sched_get_max_priority())

/**
   By convention, the default priority level should be 0 so that we
   can statically allocate tasks with a default priority.
*/
#define STARPU_DEFAULT_PRIO	0

/** @} */

/**
   @name Scheduling Context Worker Collection
   @{
*/

/**
   Create a worker collection of the type indicated by the last
   parameter for the context specified through the first parameter.
*/
struct starpu_worker_collection *starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, enum starpu_worker_collection_type type) STARPU_ATTRIBUTE_MALLOC;

/**
   Delete the worker collection of the specified scheduling context
*/
void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id);

/**
   Return the worker collection managed by the indicated context
*/
struct starpu_worker_collection *starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_H__ */
