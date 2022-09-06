/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SCHEDULER_TOOLBOX_FIFO_QUEUES_H__
#define __STARPU_SCHEDULER_TOOLBOX_FIFO_QUEUES_H__

#include <starpu.h>
#include <starpu_scheduler.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Scheduler_Toolbox Scheduler Toolbox
   @brief This is the interface for the scheduler toolbox

   The definitions of the different queue types below (e.g
   ::starpu_st_fifo_taskq_t) are private and are thus not available
   outside the StarPU source directory. Hence when defining your own
   scheduler outside of StarPU source directory, you should use the
   functions below. Look for example in the scheduler defined in
   <c>examples/cholesky/libmy_dmda.c</c>
   @{
 */

/**
   Opaque type for FIFO task queue
*/
typedef struct starpu_st_fifo_taskq *starpu_st_fifo_taskq_t;

/** Create a FIFO task queue */
starpu_st_fifo_taskq_t starpu_st_fifo_taskq_create(void) STARPU_ATTRIBUTE_MALLOC;
void starpu_st_fifo_taskq_init(starpu_st_fifo_taskq_t fifo);
void starpu_st_fifo_taskq_destroy(starpu_st_fifo_taskq_t fifo);
int starpu_st_fifo_taskq_empty(starpu_st_fifo_taskq_t fifo);
double starpu_st_fifo_taskq_get_exp_len_prev_task_list(starpu_st_fifo_taskq_t fifo_queue, struct starpu_task *task, int workerid, int nimpl, int *fifo_ntasks);

/** get the number of tasks currently in the queue */
unsigned starpu_st_fifo_ntasks_get(starpu_st_fifo_taskq_t fifo);

/** increase by n the number of tasks currently in the queue */
void starpu_st_fifo_ntasks_inc(starpu_st_fifo_taskq_t fifo, int n);

/** get the number of tasks currently in the queue corresponding to each priority */
unsigned *starpu_st_fifo_ntasks_per_priority_get(starpu_st_fifo_taskq_t fifo);

/** get the number of tasks that were processed */
unsigned starpu_st_fifo_nprocessed_get(starpu_st_fifo_taskq_t fifo);

/** increase by n the number of tasks that were processed */
void starpu_st_fifo_nprocessed_inc(starpu_st_fifo_taskq_t fifo, int n);

/** only meaningful if the queue is only used by a single worker */
/**
   Get the expected start date of next item to do in the
   queue (i.e. not started yet). This is thus updated
   when we start it.
*/
double starpu_st_fifo_exp_start_get(starpu_st_fifo_taskq_t fifo);

/**
   Set the expected start date of next item to do in the
   queue (i.e. not started yet).
 */
void starpu_st_fifo_exp_start_set(starpu_st_fifo_taskq_t fifo, double exp_start);

/** get the expected end date of last task in the queue */
double starpu_st_fifo_exp_end_get(starpu_st_fifo_taskq_t fifo);

/** set the expected end date of last task in the queue */
void starpu_st_fifo_exp_end_set(starpu_st_fifo_taskq_t fifo, double exp_end);

/** get the expected duration of the set of tasks in the queue */
double starpu_st_fifo_exp_len_get(starpu_st_fifo_taskq_t fifo);

/** set the expected duration of the set of tasks in the queue */
void starpu_st_fifo_exp_len_set(starpu_st_fifo_taskq_t fifo, double exp_len);

/** increase or decrease the expected duration of the set of tasks in the queue */
void starpu_st_fifo_exp_len_inc(starpu_st_fifo_taskq_t fifo, double exp_len);

/** get the expected duration of the set of tasks in the queue corresponding to each priority */
double *starpu_st_fifo_exp_len_per_priority_get(starpu_st_fifo_taskq_t fifo);

/** get the expected duration of what is already pushed to the worker */
double starpu_st_fifo_pipeline_len_get(starpu_st_fifo_taskq_t fifo);

/** set the expected duration of what is already pushed to the worker */
void starpu_st_fifo_pipeline_len_set(starpu_st_fifo_taskq_t fifo, double pipeline_len);

/** increase the expected duration of what is already pushed to the worker (the value can be negative) */
void starpu_st_fifo_pipeline_len_inc(starpu_st_fifo_taskq_t fifo, double pipeline_len);

int starpu_st_fifo_taskq_push_sorted_task(starpu_st_fifo_taskq_t fifo_queue, struct starpu_task *task);
int starpu_st_fifo_taskq_push_task(starpu_st_fifo_taskq_t fifo, struct starpu_task *task);
int starpu_st_fifo_taskq_push_back_task(starpu_st_fifo_taskq_t fifo_queue, struct starpu_task *task);

int starpu_st_fifo_taskq_pop_this_task(starpu_st_fifo_taskq_t fifo_queue, int workerid, struct starpu_task *task);
struct starpu_task *starpu_st_fifo_taskq_pop_task(starpu_st_fifo_taskq_t fifo, int workerid);
/**
   This is the same as starpu_st_fifo_taskq_pop_task(), but without checking that the
   worker will be able to execute this task. This is useful when the scheduler
   has already checked it.
*/
struct starpu_task *starpu_st_fifo_taskq_pop_local_task(starpu_st_fifo_taskq_t fifo);

/**
   Pop every task that can be executed on the calling driver
*/
struct starpu_task *starpu_st_fifo_taskq_pop_every_task(starpu_st_fifo_taskq_t fifo, int workerid);
struct starpu_task *starpu_st_fifo_taskq_pop_first_ready_task(starpu_st_fifo_taskq_t fifo_queue, unsigned workerid, int num_priorities);

/**
   Opaque type for PRIO task queue
*/
typedef struct starpu_st_prio_deque *starpu_st_prio_deque_t;

/** all _starpu_prio_deque_pop/deque_task function return a task or a NULL pointer if none are available
 * in O(lg(nb priorities))
 */
void starpu_st_prio_deque_init(starpu_st_prio_deque_t pdeque);
void starpu_st_prio_deque_destroy(starpu_st_prio_deque_t pdeque);
/** return 0 iff the struct starpu_st_prio_deque is not empty */
int starpu_st_prio_deque_is_empty(starpu_st_prio_deque_t pdeque);

int starpu_st_prio_deque_push_back_task(starpu_st_prio_deque_t pdeque, struct starpu_task *task);
/** push a task in O(lg(nb priorities)) */
int starpu_st_prio_deque_push_front_task(starpu_st_prio_deque_t pdeque, struct starpu_task *task);

/** deque a task of the higher priority available from the front of the list for the highest priority */
struct starpu_task *starpu_st_prio_deque_pop_task_for_worker(starpu_st_prio_deque_t pdeque, int workerid, struct starpu_task **skipped);
/** return a task that can be executed by workerid from the back of the list for the highest priority */
struct starpu_task *starpu_st_prio_deque_deque_task_for_worker(starpu_st_prio_deque_t pdeque, int workerid, struct starpu_task **skipped);
struct starpu_task *starpu_st_prio_deque_deque_first_ready_task(starpu_st_prio_deque_t pdeque, unsigned workerid);

struct starpu_task *starpu_st_prio_deque_pop_task(starpu_st_prio_deque_t pdeque);
struct starpu_task *starpu_st_prio_deque_highest_task(starpu_st_prio_deque_t pdeque);
struct starpu_task *starpu_st_prio_deque_pop_back_task(starpu_st_prio_deque_t pdeque);
int starpu_st_prio_deque_pop_this_task(starpu_st_prio_deque_t pdeque, int workerid, struct starpu_task *task);

void starpu_st_prio_deque_erase(starpu_st_prio_deque_t pdeque, struct starpu_task *task);

int starpu_st_normalize_prio(int priority, int num_priorities, unsigned sched_ctx_id);
int starpu_st_non_ready_buffers_count(struct starpu_task *task, unsigned worker);
void starpu_st_non_ready_buffers_size(struct starpu_task *task, unsigned worker, size_t *non_readyp, size_t *non_loadingp, size_t *non_allocatedp);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHEDULER_TOOLBOX_FIFO_QUEUES_H__ */
