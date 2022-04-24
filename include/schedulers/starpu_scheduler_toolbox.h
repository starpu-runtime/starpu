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
extern "C"
{
#endif

/**
   @defgroup API_Scheduler_Toolbox Scheduler Toolbox
   @brief This is the interface for the scheduler toolbox
   @{
 */

/** structure for the FIFO task queue */
struct starpu_st_fifo_taskq;

/** Create a FIFO task queue */
struct starpu_st_fifo_taskq* starpu_st_fifo_taskq_create(void) STARPU_ATTRIBUTE_MALLOC;
void starpu_st_fifo_taskq_init(struct starpu_st_fifo_taskq *fifo);
void starpu_st_fifo_taskq_destroy(struct starpu_st_fifo_taskq *fifo);
int starpu_st_fifo_taskq_empty(struct starpu_st_fifo_taskq *fifo);
double starpu_st_fifo_taskq_get_exp_len_prev_task_list(struct starpu_st_fifo_taskq *fifo_queue, struct starpu_task *task, int workerid, int nimpl, int *fifo_ntasks);

int starpu_st_fifo_taskq_push_sorted_task(struct starpu_st_fifo_taskq *fifo_queue, struct starpu_task *task);
int starpu_st_fifo_taskq_push_task(struct starpu_st_fifo_taskq *fifo, struct starpu_task *task);
int starpu_st_fifo_taskq_push_back_task(struct starpu_st_fifo_taskq *fifo_queue, struct starpu_task *task);

int starpu_st_fifo_taskq_pop_this_task(struct starpu_st_fifo_taskq *fifo_queue, int workerid, struct starpu_task *task);
struct starpu_task *starpu_st_fifo_taskq_pop_task(struct starpu_st_fifo_taskq *fifo, int workerid);
/**
   This is the same as starpu_st_fifo_taskq_pop_task(), but without checking that the
   worker will be able to execute this task. This is useful when the scheduler
   has already checked it.
*/
struct starpu_task *starpu_st_fifo_taskq_pop_local_task(struct starpu_st_fifo_taskq *fifo);

/**
   Pop every task that can be executed on the calling driver
*/
struct starpu_task *starpu_st_fifo_taskq_pop_every_task(struct starpu_st_fifo_taskq *fifo, int workerid);
struct starpu_task *starpu_st_fifo_taskq_pop_first_ready_task(struct starpu_st_fifo_taskq *fifo_queue, unsigned workerid, int num_priorities);

/** all _starpu_prio_deque_pop/deque_task function return a task or a NULL pointer if none are available
 * in O(lg(nb priorities))
 */
struct starpu_st_prio_deque;
void starpu_st_prio_deque_init(struct starpu_st_prio_deque *pdeque);
void starpu_st_prio_deque_destroy(struct starpu_st_prio_deque *pdeque);
/** return 0 iff the struct starpu_st_prio_deque is not empty */
int starpu_st_prio_deque_is_empty(struct starpu_st_prio_deque *pdeque);

int starpu_st_prio_deque_push_back_task(struct starpu_st_prio_deque *pdeque, struct starpu_task *task);
/** push a task in O(lg(nb priorities)) */
int starpu_st_prio_deque_push_front_task(struct starpu_st_prio_deque *pdeque, struct starpu_task *task);

/** deque a task of the higher priority available from the front of the list for the highest priority */
struct starpu_task *starpu_st_prio_deque_pop_task_for_worker(struct starpu_st_prio_deque *, int workerid, struct starpu_task * *skipped);
/** return a task that can be executed by workerid from the back of the list for the highest priority */
struct starpu_task *starpu_st_prio_deque_deque_task_for_worker(struct starpu_st_prio_deque *, int workerid, struct starpu_task * *skipped);
struct starpu_task *starpu_st_prio_deque_deque_first_ready_task(struct starpu_st_prio_deque *, unsigned workerid);

struct starpu_task *starpu_st_prio_deque_pop_task(struct starpu_st_prio_deque *pdeque);
struct starpu_task *starpu_st_prio_deque_highest_task(struct starpu_st_prio_deque *pdeque);
struct starpu_task *starpu_st_prio_deque_pop_back_task(struct starpu_st_prio_deque *pdeque);
int starpu_st_prio_deque_pop_this_task(struct starpu_st_prio_deque *pdeque, int workerid, struct starpu_task *task);

void starpu_st_prio_deque_erase(struct starpu_st_prio_deque *pdeque, struct starpu_task *task);

int starpu_st_normalize_prio(int priority, int num_priorities, unsigned sched_ctx_id);
int starpu_st_non_ready_buffers_count(struct starpu_task *task, unsigned worker);
void starpu_st_non_ready_buffers_size(struct starpu_task *task, unsigned worker, size_t *non_readyp, size_t *non_loadingp, size_t *non_allocatedp);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHEDULER_TOOLBOX_FIFO_QUEUES_H__ */

