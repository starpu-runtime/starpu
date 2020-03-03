/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SCHED_CTX_HYPERVISOR_H__
#define __STARPU_SCHED_CTX_HYPERVISOR_H__

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @ingroup API_Scheduling_Contexts
   @{
*/

/**
   @name Scheduling Context Link with Hypervisor
   @{
*/

/**
   Performance counters used by the starpu to indicate the hypervisor
   how the application and the resources are executing.
 */
struct starpu_sched_ctx_performance_counters
{
	/**
	   Inform the hypervisor for how long a worker has been idle
	   in the specified context
	*/
	void (*notify_idle_cycle)(unsigned sched_ctx_id, int worker, double idle_time);

	/**
	   Inform the hypervisor that a task executing a specified
	   number of instructions has been poped from the worker
	*/
	void (*notify_poped_task)(unsigned sched_ctx_id, int worker);

	/**
	   Notify the hypervisor that a task has been scheduled on
	   the queue of the worker corresponding to the specified
	   context
	*/
	void (*notify_pushed_task)(unsigned sched_ctx_id, int worker);

	/**
	   Notify the hypervisor that a task has just been executed
	*/
	void (*notify_post_exec_task)(struct starpu_task *task, size_t data_size, uint32_t footprint, int hypervisor_tag, double flops);

	/**
	   Notify the hypervisor that a task has just been submitted
	*/
	void (*notify_submitted_job)(struct starpu_task *task, uint32_t footprint, size_t data_size);

	void (*notify_empty_ctx)(unsigned sched_ctx_id, struct starpu_task *task);

	/**
	   Notify the hypervisor that the context was deleted
	*/
	void (*notify_delete_context)(unsigned sched_ctx);
};

/**
   Indicate to starpu the pointer to the performance counter
*/
void starpu_sched_ctx_set_perf_counters(unsigned sched_ctx_id, void *perf_counters);

/**
   Callback that lets the scheduling policy tell the hypervisor that a
   task was pushed on a worker
*/
void starpu_sched_ctx_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);

/**
   Allow the hypervisor to let starpu know it's initialised
*/
void starpu_sched_ctx_notify_hypervisor_exists(void);

/**
   Ask starpu if it is informed if the hypervisor is initialised
*/
unsigned starpu_sched_ctx_check_if_hypervisor_exists(void);

void starpu_sched_ctx_update_start_resizing_sample(unsigned sched_ctx_id, double start_sample);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_HYPERVISOR_H__ */
