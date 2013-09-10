/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010 - 2012  INRIA
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



struct starpu_sched_ctx_performance_counters
{
	void (*notify_idle_cycle)(unsigned sched_ctx_id, int worker, double idle_time);
	void (*notify_idle_end)(unsigned sched_ctx_id, int worker);
	void (*notify_pushed_task)(unsigned sched_ctx_id, int worker);
	void (*notify_poped_task)(unsigned sched_ctx_id, int worker, struct starpu_task *task, size_t data_size, uint32_t footprint);
	void (*notify_post_exec_hook)(unsigned sched_ctx_id, int taskid);
	void (*notify_submitted_job)(struct starpu_task *task, uint32_t footprint, size_t data_size);
	void (*notify_ready_task)(unsigned sched_ctx_id, struct starpu_task *task);
	void (*notify_empty_ctx)(unsigned sched_ctx_id, struct starpu_task *task);
	void (*notify_delete_context)(unsigned sched_ctx);
};

#ifdef STARPU_USE_SC_HYPERVISOR
void starpu_sched_ctx_set_perf_counters(unsigned sched_ctx_id, void* perf_counters);
#endif //STARPU_USE_SC_HYPERVISOR

void starpu_sched_ctx_notify_hypervisor_exists(void);

unsigned starpu_sched_ctx_check_if_hypervisor_exists(void);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_HYPERVISOR_H__ */
