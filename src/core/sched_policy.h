/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <starpu.h>
#include <core/workers.h>
#include <core/sched_ctx.h>
#include <starpu_scheduler.h>

struct starpu_machine_config_s;
struct starpu_sched_policy_s *_starpu_get_sched_policy( struct starpu_sched_ctx *sched_ctx);

void _starpu_init_sched_policy(struct starpu_machine_config_s *config, 
			       struct starpu_sched_ctx *sched_ctx, const char *policy_name);

void _starpu_deinit_sched_policy(struct starpu_machine_config_s *config, struct starpu_sched_ctx *sched_ctx);

int _starpu_push_task(starpu_job_t task, unsigned job_is_already_locked);
/* pop a task that can be executed on the worker */
struct starpu_task *_starpu_pop_task(struct starpu_worker_s *worker);
/* pop every task that can be executed on the worker */
struct starpu_task *_starpu_pop_every_task(struct starpu_sched_ctx *sched_ctx);
void _starpu_sched_post_exec_hook(struct starpu_task *task);

void _starpu_wait_on_sched_event(void);

#endif // __SCHED_POLICY_H__
