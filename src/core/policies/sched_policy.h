/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <starpu.h>
#include <core/workers.h>

#include <starpu_scheduler.h>

struct starpu_machine_config_s;
struct starpu_sched_policy_s *_starpu_get_sched_policy(void);

void _starpu_init_sched_policy(struct starpu_machine_config_s *config);
void _starpu_deinit_sched_policy(struct starpu_machine_config_s *config);

int _starpu_get_prefetch_flag(void);

int _starpu_push_task(starpu_job_t task, unsigned job_is_already_locked);
struct starpu_task *_starpu_pop_task(void);
struct starpu_task *_starpu_pop_every_task(uint32_t where);

void _starpu_wait_on_sched_event(void);

#endif // __SCHED_POLICY_H__
