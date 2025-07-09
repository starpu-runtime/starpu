/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SPLITTER_H__
#define __SPLITTER_H__

#include <starpu.h>

#pragma GCC visibility push(hidden)

#ifdef STARPU_RECURSIVE_TASKS
void _splitter_actualize_ratio_to_split(struct starpu_codelet *cl, double ratio, unsigned level);
void _splitter_reinit_cache_entry();
int _splitter_choose_three_dimensions(struct starpu_task *ptask, int nb_tasks, int sched_ctx_id);
int _splitter_simulate_three_dimensions(struct starpu_task *ptask, int nb_tasks, int sched_ctx_id);
int starpu_task_is_recursive_splitter(struct starpu_task *task);
void starpu_splitter_hook_submit_subdag(struct starpu_task *task);
void starpu_splitter_task_is_submit(struct starpu_task *task);
void starpu_splitter_task_is_ended(struct starpu_task *task);
void _starpu_splitter_all_cpu_subtasks_end(struct starpu_task *task);
void _splitter_initialize_data();

void starpu_splitter_task_submit(struct starpu_task *task);
void _starpu_splitter_all_dependencies_are_fulfilled(struct starpu_task *task);
void _starpu_splitter_one_job_ended(struct starpu_task *task);
#endif

#pragma GCC visibility pop

#endif
