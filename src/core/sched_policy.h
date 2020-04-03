/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

/** @file */

#include <starpu.h>
#include <signal.h>
#include <core/workers.h>
#include <core/sched_ctx.h>
#include <starpu_scheduler.h>

#include <core/simgrid.h>

#define _STARPU_SCHED_BEGIN \
	_STARPU_TRACE_WORKER_SCHEDULING_PUSH;	\
	_SIMGRID_TIMER_BEGIN(_starpu_simgrid_sched_cost())
#define _STARPU_SCHED_END \
	_SIMGRID_TIMER_END;			\
	_STARPU_TRACE_WORKER_SCHEDULING_POP

void _starpu_sched_init(void);

struct starpu_machine_config;
struct starpu_sched_policy *_starpu_get_sched_policy( struct _starpu_sched_ctx *sched_ctx);

void _starpu_init_sched_policy(struct _starpu_machine_config *config,
			       struct _starpu_sched_ctx *sched_ctx, struct starpu_sched_policy *policy);

void _starpu_deinit_sched_policy(struct _starpu_sched_ctx *sched_ctx);

struct starpu_sched_policy *_starpu_select_sched_policy(struct _starpu_machine_config *config, const char *required_policy);

void _starpu_sched_task_submit(struct starpu_task *task);
void _starpu_sched_do_schedule(unsigned sched_ctx_id);

int _starpu_push_task(struct _starpu_job *task);
int _starpu_repush_task(struct _starpu_job *task);

/** actually pushes the tasks to the specific worker or to the scheduler */
int _starpu_push_task_to_workers(struct starpu_task *task);

/** pop a task that can be executed on the worker */
struct starpu_task *_starpu_pop_task(struct _starpu_worker *worker);
/** pop every task that can be executed on the worker */
struct starpu_task *_starpu_pop_every_task(struct _starpu_sched_ctx *sched_ctx);
void _starpu_sched_post_exec_hook(struct starpu_task *task);
int _starpu_pop_task_end(struct starpu_task *task);

void _starpu_wait_on_sched_event(void);

struct starpu_task *_starpu_create_conversion_task(starpu_data_handle_t handle,
						   unsigned int node) STARPU_ATTRIBUTE_MALLOC;

struct starpu_task *_starpu_create_conversion_task_for_arch(starpu_data_handle_t handle,
						   enum starpu_node_kind node_kind) STARPU_ATTRIBUTE_MALLOC;

void _starpu_sched_pre_exec_hook(struct starpu_task *task);

void _starpu_print_idle_time();
/*
 *	Predefined policies
 */
extern struct starpu_sched_policy _starpu_sched_lws_policy;
extern struct starpu_sched_policy _starpu_sched_ws_policy;
extern struct starpu_sched_policy _starpu_sched_prio_policy;
extern struct starpu_sched_policy _starpu_sched_random_policy;
extern struct starpu_sched_policy _starpu_sched_dm_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_prio_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_ready_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_sorted_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_sorted_decision_policy;
extern struct starpu_sched_policy _starpu_sched_eager_policy;
extern struct starpu_sched_policy _starpu_sched_parallel_heft_policy;
extern struct starpu_sched_policy _starpu_sched_peager_policy;
extern struct starpu_sched_policy _starpu_sched_heteroprio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_eager_policy;
extern struct starpu_sched_policy _starpu_sched_modular_eager_prefetching_policy;
extern struct starpu_sched_policy _starpu_sched_modular_eager_prio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_gemm_policy;
extern struct starpu_sched_policy _starpu_sched_modular_prio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_prio_prefetching_policy;
extern struct starpu_sched_policy _starpu_sched_modular_random_policy;
extern struct starpu_sched_policy _starpu_sched_modular_random_prio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_random_prefetching_policy;
extern struct starpu_sched_policy _starpu_sched_modular_random_prio_prefetching_policy;
extern struct starpu_sched_policy _starpu_sched_modular_parallel_random_policy;
extern struct starpu_sched_policy _starpu_sched_modular_parallel_random_prio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_ws_policy;
extern struct starpu_sched_policy _starpu_sched_modular_heft_policy;
extern struct starpu_sched_policy _starpu_sched_modular_heft_prio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_heft2_policy;
extern struct starpu_sched_policy _starpu_sched_modular_heteroprio_policy;
extern struct starpu_sched_policy _starpu_sched_modular_heteroprio_heft_policy;
extern struct starpu_sched_policy _starpu_sched_modular_parallel_heft_policy;
extern struct starpu_sched_policy _starpu_sched_graph_test_policy;
extern struct starpu_sched_policy _starpu_sched_tree_heft_hierarchical_policy;

extern long _starpu_task_break_on_push;
extern long _starpu_task_break_on_sched;
extern long _starpu_task_break_on_pop;
extern long _starpu_task_break_on_exec;

#ifdef SIGTRAP
#define _STARPU_TASK_BREAK_ON(task, what) do { \
	if (_starpu_get_job_associated_to_task(task)->job_id == (unsigned long) _starpu_task_break_on_##what) \
		raise(SIGTRAP); \
} while(0)
#else
#define _STARPU_TASK_BREAK_ON(task, what) ((void) 0)
#endif

#endif // __SCHED_POLICY_H__
