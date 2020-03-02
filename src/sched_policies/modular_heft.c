/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include <float.h>
#include <limits.h>

/* The scheduling strategy look like this :
 *
 *                                    |
 *                              window_component
 *                                    |
 * mct_component <--push-- perfmodel_select_component --push--> eager_component
 *          |                                                    |
 *          |                                                    |
 *          >----------------------------------------------------<
 *                    |                                |
 *              best_impl_component                    best_impl_component
 *                    |                                |
 *                prio_component                        prio_component
 *                    |                                |
 *               worker_component                   worker_component
 *
 * A window contain the tasks that failed to be pushed, so as when the prio_components reclaim
 * tasks by calling can_push to their parent (classically, just after a successful pop have
 * been made by its associated worker_component), this call goes up to the window_component which
 * pops a task from its local queue and try to schedule it by pushing it to the
 * decision_component. 
 * Finally, the task will be pushed to the prio_component which is the direct
 * parent in the tree of the worker_component the task has been scheduled on. This
 * component will push the task on its local queue if no one of the two thresholds
 * have been reached for it, or send a push_error signal to its parent.
 */

static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_mct_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_WORKERS |
			STARPU_SCHED_SIMPLE_PERFMODEL |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(t);
}

struct starpu_sched_policy _starpu_sched_modular_heft_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "modular-heft",
	.policy_description = "heft modular policy",
	.worker_type = STARPU_WORKER_LIST,
};
