/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
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

static void initialize_readys_policy(unsigned sched_ctx_id)
{
    starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_mct_create, NULL,
                                                       0, sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_readys_policy =
{
    .init_sched = initialize_readys_policy,
    .deinit_sched = NULL,
    .add_workers = NULL,
    .remove_workers = NULL,
    .push_task = NULL,
    .pop_task = NULL,
    .pre_exec_hook = NULL,
    .post_exec_hook = NULL,
    .pop_every_task = NULL,
    .policy_name = "modular-readys",
    .policy_description = "READYS modular policy",
    .worker_type = STARPU_WORKER_LIST,
};
