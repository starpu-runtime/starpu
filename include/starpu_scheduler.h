/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
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

#ifndef __STARPU_SCHEDULER_H__
#define __STARPU_SCHEDULER_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_task;

struct starpu_sched_policy
{
	void (*init_sched)(unsigned sched_ctx_id);
	void (*deinit_sched)(unsigned sched_ctx_id);

	int (*push_task)(struct starpu_task *);
	void (*push_task_notify)(struct starpu_task *, int workerid, unsigned sched_ctx_id);
	struct starpu_task *(*pop_task)(unsigned sched_ctx_id);
	struct starpu_task *(*pop_every_task)(unsigned sched_ctx_id);

	void (*pre_exec_hook)(struct starpu_task *);
	void (*post_exec_hook)(struct starpu_task *);

	void (*add_workers)(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
	void (*remove_workers)(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

	const char *policy_name;
	const char *policy_description;
};

struct starpu_sched_policy **starpu_sched_get_predefined_policies();

void starpu_worker_get_sched_condition(int workerid, starpu_pthread_mutex_t **sched_mutex, starpu_pthread_cond_t **sched_cond);

int starpu_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

int starpu_push_local_task(int workerid, struct starpu_task *task, int back);

int starpu_push_task_end(struct starpu_task *task);

int starpu_combined_worker_assign_workerid(int nworkers, int workerid_array[]);
int starpu_combined_worker_get_description(int workerid, int *worker_size, int **combined_workerid);
int starpu_combined_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

int starpu_get_prefetch_flag(void);
int starpu_prefetch_task_input_on_node(struct starpu_task *task, unsigned node);

uint32_t starpu_task_footprint(struct starpu_perfmodel *model, struct starpu_task *task, enum starpu_perfmodel_archtype arch, unsigned nimpl);
double starpu_task_expected_length(struct starpu_task *task, enum starpu_perfmodel_archtype arch, unsigned nimpl);
double starpu_worker_get_relative_speedup(enum starpu_perfmodel_archtype perf_archtype);
double starpu_task_expected_data_transfer_time(unsigned memory_node, struct starpu_task *task);
double starpu_data_expected_transfer_time(starpu_data_handle_t handle, unsigned memory_node, enum starpu_data_access_mode mode);
double starpu_task_expected_power(struct starpu_task *task, enum starpu_perfmodel_archtype arch, unsigned nimpl);
double starpu_task_expected_conversion_time(struct starpu_task *task, enum starpu_perfmodel_archtype arch, unsigned nimpl);

double starpu_task_bundle_expected_length(starpu_task_bundle_t bundle, enum starpu_perfmodel_archtype arch, unsigned nimpl);
double starpu_task_bundle_expected_data_transfer_time(starpu_task_bundle_t bundle, unsigned memory_node);
double starpu_task_bundle_expected_power(starpu_task_bundle_t bundle, enum starpu_perfmodel_archtype arch, unsigned nimpl);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHEDULER_H__ */
