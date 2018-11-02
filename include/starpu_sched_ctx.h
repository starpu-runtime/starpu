/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2012-2017                           Inria
 * Copyright (C) 2017                                     Arthur Chevalier
 * Copyright (C) 2012-2014,2017                           CNRS
 * Copyright (C) 2012,2014,2016                           Université de Bordeaux
 * Copyright (C) 2016                                     Uppsala University
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

#ifndef __STARPU_SCHED_CTX_H__
#define __STARPU_SCHED_CTX_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define STARPU_SCHED_CTX_POLICY_NAME		 (1<<16)
#define STARPU_SCHED_CTX_POLICY_STRUCT		 (2<<16)
#define STARPU_SCHED_CTX_POLICY_MIN_PRIO	 (3<<16)
#define STARPU_SCHED_CTX_POLICY_MAX_PRIO	 (4<<16)
#define STARPU_SCHED_CTX_HIERARCHY_LEVEL         (5<<16)
#define STARPU_SCHED_CTX_NESTED                  (6<<16)
#define STARPU_SCHED_CTX_AWAKE_WORKERS           (7<<16)
#define STARPU_SCHED_CTX_POLICY_INIT             (8<<16)
#define STARPU_SCHED_CTX_USER_DATA               (9<<16)
#define STARPU_SCHED_CTX_CUDA_NSMS               (10<<16)
#define STARPU_SCHED_CTX_SUB_CTXS                (11<<16)

unsigned starpu_sched_ctx_create(int *workerids_ctx, int nworkers_ctx, const char *sched_ctx_name, ...);

unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_ctx_name, int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus, unsigned allow_overlap);

void starpu_sched_ctx_register_close_callback(unsigned sched_ctx_id, void (*close_callback)(unsigned sched_ctx_id, void* args), void *args);

void starpu_sched_ctx_add_workers(int *workerids_ctx, unsigned nworkers_ctx, unsigned sched_ctx_id);

void starpu_sched_ctx_remove_workers(int *workerids_ctx, unsigned nworkers_ctx, unsigned sched_ctx_id);

void starpu_sched_ctx_display_workers(unsigned sched_ctx_id, FILE *f);

void starpu_sched_ctx_delete(unsigned sched_ctx_id);

void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor);

unsigned starpu_sched_ctx_get_inheritor(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_hierarchy_level(unsigned sched_ctx_id);

void starpu_sched_ctx_set_context(unsigned *sched_ctx_id);

unsigned starpu_sched_ctx_get_context(void);

void starpu_sched_ctx_stop_task_submission(void);

void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_workers_list(unsigned sched_ctx_id, int **workerids);
unsigned starpu_sched_ctx_get_workers_list_raw(unsigned sched_ctx_id, int **workerids);

unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);

unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id);

unsigned starpu_sched_ctx_contains_type_of_worker(enum starpu_worker_archtype arch, unsigned sched_ctx_id);

unsigned starpu_sched_ctx_worker_get_id(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_ctx_for_task(struct starpu_task *task);

unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid);

int starpu_sched_get_min_priority(void);

int starpu_sched_get_max_priority(void);

int starpu_sched_set_min_priority(int min_prio);

int starpu_sched_set_max_priority(int max_prio);

int starpu_sched_ctx_get_min_priority(unsigned sched_ctx_id);

int starpu_sched_ctx_get_max_priority(unsigned sched_ctx_id);

int starpu_sched_ctx_set_min_priority(unsigned sched_ctx_id, int min_prio);

int starpu_sched_ctx_set_max_priority(unsigned sched_ctx_id, int max_prio);

int starpu_sched_ctx_min_priority_is_set(unsigned sched_ctx_id);

int starpu_sched_ctx_max_priority_is_set(unsigned sched_ctx_id);

#define STARPU_MIN_PRIO		(starpu_sched_get_min_priority())
#define STARPU_MAX_PRIO		(starpu_sched_get_max_priority())

#define STARPU_DEFAULT_PRIO	0

void *starpu_sched_ctx_get_user_data(unsigned sched_ctx_id);

void starpu_sched_ctx_set_user_data(unsigned sched_ctx_id, void* user_data);

struct starpu_worker_collection *starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, enum starpu_worker_collection_type type) STARPU_ATTRIBUTE_MALLOC;

void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id);

struct starpu_worker_collection *starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id);

void starpu_sched_ctx_set_policy_data(unsigned sched_ctx_id, void *policy_data);

void *starpu_sched_ctx_get_policy_data(unsigned sched_ctx_id);

struct starpu_sched_policy *starpu_sched_ctx_get_sched_policy(unsigned sched_ctx_id);

void *starpu_sched_ctx_exec_parallel_code(void* (*func)(void*), void *param, unsigned sched_ctx_id);

int starpu_sched_ctx_get_nready_tasks(unsigned sched_ctx_id);

double starpu_sched_ctx_get_nready_flops(unsigned sched_ctx_id);

void starpu_sched_ctx_list_task_counters_increment(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_list_task_counters_decrement(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_list_task_counters_reset(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_list_task_counters_increment_all_ctx_locked(struct starpu_task *task, unsigned sched_ctx_id);

void starpu_sched_ctx_list_task_counters_decrement_all_ctx_locked(struct starpu_task *task, unsigned sched_ctx_id);

void starpu_sched_ctx_list_task_counters_reset_all(struct starpu_task *task, unsigned sched_ctx_id);

void starpu_sched_ctx_set_priority(int *workers, int nworkers, unsigned sched_ctx_id, unsigned priority);

unsigned starpu_sched_ctx_get_priority(int worker, unsigned sched_ctx_id);

void starpu_sched_ctx_get_available_cpuids(unsigned sched_ctx_id, int **cpuids, int *ncpuids);

void starpu_sched_ctx_bind_current_thread_to_cpuid(unsigned cpuid);

int starpu_sched_ctx_book_workers_for_task(unsigned sched_ctx_id, int *workerids, int nworkers);

void starpu_sched_ctx_unbook_workers_for_task(unsigned sched_ctx_id, int master);

/* return the first context (child of sched_ctx_id) where the workerid is master */
unsigned starpu_sched_ctx_worker_is_master_for_child_ctx(int workerid, unsigned sched_ctx_id);

/* Returns the context id of masterid if it master of a context. */
/* If not, returns STARPU_NMAX_SCHED_CTXS. */
unsigned starpu_sched_ctx_master_get_context(int masterid);

void starpu_sched_ctx_revert_task_counters_ctx_locked(unsigned sched_ctx_id, double flops);

void starpu_sched_ctx_move_task_to_ctx_locked(struct starpu_task *task, unsigned sched_ctx, unsigned with_repush);

int starpu_sched_ctx_get_worker_rank(unsigned sched_ctx_id);

void (*starpu_sched_ctx_get_sched_policy_init(unsigned sched_ctx_id))(unsigned);

unsigned starpu_sched_ctx_has_starpu_scheduler(unsigned sched_ctx_id, unsigned *awake_workers);
#ifdef STARPU_USE_SC_HYPERVISOR
void starpu_sched_ctx_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);
#endif /* STARPU_USE_SC_HYPERVISOR */

int starpu_sched_ctx_get_stream_worker(unsigned sub_ctx);
int starpu_sched_ctx_get_nsms(unsigned sched_ctx);
void starpu_sched_ctx_get_sms_interval(int stream_workerid, int *start, int *end);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_H__ */
