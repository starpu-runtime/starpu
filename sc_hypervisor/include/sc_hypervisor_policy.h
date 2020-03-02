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

#ifndef SC_HYPERVISOR_POLICY_H
#define SC_HYPERVISOR_POLICY_H

#include <sc_hypervisor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_SC_Hypervisor Scheduling Context Hypervisor - Building a new resizing policy
   @{
*/

#define HYPERVISOR_REDIM_SAMPLE 0.02
#define HYPERVISOR_START_REDIM_SAMPLE 0.1
#define SC_NOTHING 0
#define SC_IDLE 1
#define SC_SPEED 2

struct types_of_workers
{
	unsigned ncpus;
	unsigned ncuda;
	unsigned nw;
};

/**
   Task wrapper linked list
   @ingroup API_SC_Hypervisor
*/
struct sc_hypervisor_policy_task_pool
{
	/**
	   Which codelet has been executed
	*/
	struct starpu_codelet *cl;

	/**
	   Task footprint key
	*/
	uint32_t footprint;

	/**
	   Context the task belongs to
	*/
	unsigned sched_ctx_id;

	/**
	   Number of tasks of this kind
	*/
	unsigned long n;

	/**
	   The quantity of data(in bytes) needed by the task to execute
	*/
	size_t data_size;

	/**
	   Other task kinds
	*/
	struct sc_hypervisor_policy_task_pool *next;
};

/**
   add task information to a task wrapper linked list
*/
void sc_hypervisor_policy_add_task_to_pool(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint, struct sc_hypervisor_policy_task_pool **task_pools, size_t data_size);

/**
   remove task information from a task wrapper linked list
*/
void sc_hypervisor_policy_remove_task_from_pool(struct starpu_task *task, uint32_t footprint, struct sc_hypervisor_policy_task_pool **task_pools);

/**
   clone a task wrapper linked list
*/
struct sc_hypervisor_policy_task_pool* sc_hypervisor_policy_clone_task_pool(struct sc_hypervisor_policy_task_pool *tp);

/**
   get the execution time of the submitted tasks out of starpu's calibration files
*/
void sc_hypervisor_get_tasks_times(int nw, int nt, double times[nw][nt], int *workers, unsigned size_ctxs, struct sc_hypervisor_policy_task_pool *task_pools);

/**
   find the context with the lowest priority in order to move some workers
*/
unsigned sc_hypervisor_find_lowest_prio_sched_ctx(unsigned req_sched_ctx, int nworkers_to_move);

/**
   find the first most idle workers of a contex
*/
int* sc_hypervisor_get_idlest_workers(unsigned sched_ctx, int *nworkers, enum starpu_worker_archtype arch);

/**
   find the first most idle workers in a list
*/
int* sc_hypervisor_get_idlest_workers_in_list(int *start, int *workers, int nall_workers,  int *nworkers, enum starpu_worker_archtype arch);

/**
   find workers that can be moved from a context (if the constraints of min, max, etc allow this)
*/
int sc_hypervisor_get_movable_nworkers(struct sc_hypervisor_policy_config *config, unsigned sched_ctx, enum starpu_worker_archtype arch);

/**
   compute how many workers should be moved from this context
*/
int sc_hypervisor_compute_nworkers_to_move(unsigned req_sched_ctx);

/**
   check the policy's constraints in order to resize
*/
unsigned sc_hypervisor_policy_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize, unsigned now);

/**
   check the policy's constraints in order to resize  and find a context willing the resources
*/
unsigned sc_hypervisor_policy_resize_to_unknown_receiver(unsigned sender_sched_ctx, unsigned now);

/**
   compute the speed of a context
*/
double sc_hypervisor_get_ctx_speed(struct sc_hypervisor_wrapper* sc_w);

/**
   get the time of execution of the slowest context
*/
double sc_hypervisor_get_slowest_ctx_exec_time(void);

/**
   get the time of execution of the fastest context
*/
double sc_hypervisor_get_fastest_ctx_exec_time(void);

/**
   compute the speed of a workers in a context
*/
double sc_hypervisor_get_speed_per_worker(struct sc_hypervisor_wrapper *sc_w, unsigned worker);

/**
   compute the speed of a type of worker in a context
*/
double sc_hypervisor_get_speed_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch);

/**
   compute the speed of a type of worker in a context depending on its history
*/
double sc_hypervisor_get_ref_speed_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch);

/**
   compute the average speed of a type of worker in all ctxs from the begining of appl
*/
double sc_hypervisor_get_avg_speed(enum starpu_worker_archtype arch);

/**
   verify if we need to consider the max in the lp
*/
void sc_hypervisor_check_if_consider_max(struct types_of_workers *tw);

/**
   get the list of workers grouped by type
*/
void sc_hypervisor_group_workers_by_type(struct types_of_workers *tw, int *total_nw);

/**
   get what type of worker corresponds to a certain index of types of workers
*/
enum starpu_worker_archtype sc_hypervisor_get_arch_for_index(unsigned w, struct types_of_workers *tw);

/**
   get the index of types of workers corresponding to the type of workers indicated
*/
unsigned sc_hypervisor_get_index_for_arch(enum starpu_worker_archtype arch, struct types_of_workers *tw);

/**
   check if we trigger resizing or not
*/
unsigned sc_hypervisor_criteria_fulfilled(unsigned sched_ctx, int worker);

/**
   check if worker was idle long enough
*/
unsigned sc_hypervisor_check_idle(unsigned sched_ctx, int worker);

/**
   check if there is a speed gap btw ctxs
*/
unsigned sc_hypervisor_check_speed_gap_btw_ctxs(unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

/**
   check if there is a speed gap btw ctxs on one level
*/
unsigned sc_hypervisor_check_speed_gap_btw_ctxs_on_level(int level, int *workers_in, int nworkers_in, unsigned father_sched_ctx_id, unsigned **sched_ctxs, int *nsched_ctxs);

/**
   check what triggers resizing (idle, speed, etc.
*/
unsigned sc_hypervisor_get_resize_criteria();

/**
   load information concerning the type of workers into a types_of_workers struct
*/
struct types_of_workers* sc_hypervisor_get_types_of_workers(int *workers, unsigned nworkers);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
