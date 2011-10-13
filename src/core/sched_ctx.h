/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
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

#ifndef __SCHED_CONTEXT_H__
#define __SCHED_CONTEXT_H__

#include <starpu.h>
#include <starpu_scheduler.h>
#include <common/config.h>
#include <common/barrier_counter.h>
#include <profiling/profiling.h>

#define NO_RESIZE -1
#define REQ_RESIZE 0
#define DO_RESIZE 1

struct starpu_sched_ctx {
	/* id of the context used in user mode*/
	unsigned id;

	/* name of context */
	const char *name;

	/* policy of the context */
	struct starpu_sched_policy_s *sched_policy;

	/* data necessary for the policy */
	void *policy_data;
	
	/* list of indices of workers */
	int workerids[STARPU_NMAXWORKERS]; 

	/* list of workers, those checked have to be deleted */
	int workerids_to_remove[STARPU_NMAXWORKERS]; 

	/* list of workers, those checked have to be added */
	int workerids_to_add[STARPU_NMAXWORKERS]; 
	
	/* number of threads in contex */
	int nworkers; 

	/* mutext for temp_nworkers_in_ctx*/
	pthread_mutex_t changing_ctx_mutex;

	/* we keep an initial sched which we never delete */
	unsigned is_initial_sched; 

	/* wait for the tasks submitted to the context to be executed */
	struct _starpu_barrier_counter_t tasks_barrier;

	/* table of sched cond corresponding to each worker in this ctx */
	pthread_cond_t **sched_cond;

	/* table of sched mutex corresponding to each worker in this ctx */
	pthread_mutex_t **sched_mutex;
#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	/* a structure containing a series of criteria determining the resize procedure */
	struct starpu_sched_ctx_hypervisor_criteria *criteria;
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR
	unsigned modified;
};

struct starpu_machine_config_s;

/* init sched_ctx_id of all contextes*/
void _starpu_init_all_sched_ctxs(struct starpu_machine_config_s *config);

/* init the list of contextes of the worker */
void _starpu_init_sched_ctx_for_worker(unsigned workerid);

/* allocate all structures belonging to a context */
struct starpu_sched_ctx*  _starpu_create_sched_ctx(const char *policy_name, int *workerid, int nworkerids, unsigned is_init_sched, const char *sched_name);

/* delete all sched_ctx */
void _starpu_delete_all_sched_ctxs();

/* Keeps track of the number of tasks currently submitted to a worker */
void _starpu_decrement_nsubmitted_tasks_of_worker(int workerid);
void _starpu_increment_nsubmitted_tasks_of_worker(int workerid);

/* In order to implement starpu_wait_for_all_tasks_of_ctx, we keep track of the number of 
 * task currently submitted to the context */
void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);
void _starpu_increment_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);

/* Return the corresponding index of the workerid in the ctx table */
int _starpu_get_index_in_ctx_of_workerid(unsigned sched_ctx, unsigned workerid);

/* Get the mutex corresponding to the global workerid */
pthread_mutex_t *_starpu_get_sched_mutex(struct starpu_sched_ctx *sched_ctx, int worker);

/* Get the cond corresponding to the global workerid */
pthread_cond_t *_starpu_get_sched_cond(struct starpu_sched_ctx *sched_ctx, int worker);

/* Get the total number of sched_ctxs created till now */
unsigned _starpu_get_nsched_ctxs();

/* Treat add workers requests */
void _starpu_actually_add_workers_to_sched_ctx(struct starpu_sched_ctx *sched_ctx);

/* Treat remove workers requests */
void _starpu_actually_remove_workers_from_sched_ctx(struct starpu_sched_ctx *sched_ctx);

#endif // __SCHED_CONTEXT_H__
