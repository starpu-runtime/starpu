/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef SC_HYPERVISOR_H
#define SC_HYPERVISOR_H

#include <starpu.h>
#include <starpu_sched_ctx_hypervisor.h>
#include <sc_hypervisor_config.h>
#include <sc_hypervisor_monitoring.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @ingroup API_SC_Hypervisor
   Methods to implement a hypervisor resizing policy.
*/
struct sc_hypervisor_policy
{
	/**
	   Indicate the name of the policy, if there is not a custom
	   policy, the policy corresponding to this name will be used
	   by the hypervisor
	*/
	const char* name;

	/**
	   Indicate whether the policy is custom or not
	*/
	unsigned custom;

	/**
	   Distribute workers to contexts even at the begining of the
	   program
	*/
	void (*size_ctxs)(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers);

	/**
	   Require explicit resizing
	*/
	void (*resize_ctxs)(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers);

	/**
	   Called whenever the indicated worker executes another idle
	   cycle in sched_ctx
	*/
	void (*handle_idle_cycle)(unsigned sched_ctx, int worker);

	/**
	   Called whenever a task is pushed on the worker’s queue
	   corresponding to the context sched_ctx
	*/
	void (*handle_pushed_task)(unsigned sched_ctx, int worker);

	/**
	   Called whenever a task is poped from the worker’s queue
	   corresponding to the context sched_ctx
	*/
	void (*handle_poped_task)(unsigned sched_ctx, int worker,struct starpu_task *task, uint32_t footprint);

	/**
	   Called whenever a task is executed on the indicated worker
	   and context after a long period of idle time
	*/
	void (*handle_idle_end)(unsigned sched_ctx, int worker);

	/**
	   Called whenever a tag task has just been executed. The
	   table of resize requests is provided as well as the tag
	*/
	void (*handle_post_exec_hook)(unsigned sched_ctx, int task_tag);

	/**
	   the hypervisor takes a decision when a job was submitted in
	   this ctx
	*/
	void (*handle_submitted_job)(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint, size_t data_size);

	/**
	   the hypervisor takes a decision when a certain ctx was
	   deleted
	*/
	void (*end_ctx)(unsigned sched_ctx);

	/**
	   the hypervisor takes a decision when a certain ctx was
	   registerd
	*/
	void (*start_ctx)(unsigned sched_ctx);

	/**
	   the hypervisor initializes values for the workers
	*/
	void (*init_worker)(int workerid, unsigned sched_ctx);
};

/**
   @defgroup API_SC_Hypervisor_usage Scheduling Context Hypervisor - Regular usage
   There is a single hypervisor that is in charge of resizing contexts
   and the resizing strategy is chosen at the initialization of the
   hypervisor. A single resize can be done at a time.

   The Scheduling Context Hypervisor Plugin provides a series of
   performance counters to StarPU. By incrementing them, StarPU can
   help the hypervisor in the resizing decision making process.

   The function sc_hypervisor_init() initializes the hypervisor to use
   the strategy provided as parameter and creates the performance
   counters (see starpu_sched_ctx_performance_counters). These
   performance counters represent actually some callbacks that will be
   used by the contexts to notify the information needed by the
   hypervisor.

   Scheduling Contexts that have to be resized by the hypervisor must
   be first registered to the hypervisor using the function
   sc_hypervisor_register_ctx()

   Note: The Hypervisor is actually a worker that takes this role once
   certain conditions trigger the resizing process (there is no
   additional thread assigned to the hypervisor).
   @{
*/

/**
   synchronise the hypervisor when several workers try to update its
   information
*/
extern starpu_pthread_mutex_t act_hypervisor_mutex;

/**
   Start the hypervisor with the given policy
*/
void* sc_hypervisor_init(struct sc_hypervisor_policy *policy);

/**
   Shutdown the hypervisor.
   The hypervisor and all information concerning it is cleaned. There
   is no synchronization between this function and starpu_shutdown().
   Thus, this should be called after starpu_shutdown(), because the
   performance counters will still need allocated callback functions.
*/
void sc_hypervisor_shutdown(void);

/**
   Register the context to the hypervisor, and indicate the number of
   flops the context will execute (used for Gflops rate based strategy)
*/
void sc_hypervisor_register_ctx(unsigned sched_ctx, double total_flops);

/**
   Unregister a context from the hypervisor, and so exclude the
   context from the resizing process
*/
void sc_hypervisor_unregister_ctx(unsigned sched_ctx);

/**
   Require resizing the context \p sched_ctx whenever a task tagged
   with the id \p task_tag finished executing
*/
void sc_hypervisor_post_resize_request(unsigned sched_ctx, int task_tag);

/**
   Require reconsidering the distribution of ressources over the
   indicated scheduling contexts, i.e reevaluate the distribution of
   the resources and eventually resize if needed
*/
void sc_hypervisor_resize_ctxs(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers);

/**
   Do not allow the hypervisor to resize a context.
*/
void sc_hypervisor_stop_resize(unsigned sched_ctx);

/**
   Allow the hypervisor to resize a context if necessary.
*/
void sc_hypervisor_start_resize(unsigned sched_ctx);

/**
   Return the name of the resizing policy used by the hypervisor
*/
const char *sc_hypervisor_get_policy();

/**
   Ask the hypervisor to add workers to a sched_ctx
*/
void sc_hypervisor_add_workers_to_sched_ctx(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx);

/**
   Ask the hypervisor to remove workers from a sched_ctx
*/
void sc_hypervisor_remove_workers_from_sched_ctx(int* workers_to_remove, unsigned nworkers_to_remove, unsigned sched_ctx, unsigned now);

/**
   Ask the hypervisor to move workers from one context to another
*/
void sc_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *workers_to_move, unsigned nworkers_to_move, unsigned now);

/**
   Ask the hypervisor to choose a distribution of workers in the
   required contexts
*/
void sc_hypervisor_size_ctxs(unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

/**
   Check if there are pending demands of resizing
*/
unsigned sc_hypervisor_get_size_req(unsigned **sched_ctxs, int* nsched_ctxs, int **workers, int *nworkers);

/**
   Save a demand of resizing
*/
void sc_hypervisor_save_size_req(unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

/**
   Clear the list of pending demands of resizing
*/
void sc_hypervisor_free_size_req(void);

/**
   Check out if a context can be resized
*/
unsigned sc_hypervisor_can_resize(unsigned sched_ctx);

/**
   Indicate the types of tasks a context will execute in order to
   better decide the sizing of ctxs
*/
void sc_hypervisor_set_type_of_task(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint, size_t data_size);

/**
   Change dynamically the total number of flops of a context, move the
   deadline of the finishing time of the context
*/
void sc_hypervisor_update_diff_total_flops(unsigned sched_ctx, double diff_total_flops);

/**
   Change dynamically the number of the elapsed flops in a context,
   modify the past in order to better compute the speed
*/
void sc_hypervisor_update_diff_elapsed_flops(unsigned sched_ctx, double diff_task_flops);

/**
   Update the min and max workers needed by each context
*/
void sc_hypervisor_update_resize_interval(unsigned *sched_ctxs, int nsched_ctxs, int max_nworkers);

/**
   Return a list of contexts that are on the same level in the
   hierarchy of contexts
*/
void sc_hypervisor_get_ctxs_on_level(unsigned **sched_ctxs, int *nsched_ctxs, unsigned hierarchy_level, unsigned father_sched_ctx_id);

/**
   Returns the number of levels of ctxs registered to the hyp
*/
unsigned sc_hypervisor_get_nhierarchy_levels(void);

/**
   Return the leaves ctxs from the list of ctxs
*/
void sc_hypervisor_get_leaves(unsigned *sched_ctxs, int nsched_ctxs, unsigned *leaves, int *nleaves);

/**
   Return the nready flops of all ctxs below in hierachy of sched_ctx
*/
double sc_hypervisor_get_nready_flops_of_all_sons_of_sched_ctx(unsigned sched_ctx);

void sc_hypervisor_print_overhead();

void sc_hypervisor_init_worker(int workerid, unsigned sched_ctx);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
