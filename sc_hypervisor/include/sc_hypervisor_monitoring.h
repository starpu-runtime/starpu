/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2015       Mathieu Lirzin
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

#ifndef SC_HYPERVISOR_MONITORING_H
#define SC_HYPERVISOR_MONITORING_H

#include <sc_hypervisor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @ingroup API_SC_Hypervisor
   @{
*/

/**
   Structure to check if the workers moved to another context are
   actually taken into account in that context.
*/
struct sc_hypervisor_resize_ack
{
	/**
	   The context receiving the new workers
	*/
	int receiver_sched_ctx;

	/**
	   List of workers required to be moved
	*/
	int *moved_workers;

	/**
	   Number of workers required to be moved
	*/
	int nmoved_workers;

	/**
	   List of workers that actually got in the receiver ctx. If
	   the value corresponding to a worker is 1, this worker got
	   moved in the new context.
	*/
	int *acked_workers;
};

/**
   Wrapper of the contexts available in StarPU which contains all
   information about a context obtained by incrementing the
   performance counters. it is attached to a sched_ctx storing
   monitoring information
*/
struct sc_hypervisor_wrapper
{
	/**
	   the monitored context
	*/
	unsigned sched_ctx;

	/**
	   The corresponding resize configuration
	*/
	struct sc_hypervisor_policy_config *config;

	/**
	   the start time of the resizing sample of the workers of
	   this context
	*/
	double start_time_w[STARPU_NMAXWORKERS];

	/**
	   The idle time counter of each worker of the context
	*/
	double current_idle_time[STARPU_NMAXWORKERS];

	/**
	   The time the workers were idle from the last resize
	*/
	double idle_time[STARPU_NMAXWORKERS];

	/**
	   The moment when the workers started being idle
	*/
	double idle_start_time[STARPU_NMAXWORKERS];

	/**
	   Time during which the worker executed tasks
	*/
	double exec_time[STARPU_NMAXWORKERS];

	/**
	   Time when the worker started executing a task
	*/
	double exec_start_time[STARPU_NMAXWORKERS];

	/**
	   List of workers that will leave the context (lazy resizing
	   process)
	*/
	int worker_to_be_removed[STARPU_NMAXWORKERS];

	/**
	   Number of tasks pushed on each worker in this context
	*/
	int pushed_tasks[STARPU_NMAXWORKERS];

	/**
	   Number of tasks poped from each worker in this context
	*/
	int poped_tasks[STARPU_NMAXWORKERS];

	/**
	   The total number of flops to execute by the context
	*/
	double total_flops;

	/**
	   The number of flops executed by each workers of the context
	*/
	double total_elapsed_flops[STARPU_NMAXWORKERS];

	/**
	   number of flops executed since last resizing
	*/
	double elapsed_flops[STARPU_NMAXWORKERS];

	/**
	   Quantity of data (in bytes) used to execute tasks on each
	   worker in this context
	*/
	size_t elapsed_data[STARPU_NMAXWORKERS];

	/**
	   Number of tasks executed on each worker in this context
	*/
	int elapsed_tasks[STARPU_NMAXWORKERS];

	/**
	   the average speed of the type of workers when they belonged
	   to this context
	   0 - cuda 1 - cpu
	*/
	double ref_speed[2];

	/**
	   Number of flops submitted to this context
	*/
	double submitted_flops;

	/**
	   Number of flops that still have to be executed by the
	   workers in this context
	*/
	double remaining_flops;

	/**
	   Start time of the resizing sample of this context
	*/
	double start_time;

	/**
	   First time a task was pushed to this context
	*/
	double real_start_time;

	/**
	   Start time for sample in which the hypervisor is not allowed to
	   react bc too expensive */
	double hyp_react_start_time;

	/**
	   Structure confirming the last resize finished and a new one
	   can be done.
	   Workers do not leave the current context until the receiver
	   context does not ack the receive of these workers
	*/
	struct sc_hypervisor_resize_ack resize_ack;

	/**
	   Mutex needed to synchronize the acknowledgment of the
	   workers into the receiver context
	*/
	starpu_pthread_mutex_t mutex;

	/**
	   Boolean indicating if the hypervisor can use the flops
	   corresponding to the entire execution of the context
	*/
	unsigned total_flops_available;

	/**
	   boolean indicating that a context is being sized
	*/
	unsigned to_be_sized;

	/**
	   Boolean indicating if we add the idle of this worker to the
	   idle of the context
	*/
	unsigned compute_idle[STARPU_NMAXWORKERS];

	/**
	   Boolean indicating if we add the entiere idle of this
	   worker to the idle of the context or just half
	*/
	unsigned compute_partial_idle[STARPU_NMAXWORKERS];

	/**
	   consider the max in the lp
	*/
	unsigned consider_max;
};

/**
   Return the wrapper of the given context
   @ingroup API_SC_Hypervisor
*/
struct sc_hypervisor_wrapper *sc_hypervisor_get_wrapper(unsigned sched_ctx);

/**
   Get the list of registered contexts
   @ingroup API_SC_Hypervisor
*/
unsigned *sc_hypervisor_get_sched_ctxs();

/**
   Get the number of registered contexts
   @ingroup API_SC_Hypervisor
*/
int sc_hypervisor_get_nsched_ctxs();

/**
   Get the number of workers of a certain architecture in a context
*/
int sc_hypervisor_get_nworkers_ctx(unsigned sched_ctx, enum starpu_worker_archtype arch);

/**
   Get the number of flops executed by a context since last resizing
   (reset to 0 when a resizing is done)
   @ingroup API_SC_Hypervisor
*/
double sc_hypervisor_get_elapsed_flops_per_sched_ctx(struct sc_hypervisor_wrapper *sc_w);

/**
   Get the number of flops executed by a context since the begining
*/
double sc_hypervisor_get_total_elapsed_flops_per_sched_ctx(struct sc_hypervisor_wrapper* sc_w);

/**
   Compute an average value of the cpu/cuda speed
*/
double sc_hypervisorsc_hypervisor_get_speed_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch);

/**
   Compte the actual speed of all workers of a specific type of worker
*/
double sc_hypervisor_get_speed(struct sc_hypervisor_wrapper *sc_w, enum starpu_worker_archtype arch);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
