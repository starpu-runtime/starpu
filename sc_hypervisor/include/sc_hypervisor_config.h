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

#ifndef SC_HYPERVISOR_CONFIG_H
#define SC_HYPERVISOR_CONFIG_H

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
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 3 arguments: an array of int for the workerids to apply
   the condition, an int to indicate the size of the array, and a
   double value indicating the maximum idle time allowed for a worker
   before the resizing process should be triggered
*/
#define SC_HYPERVISOR_MAX_IDLE -1

#define SC_HYPERVISOR_MIN_WORKING -2

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 3 arguments: an array of int for the workerids to apply
   the condition, an int to indicate the size of the array, and an int
   value indicating the priority of the workers previously mentioned.
   The workers with the smallest priority are moved the first.
*/
#define SC_HYPERVISOR_PRIORITY -3

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument(int) indicating the minimum number of
   workers a context should have, underneath this limit the context
   cannot execute.
*/
#define SC_HYPERVISOR_MIN_WORKERS -4

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument(int) indicating the maximum number of
   workers a context should have, above this limit the context would
   not be able to scale
*/
#define SC_HYPERVISOR_MAX_WORKERS -5

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument(int) indicating the granularity of the
   resizing process (the number of workers should be moved from the
   context once it is resized) This parameter is ignore for the Gflops
   rate based strategy (see \ref ResizingStrategies), the number of
   workers that have to be moved is calculated by the strategy.
*/
#define SC_HYPERVISOR_GRANULARITY -6

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 2 arguments: an array of int for the workerids to apply
   the condition and an int to indicate the size of the array. These
   workers are not allowed to be moved from the context.
*/
#define SC_HYPERVISOR_FIXED_WORKERS -7

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument (int) that indicated the minimum number of
   tasks that have to be executed before the context could be resized.
   This parameter is ignored for the Application Driven strategy (see
   \ref ResizingStrategies) where the user indicates exactly when the
   resize should be done.
*/
#define SC_HYPERVISOR_MIN_TASKS -8

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument, a double value indicating the maximum idle
   time allowed for workers that have just been moved from other
   contexts in the current context.
*/
#define SC_HYPERVISOR_NEW_WORKERS_MAX_IDLE -9

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument (int) indicating the tag an executed task
   should have such that this configuration should be taken into
   account.
*/
#define SC_HYPERVISOR_TIME_TO_APPLY -10

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument
 */
#define SC_HYPERVISOR_NULL -11

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument, a double, that indicates the number of
   flops needed to be executed before computing the speed of a worker
*/
#define	SC_HYPERVISOR_ISPEED_W_SAMPLE -12

/**
   This macro is used when calling sc_hypervisor_ctl() and must be
   followed by 1 argument, a double, that indicates the number of
   flops needed to be executed before computing the speed of a context
*/
#define SC_HYPERVISOR_ISPEED_CTX_SAMPLE -13

#define SC_HYPERVISOR_TIME_SAMPLE -14

#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

/**
   Methods that implement a hypervisor resizing policy.
*/
struct sc_hypervisor_policy_config
{
	/**
	   Indicate the minimum number of workers needed by the context
	*/
	int min_nworkers;

	/**
	   Indicate the maximum number of workers needed by the context
	*/
	int max_nworkers;

	/**
	   Indicate the workers granularity of the context
	*/
	int granularity;

	/**
	   Indicate the priority of each worker to stay in the context
	   the smaller the priority the faster it will be moved to
	   another context
	*/
	int priority[STARPU_NMAXWORKERS];

	/**
	   Indicate the maximum idle time accepted before a resize is
	   triggered
	   above this limit the priority of the worker is reduced
	*/
	double max_idle[STARPU_NMAXWORKERS];

	/**
	   Indicate that underneath this limit the priority of the
	   worker is reduced
	*/
	double min_working[STARPU_NMAXWORKERS];

	/**
	   Indicate which workers can be moved and which ones are
	   fixed
	*/
	int fixed_workers[STARPU_NMAXWORKERS];

	/**
	   Indicate the maximum idle time accepted before a resize is
	   triggered for the workers that just arrived in the new
	   context
	*/
	double new_workers_max_idle;

	/**
	   Indicate the sample used to compute the instant speed per
	   worker
	*/
	double ispeed_w_sample[STARPU_NMAXWORKERS];

	/**
	   Indicate the sample used to compute the instant speed per
	   ctxs
	*/
	double ispeed_ctx_sample;

        /**
	   Indicate the sample used to compute the instant speed per
	   ctx (in seconds)
	*/
	double time_sample;
};

/**
   Specify the configuration for a context
*/
void sc_hypervisor_set_config(unsigned sched_ctx, void *config);

/**
   Return the configuration of a context
*/
struct sc_hypervisor_policy_config *sc_hypervisor_get_config(unsigned sched_ctx);

/**
   Specify different parameters for the configuration of a context.
   The list must be zero-terminated
*/
void sc_hypervisor_ctl(unsigned sched_ctx, ...);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
