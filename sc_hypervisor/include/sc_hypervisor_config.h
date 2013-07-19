/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011 - 2013  INRIA
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

/* ctl properties*/
#define SC_HYPERVISOR_MAX_IDLE -1
#define SC_HYPERVISOR_MIN_WORKING -2
#define SC_HYPERVISOR_PRIORITY -3
#define SC_HYPERVISOR_MIN_WORKERS -4
#define SC_HYPERVISOR_MAX_WORKERS -5
#define SC_HYPERVISOR_GRANULARITY -6
#define SC_HYPERVISOR_FIXED_WORKERS -7
#define SC_HYPERVISOR_MIN_TASKS -8
#define SC_HYPERVISOR_NEW_WORKERS_MAX_IDLE -9
#define SC_HYPERVISOR_TIME_TO_APPLY -10
#define SC_HYPERVISOR_EMPTY_CTX_MAX_IDLE -11
#define SC_HYPERVISOR_NULL -12
#define	SC_HYPERVISOR_ISPEED_W_SAMPLE -13
#define SC_HYPERVISOR_ISPEED_CTX_SAMPLE -14


#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

struct sc_hypervisor_policy_config
{
	/* underneath this limit we cannot resize */
	int min_nworkers;

	/* above this limit we cannot resize */
	int max_nworkers;

	/*resize granularity */
	int granularity;

	/* priority for a worker to stay in this context */
	/* the smaller the priority the faster it will be moved */
	/* to another context */
	int priority[STARPU_NMAXWORKERS];

	/* above this limit the priority of the worker is reduced */
	double max_idle[STARPU_NMAXWORKERS];

	/* underneath this limit the priority of the worker is reduced */
	double min_working[STARPU_NMAXWORKERS];

	/* workers that will not move */
	int fixed_workers[STARPU_NMAXWORKERS];

	/* max idle for the workers that will be added during the resizing process*/
	double new_workers_max_idle;

	/* above this context we allow removing all workers */
	double empty_ctx_max_idle[STARPU_NMAXWORKERS];

	/* sample used to compute the instant speed per worker*/
	double ispeed_w_sample[STARPU_NMAXWORKERS];

	/* sample used to compute the instant speed per ctx*/
	double ispeed_ctx_sample;

};

/* set a certain configuration to a context */
void sc_hypervisor_set_config(unsigned sched_ctx, void *config);

/* check out the configuration of a context */
struct sc_hypervisor_policy_config *sc_hypervisor_get_config(unsigned sched_ctx);

/* impose different parameters to a configuration of a context */
void sc_hypervisor_ctl(unsigned sched_ctx, ...);

#ifdef __cplusplus
}
#endif

#endif
