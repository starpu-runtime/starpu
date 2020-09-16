/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __LOAD_BALANCER_POLICY_H__
#define __LOAD_BALANCER_POLICY_H__

#include <starpu_mpi_lb.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

/** A load balancer consists in a collection of operations on a data
 * representing the load of the application (in terms of computation, memory,
 * whatever). StarPU allows several entry points for the user. The load
 * balancer allows the user to give its load balancing methods to be used on
 * these entry points of the runtime system. */
struct load_balancer_policy
{
	int (*init)(struct starpu_mpi_lb_conf *);
	int (*deinit)();
	void (*submitted_task_entry_point)();
	void (*finished_task_entry_point)();

	/** Name of the load balancing policy. The selection of the load balancer is
	 * performed through the use of the STARPU_MPI_LB=name environment
	 * variable.
	 */
	const char *policy_name;
};

extern struct load_balancer_policy load_heat_propagation_policy;

#ifdef __cplusplus
}
#endif

#endif // __LOAD_BALANCER_POLICY_H__
