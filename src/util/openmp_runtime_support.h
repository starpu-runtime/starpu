/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  Inria
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

#ifndef __OPENMP_RUNTIME_SUPPORT_H__
#define __OPENMP_RUNTIME_SUPPORT_H__

#include <starpu.h>

#ifdef STARPU_OPENMP

/* 
 * Internal Control Variables (ICVs) declared following
 * OpenMP 4.0.0 spec section 2.3.1
 */
struct starpu_omp_data_environment_icvs
{
	/* parallel region icvs */
	int dyn_var;
	int nest_var;
	int *nthreads_var; /* nthreads_var ICV is a list */
	int thread_limit_var;

	int active_levels_var;
	int levels_var;
	int *bind_var; /* bind_var ICV is a list */

	/* loop region icvs */
	int run_sched_var;

	/* program execution icvs */
	int default_device_var;
};

struct starpu_omp_device_icvs
{
	/* parallel region icvs */
	int max_active_levels_var;

	/* loop region icvs */
	int def_sched_var;

	/* program execution icvs */
	int stacksize_var;
	int wait_policy_var;
};

struct starpu_omp_implicit_task_icvs
{
	/* parallel region icvs */
	int place_partition_var;
};

struct starpu_omp_global_icvs
{
	/* program execution icvs */
	int cancel_var;
};

struct starpu_omp_initial_icv_values
{
	int dyn_var;
	int nest_var;
	int *nthreads_var;
	int run_sched_var;
	int def_sched_var;
	int *bind_var;
	int stacksize_var;
	int wait_policy_var;
	int thread_limit_var;
	int max_active_levels_var;
	int active_levels_var;
	int levels_var;
	int place_partition_var;
	int cancel_var;
	int default_device_var;
};

extern struct starpu_omp_global_icvs *starpu_omp_global_icvs;
extern struct starpu_omp_initial_icv_values *starpu_omp_initial_icv_values;
#endif // STARPU_OPENMP

#endif // __OPENMP_RUNTIME_SUPPORT_H__
