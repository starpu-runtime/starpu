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
 * Arbitrary limit on the number of nested parallel sections
 */
#define STARPU_OMP_MAX_ACTIVE_LEVELS 4

/*
 * Proc bind modes defined by the OpenMP spec
 */
enum starpu_omp_bind_mode
{
	starpu_omp_bind_false  = 0,
	starpu_omp_bind_true   = 1,
	starpu_omp_bind_master = 2,
	starpu_omp_bind_close  = 3,
	starpu_omp_bind_spread = 4
};

enum starpu_omp_schedule_mode
{
	starpu_omp_schedule_static  = 0,
	starpu_omp_schedule_dynamic = 1,
	starpu_omp_schedule_guided  = 2,
	starpu_omp_schedule_auto    = 3
};

/*
 * Possible abstract names for OpenMP places
 */
enum starpu_omp_place_name
{
	starpu_omp_place_undefined = 0,
	starpu_omp_place_threads   = 1,
	starpu_omp_place_cores     = 2,
	starpu_omp_place_sockets   = 3,
	starpu_omp_place_numerical = 4 /* place specified numerically */
};

struct starpu_omp_numeric_place
{
	int excluded_place;
	int *included_numeric_items;
	int nb_included_numeric_items;
	int *excluded_numeric_items;
	int nb_excluded_numeric_items;
};

/*
 * OpenMP place for thread afinity, defined by the OpenMP spec
 */
struct starpu_omp_place
{
	int abstract_name;
	int abstract_excluded;
	int abstract_length;
	struct starpu_omp_numeric_place *numeric_places;
	int nb_numeric_places;
};

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
	int run_sched_chunk_var;
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

	/* not a real ICV, but needed to store the contents of OMP_PLACES */
	struct starpu_omp_place places;
};

extern struct starpu_omp_global_icvs *starpu_omp_global_icvs;
extern struct starpu_omp_initial_icv_values *starpu_omp_initial_icv_values;
#endif // STARPU_OPENMP

#endif // __OPENMP_RUNTIME_SUPPORT_H__
