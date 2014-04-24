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

#include <starpu.h>
#ifdef STARPU_OPENMP
#include <util/openmp_runtime_support.h>

#define __not_implemented__ do { fprintf (stderr, "omp lib function %s not implemented\n", __func__); abort(); } while (0)

void starpu_omp_set_num_threads(int threads)
{
	(void) threads;
	__not_implemented__;
}

int starpu_omp_get_num_threads()
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *region;
	if (task == NULL)
		return 1;

	region = task->owner_region;
	return region->nb_threads;
}

int starpu_omp_get_thread_num()
{
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *region;
	if (thread == NULL || task == NULL)
		return 0;

	region = task->owner_region;
	if (thread == region->master_thread)
		return 0;

	struct starpu_omp_thread * region_thread;
	int tid = 1;
	for (region_thread  = starpu_omp_thread_list_begin(region->thread_list);
			region_thread != starpu_omp_thread_list_end(region->thread_list);
			region_thread  = starpu_omp_thread_list_next(region_thread))
	{
		if (thread == region_thread)
		{
			return tid;
		}
		tid++;
	}
	_STARPU_ERROR("unrecognized omp thread\n");
}

int starpu_omp_get_max_threads()
{
	/* arbitrary limit */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_get_num_procs (void)
{
	/* starpu_cpu_worker_get_count defined as topology.ncpus */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_in_parallel (void)
{
	__not_implemented__;
}

void starpu_omp_set_dynamic (int dynamic_threads)
{
	(void) dynamic_threads;
	/* TODO: dynamic adjustment of the number of threads is not supported for now */
}

int starpu_omp_get_dynamic (void)
{
	/* TODO: dynamic adjustment of the number of threads is not supported for now 
	 * return false as required */
	return 0;
}

void starpu_omp_set_nested (int nested)
{
	(void) nested;
	/* TODO: nested parallelism not supported for now */
}

int starpu_omp_get_nested (void)
{
	/* TODO: nested parallelism not supported for now
	 * return false as required */
	return 0;
}

int starpu_omp_get_cancellation(void)
{
	/* TODO: cancellation not supported for now
	 * return false as required */
	return 0;
}

void starpu_omp_set_schedule (starpu_omp_sched_t kind, int modifier)
{
	(void) kind;
	(void) modifier;
	/* TODO: no starpu_omp scheduler scheme implemented for now */
	__not_implemented__;
	assert(kind >= 1 && kind <=4);
}

void starpu_omp_get_schedule (starpu_omp_sched_t *kind, int *modifier)
{
	(void) kind;
	(void) modifier;
	/* TODO: no starpu_omp scheduler scheme implemented for now */
	__not_implemented__;
}

int starpu_omp_get_thread_limit (void)
{
	/* arbitrary limit */
	return 1024;
}

void starpu_omp_set_max_active_levels (int max_levels)
{
	(void) max_levels;
	/* TODO: nested parallelism not supported for now */
}

int starpu_omp_get_max_active_levels (void)
{
	/* TODO: nested parallelism not supported for now
	 * assume a single level */
	return 1;
}

int starpu_omp_get_level (void)
{
	/* TODO: nested parallelism not supported for now
	 * assume a single level */
	return 1;
}

int starpu_omp_get_ancestor_thread_num (int level)
{
	if (level == 0) {
		return 0; /* spec required answer */
	}

	if (level == starpu_omp_get_level()) {
		return starpu_omp_get_thread_num(); /* spec required answer */
	}

	/* TODO: nested parallelism not supported for now
	 * assume ancestor is thread number '0' */
	return 0;
}

int starpu_omp_get_team_size (int level)
{
	if (level == 0) {
		return 1; /* spec required answer */
	}

	if (level == starpu_omp_get_level()) {
		return starpu_omp_get_num_threads(); /* spec required answer */
	}

	/* TODO: nested parallelism not supported for now
	 * assume the team size to be the number of cpu workers */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_get_active_level (void)
{
	/* TODO: nested parallelism not supported for now
	 * assume a single active level */
	return 1;
}

int starpu_omp_in_final(void)
{
	/* TODO: final not supported for now
	 * assume not in final */
	return 0;
}

starpu_omp_proc_bind_t starpu_omp_get_proc_bind(void)
{
	/* TODO: proc_bind not supported for now
	 * assumre false */
	return starpu_omp_proc_bind_false;
}

void starpu_omp_set_default_device(int device_num)
{
	(void) device_num;
	/* TODO: set_default_device not supported for now */
}

int starpu_omp_get_default_device(void)
{
	/* TODO: set_default_device not supported for now
	 * assume device 0 as default */
	return 0;
}

int starpu_omp_get_num_devices(void)
{
	/* TODO: get_num_devices not supported for now
	 * assume 1 device */
	return 1;
}

int starpu_omp_get_num_teams(void)
{
	/* TODO: num_teams not supported for now
	 * assume 1 team */
	return 1;
}

int starpu_omp_get_team_num(void)
{
	/* TODO: team_num not supported for now
	 * assume team_num 0 */
	return 0;
}

int starpu_omp_is_initial_device(void)
{
	/* TODO: is_initial_device not supported for now
	 * assume host device */
	return 1;
}


void starpu_omp_init_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_destroy_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_set_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_unset_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

int starpu_omp_test_lock (starpu_omp_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_init_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_destroy_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_set_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

void starpu_omp_unset_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

int starpu_omp_test_nest_lock (starpu_omp_nest_lock_t *lock)
{
	(void) lock;
	__not_implemented__;
}

double starpu_omp_get_wtime (void)
{
	return starpu_timing_now() - _starpu_omp_clock_ref;
}

double starpu_omp_get_wtick (void)
{
	/* arbitrary precision value */
	return 1e-6;
}
#endif /* STARPU_OPENMP */
