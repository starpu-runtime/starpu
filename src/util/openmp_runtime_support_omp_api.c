/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpu_omp_set_num_threads(int threads)
{
	STARPU_ASSERT(threads > 0);
	struct starpu_omp_task *task = _starpu_omp_get_task();
	STARPU_ASSERT(task != NULL);
	struct starpu_omp_region *region;
	region = task->owner_region;
	STARPU_ASSERT(region != NULL);
	region->icvs.nthreads_var[0] = threads;
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
	struct starpu_omp_task *task = _starpu_omp_get_task();
	if (task == NULL)
		return 0;
	return _starpu_omp_get_region_thread_num(task->owner_region);
}

int starpu_omp_get_max_threads()
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	int max_threads = parallel_region->icvs.nthreads_var[0];
	/* TODO: for now, nested parallel sections are not supported, thus we
	 * open an active parallel section only if the generating region is the
	 * initial region */
	if (parallel_region->level > 0)
	{
		max_threads = 1;
	}

	return max_threads;
}

int starpu_omp_get_num_procs (void)
{
	/* starpu_cpu_worker_get_count defined as topology.ncpus */
	return starpu_cpu_worker_get_count();
}

int starpu_omp_in_parallel (void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.active_levels_var > 0;
}

void starpu_omp_set_dynamic (int dynamic_threads)
{
	(void) dynamic_threads;
	/* TODO: dynamic adjustment of the number of threads is not supported for now */
}

int starpu_omp_get_dynamic (void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.dyn_var;
}

void starpu_omp_set_nested (int nested)
{
	(void) nested;
	/* TODO: nested parallelism not supported for now */
}

int starpu_omp_get_nested (void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.nest_var;
}

int starpu_omp_get_cancellation(void)
{
	return _starpu_omp_global_state->icvs.cancel_var;
}

void starpu_omp_set_schedule (enum starpu_omp_sched_value kind, int modifier)
{
	struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	STARPU_ASSERT(     kind == starpu_omp_sched_static
			|| kind == starpu_omp_sched_dynamic
			|| kind == starpu_omp_sched_guided
			|| kind == starpu_omp_sched_auto);
	STARPU_ASSERT(modifier >= 0);
	parallel_region->icvs.run_sched_var = kind;
	parallel_region->icvs.run_sched_chunk_var = (unsigned long long)modifier;
}

void starpu_omp_get_schedule (enum starpu_omp_sched_value *kind, int *modifier)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	*kind = parallel_region->icvs.run_sched_var;
	*modifier = (int)parallel_region->icvs.run_sched_chunk_var;
}

int starpu_omp_get_thread_limit (void)
{
	return starpu_cpu_worker_get_count();
}

void starpu_omp_set_max_active_levels (int max_levels)
{
	struct starpu_omp_device * const device = _starpu_omp_get_task()->owner_region->owner_device;
	if (max_levels > 1)
	{
		/* TODO: nested parallelism not supported for now */
		max_levels = 1;
	}
	device->icvs.max_active_levels_var = max_levels;
}

int starpu_omp_get_max_active_levels (void)
{
	const struct starpu_omp_device * const device = _starpu_omp_get_task()->owner_region->owner_device;
	return device->icvs.max_active_levels_var;
}

int starpu_omp_get_level (void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.levels_var;
}

int starpu_omp_get_ancestor_thread_num (int level)
{
	struct starpu_omp_region *parallel_region;

	if (level == 0)
		return 0;

	parallel_region = _starpu_omp_get_region_at_level(level);
	if (!parallel_region)
		return -1;

	return _starpu_omp_get_region_thread_num(parallel_region);
}

int starpu_omp_get_team_size (int level)
{
	struct starpu_omp_region *parallel_region;

	if (level == 0)
		return 1;

	parallel_region = _starpu_omp_get_region_at_level(level);
	if (!parallel_region)
		return -1;

	return parallel_region->nb_threads;
}

int starpu_omp_get_active_level (void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.active_levels_var;
}

int starpu_omp_in_final(void)
{
	const struct starpu_omp_task *task = _starpu_omp_get_task();
	return task->flags & STARPU_OMP_TASK_FLAGS_FINAL;
}

enum starpu_omp_proc_bind_value starpu_omp_get_proc_bind(void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	int proc_bind = parallel_region->icvs.bind_var[0];
	return proc_bind;
}

int starpu_omp_get_num_places(void)
{
	struct starpu_omp_place *places = &_starpu_omp_initial_icv_values->places;
	return places->nb_numeric_places;
}

int starpu_omp_get_place_num_procs(int place_num)
{
	(void) place_num;
	/* TODO */
	return 0;
}

void starpu_omp_get_place_proc_ids(int place_num, int *ids)
{
	(void) place_num;
	(void) ids;
	/* TODO */
}

int starpu_omp_get_place_num(void)
{
	/* TODO */
	return -1;
}

int starpu_omp_get_partition_num_places(void)
{
	/* TODO */
	return 0;
}

void starpu_omp_get_partition_place_nums(int *place_nums)
{
	(void) place_nums;
	/* TODO */
}

void starpu_omp_set_default_device(int device_num)
{
	(void) device_num;
	/* TODO: set_default_device not supported for now */
}

int starpu_omp_get_default_device(void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.default_device_var;
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
	struct starpu_omp_task *task = _starpu_omp_get_task();
	if (!task)
		return 0;
	const struct starpu_omp_device * const device = task->owner_region->owner_device;
	return device == _starpu_omp_global_state->initial_device;
}

int starpu_omp_get_initial_device(void)
{
	/* Assume only one device for now. */
	return 0;
}

int starpu_omp_get_max_task_priority(void)
{
	const struct starpu_omp_region * const parallel_region = _starpu_omp_get_task()->owner_region;
	return parallel_region->icvs.max_task_priority_var;
}

double starpu_omp_get_wtime (void)
{
	return 1e-6 * (starpu_timing_now() - _starpu_omp_clock_ref);
}

double starpu_omp_get_wtick (void)
{
	/* arbitrary precision value */
	return 1e-6;
}
#endif /* STARPU_OPENMP */
