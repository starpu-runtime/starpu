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
#include "../helper.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * Check the OpenMP API getters return proper default results.
 */

#if !defined(STARPU_OPENMP)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
__attribute__((constructor))
static void omp_constructor(void)
{
	int ret;
	/* we clear the whole OMP environment for this test, to check the
	 * default behaviour of API functions */
	unsetenv("OMP_DYNAMIC");
	unsetenv("OMP_NESTED");
	unsetenv("OMP_SCHEDULE");
	unsetenv("OMP_STACKSIZE");
	unsetenv("OMP_WAIT_POLICY");
	unsetenv("OMP_THREAD_LIMIT");
	unsetenv("OMP_MAX_ACTIVE_LEVELS");
	unsetenv("OMP_CANCELLATION");
	unsetenv("OMP_DEFAULT_DEVICE");
	unsetenv("OMP_MAX_TASK_PRIORITY");
	unsetenv("OMP_PROC_BIND");
	unsetenv("OMP_NUM_THREADS");
	unsetenv("OMP_PLACES");
	unsetenv("OMP_DISPLAY_ENV");
	ret = starpu_omp_init();
	if (ret == -EINVAL) exit(STARPU_TEST_SKIPPED);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
}

__attribute__((destructor))
static void omp_destructor(void)
{
	starpu_omp_shutdown();
}

#define check_omp_func(f,_tv)					\
{								\
	const int v = (f());					\
	const int tv = (_tv);					\
	printf(#f ": %d (should be %d)\n", v, tv);		\
	STARPU_ASSERT(v == tv);					\
}

const char * get_sched_name(int sched_value)
{
	const char *sched_name = NULL;

	switch (sched_value)
	{
		case starpu_omp_sched_undefined: sched_name = "<undefined>"; break;
		case starpu_omp_sched_static:    sched_name = "static"; break;
		case starpu_omp_sched_dynamic:   sched_name = "dynamic"; break;
		case starpu_omp_sched_guided:    sched_name = "guided"; break;
		case starpu_omp_sched_auto:      sched_name = "auto"; break;
		case starpu_omp_sched_runtime:   sched_name = "runtime"; break;
		default: _STARPU_ERROR("invalid omp schedule value");
	}
	return sched_name;
}

int
main (void)
{
	const int nb_cpus = starpu_cpu_worker_get_count();

	check_omp_func(starpu_omp_get_num_threads, 1);
	check_omp_func(starpu_omp_get_thread_num, 0);
	/* since OMP_NUM_THREADS is cleared, starpu_omp_get_max_threads() should return nb_cpus */
	check_omp_func(starpu_omp_get_max_threads, nb_cpus);
	check_omp_func(starpu_omp_get_num_procs, nb_cpus);
	check_omp_func(starpu_omp_in_parallel, 0);
	check_omp_func(starpu_omp_get_dynamic, 0);
	check_omp_func(starpu_omp_get_nested, 0);
	check_omp_func(starpu_omp_get_cancellation, 0);
	{
		const enum starpu_omp_sched_value target_kind = starpu_omp_sched_static;
		const int target_modifier = 0;
		enum starpu_omp_sched_value kind;
		int modifier;
		const char *sched_name;
		const char *target_sched_name;
		starpu_omp_get_schedule(&kind, &modifier);
		sched_name = get_sched_name(kind);
		target_sched_name = get_sched_name(target_kind);
		printf("starpu_omp_get_schedule: %s,%d (should be %s,%d)\n", sched_name, modifier, target_sched_name, target_modifier);
		STARPU_ASSERT(kind == target_kind && modifier == target_modifier);
	}
	check_omp_func(starpu_omp_get_thread_limit, nb_cpus);
	check_omp_func(starpu_omp_get_max_active_levels, 1);
	check_omp_func(starpu_omp_get_level, 0);
	{
		const int tv = 0;
		const int v = starpu_omp_get_ancestor_thread_num(0);
		printf("starpu_omp_get_ancestor_thread_num(0): %d (should be %d)\n", v, tv);
		STARPU_ASSERT(v == tv);
	}
	{
		const int tv = 1;
		const int v = starpu_omp_get_team_size(0);
		printf("starpu_omp_get_team_size(0): %d (should be %d)\n", v, tv);
		STARPU_ASSERT(v == tv);
	}
	check_omp_func(starpu_omp_get_active_level, 0);
	check_omp_func(starpu_omp_in_final, 0);
	check_omp_func(starpu_omp_get_proc_bind, starpu_omp_proc_bind_false);
	check_omp_func(starpu_omp_get_default_device, 0);
	/* TODO: support more than one device */
	check_omp_func(starpu_omp_get_num_devices, 1);
	check_omp_func(starpu_omp_get_num_teams, 1);
	check_omp_func(starpu_omp_get_team_num, 0);
	check_omp_func(starpu_omp_is_initial_device, 1);
	check_omp_func(starpu_omp_get_initial_device, 0);
	check_omp_func(starpu_omp_get_max_task_priority, 0);
	return 0;
}
#endif
