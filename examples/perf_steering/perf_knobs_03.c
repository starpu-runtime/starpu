/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <assert.h>
#include <string.h>

#define NTASKS 100

volatile int task_count[2];

void cpu_func(void *buffer[], void *cl_arg)
{
	(void)buffer;
	(void)cl_arg;
	int workerid = starpu_worker_get_id();
	STARPU_ASSERT(workerid == 0 || workerid == 1);
	task_count[workerid]++;
}

int main(int argc, char **argv)
{
	int ret;

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = 2;
	{
		const char *sched_pol_name = starpu_getenv("STARPU_SCHED");
		if (sched_pol_name != NULL && strcmp(sched_pol_name, "prio") != 0)
		{
			fprintf(stderr, "example uses 'prio' scheduling policy.\n");
			return 77;
		}
	}

	conf.sched_policy_name = "prio";

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() != 2
		|| starpu_cuda_worker_get_count() != 0
		|| starpu_opencl_worker_get_count() != 0
		|| starpu_mpi_ms_worker_get_count() != 0)
	{
		starpu_shutdown();
		fprintf(stderr, "example needs exactly two cpu cores.\n");
		return 77;
	}

	{
		const char * const max_prio_knob_name       = "starpu.task.s_max_priority_cap_knob";
		const char * const min_prio_knob_name       = "starpu.task.s_min_priority_cap_knob";
		const char * const knob_scope_name = "per_scheduler";
		const char * const knob_type_name  = "int32";
		int32_t max_prio_val;
		int32_t min_prio_val;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);

		const int max_prio_id = starpu_perf_knob_name_to_id(scope_id, max_prio_knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(max_prio_id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		const int min_prio_id = starpu_perf_knob_name_to_id(scope_id, min_prio_knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(min_prio_id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		printf("%s:\n", max_prio_knob_name);
		max_prio_val = starpu_perf_knob_get_per_scheduler_int32_value(max_prio_id, "prio");
		printf("- %d\n", max_prio_val);

		printf("%s:\n", min_prio_knob_name);
		min_prio_val = starpu_perf_knob_get_per_scheduler_int32_value(min_prio_id, "prio");
		printf("- %d\n", min_prio_val);
		STARPU_ASSERT (max_prio_val >= min_prio_val);

		if (min_prio_val > 0)
		{
			starpu_perf_knob_set_per_scheduler_int32_value(min_prio_id, "prio", 0);
			starpu_perf_knob_set_per_scheduler_int32_value(max_prio_id, "prio", 0);
		}
		else
		{
			starpu_perf_knob_set_per_scheduler_int32_value(max_prio_id, "prio", 0);
			starpu_perf_knob_set_per_scheduler_int32_value(min_prio_id, "prio", 0);
		}

		printf("%s:\n", max_prio_knob_name);
		max_prio_val = starpu_perf_knob_get_per_scheduler_int32_value(max_prio_id, "prio");
		printf("- %d\n", max_prio_val);

		printf("%s:\n", min_prio_knob_name);
		min_prio_val = starpu_perf_knob_get_per_scheduler_int32_value(min_prio_id, "prio");
		printf("- %d\n", min_prio_val);
		STARPU_ASSERT (max_prio_val == 0);
		STARPU_ASSERT (min_prio_val == 0);

	}

	{
		const char * const knob_name       = "starpu.worker.w_enable_worker_knob";
		const char * const knob_scope_name = "per_worker";
		const char * const knob_type_name  = "int32";
		int32_t val;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);
		const int id = starpu_perf_knob_name_to_id(scope_id, knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		struct starpu_codelet cl =
		{
			.cpu_funcs = {cpu_func}
		};

		task_count[0] = 0;
		task_count[1] = 0;

		val = starpu_perf_knob_get_per_worker_int32_value(id, 0);
		STARPU_ASSERT(val == 1);
		val = starpu_perf_knob_get_per_worker_int32_value(id, 1);
		STARPU_ASSERT(val == 1);

		starpu_perf_knob_set_per_worker_int32_value(id, 1, 0);
		val = starpu_perf_knob_get_per_worker_int32_value(id, 1);
		STARPU_ASSERT(val == 0);

		int i;
		for (i=0; i<NTASKS; i++)
		{
			ret = starpu_task_insert(&cl, 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		starpu_task_wait_for_all();
		STARPU_ASSERT(task_count[0] == NTASKS);
		STARPU_ASSERT(task_count[1] == 0);

		task_count[0] = 0;

		starpu_perf_knob_set_per_worker_int32_value(id, 1, 1);
		val = starpu_perf_knob_get_per_worker_int32_value(id, 1);
		STARPU_ASSERT(val == 1);

		starpu_perf_knob_set_per_worker_int32_value(id, 0, 0);
		val = starpu_perf_knob_get_per_worker_int32_value(id, 0);
		STARPU_ASSERT(val == 0);

		for (i=0; i<NTASKS; i++)
		{
			ret = starpu_task_insert(&cl, 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
		starpu_task_wait_for_all();
		STARPU_ASSERT(task_count[0] == 0);
		STARPU_ASSERT(task_count[1] == NTASKS);

		starpu_perf_knob_set_per_worker_int32_value(id, 0, 1);
		val = starpu_perf_knob_get_per_worker_int32_value(id, 0);
		STARPU_ASSERT(val == 1);
	}

	starpu_shutdown();

	return 0;
}
