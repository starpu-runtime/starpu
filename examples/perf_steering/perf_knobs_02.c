/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	{
		const char * const knob_name       = "starpu.global.g_calibrate_knob";
		const char * const knob_scope_name = "global";
		const char * const knob_type_name  = "int32";
		int32_t val, val_save;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);
		const int id = starpu_perf_knob_name_to_id(scope_id, knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		printf("%s:\n", knob_name);

		val_save = val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);

		starpu_perf_knob_set_global_int32_value(id, 1);
		val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);
		STARPU_ASSERT(val == 1);

		starpu_perf_knob_set_global_int32_value(id, 0);
		val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);
		STARPU_ASSERT(val == 0);

		starpu_perf_knob_set_global_int32_value(id, val_save);
		val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);
		STARPU_ASSERT(val == val_save);
	}

	{
		const char * const knob_name       = "starpu.global.g_enable_catch_signal_knob";
		const char * const knob_scope_name = "global";
		const char * const knob_type_name  = "int32";
		int32_t val, val_save;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);
		const int id = starpu_perf_knob_name_to_id(scope_id, knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		printf("%s:\n", knob_name);

		val_save = val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);

		starpu_perf_knob_set_global_int32_value(id, 1);
		val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);
		STARPU_ASSERT(val == 1);

		starpu_perf_knob_set_global_int32_value(id, 0);
		val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);
		STARPU_ASSERT(val == 0);

		starpu_perf_knob_set_global_int32_value(id, val_save);
		val = starpu_perf_knob_get_global_int32_value(id);
		printf("- %d\n", val);
		STARPU_ASSERT(val == val_save);
	}


	{
		const char * const knob_name       = "starpu.worker.w_bind_to_pu_knob";
		const char * const knob_scope_name = "per_worker";
		const char * const knob_type_name  = "int32";
		int32_t val;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);
		const int id = starpu_perf_knob_name_to_id(scope_id, knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		printf("%s:\n", knob_name);

		unsigned int ncpu  = starpu_cpu_worker_get_count();
		unsigned int i;
		for (i=0; i<ncpu; i++)
		{
			val = starpu_perf_knob_get_per_worker_int32_value(id, i);
			STARPU_ASSERT(val >= 0);
			printf("- %u: %d\n", i, val);
		}
	}

	{
		const char * const knob_name       = "starpu.task.s_max_priority_cap_knob";
		const char * const knob_scope_name = "per_scheduler";
		const char * const knob_type_name  = "int32";
		int32_t val;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);
		const int id = starpu_perf_knob_name_to_id(scope_id, knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		printf("%s:\n", knob_name);
		val = starpu_perf_knob_get_per_scheduler_int32_value(id, "prio");
		printf("- %d\n", val);
	}

	{
		const char * const knob_name       = "starpu.task.s_min_priority_cap_knob";
		const char * const knob_scope_name = "per_scheduler";
		const char * const knob_type_name  = "int32";
		int32_t val;

		const int scope_id = starpu_perf_knob_scope_name_to_id(knob_scope_name);
		const int id = starpu_perf_knob_name_to_id(scope_id, knob_name);
		STARPU_ASSERT(starpu_perf_knob_get_type_id(id) == starpu_perf_knob_type_name_to_id(knob_type_name));

		printf("%s:\n", knob_name);
		val = starpu_perf_knob_get_per_scheduler_int32_value(id, "prio");
		printf("- %d\n", val);
	}


	starpu_shutdown();

	return 0;
}
