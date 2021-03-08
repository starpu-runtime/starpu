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

static void print_scope(const enum starpu_perf_counter_scope scope)
{
	int nb = starpu_perf_counter_nb(scope);
	int i;
	printf("scope %s\n", starpu_perf_counter_scope_id_to_name(scope));
	for (i=0; i<nb; i++)
	{
		const int id = starpu_perf_counter_nth_to_id(scope, i);
		const char *name = starpu_perf_counter_id_to_name(id);
		const char *help = starpu_perf_counter_get_help_string(id);
		int type_id = starpu_perf_counter_get_type_id(id);
		const char *type_name = starpu_perf_counter_type_id_to_name(type_id);
		printf("%d/%d - %s (0x%08x): [%s] / %s\n", i+1, nb, name, id, type_name, help);
	}
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	{
		int id;

		id = starpu_perf_counter_scope_name_to_id("global");
		STARPU_ASSERT(id == starpu_perf_counter_scope_global);
		
		id = starpu_perf_counter_scope_name_to_id("per_worker");
		STARPU_ASSERT(id == starpu_perf_counter_scope_per_worker);
		
		id = starpu_perf_counter_scope_name_to_id("per_codelet");
		STARPU_ASSERT(id == starpu_perf_counter_scope_per_codelet);

		(void)id;
	}

	{
		const char *name;
		
		name = starpu_perf_counter_scope_id_to_name(starpu_perf_counter_scope_global);
		STARPU_ASSERT(strcmp(name, "global") == 0);
		
		name = starpu_perf_counter_scope_id_to_name(starpu_perf_counter_scope_per_worker);
		STARPU_ASSERT(strcmp(name, "per_worker") == 0);
		
		name = starpu_perf_counter_scope_id_to_name(starpu_perf_counter_scope_per_codelet);
		STARPU_ASSERT(strcmp(name, "per_codelet") == 0);

		(void)name;
	}

	{
		int id;

		id = starpu_perf_counter_type_name_to_id("int32");
		STARPU_ASSERT(id == starpu_perf_counter_type_int32);

		id = starpu_perf_counter_type_name_to_id("int64");
		STARPU_ASSERT(id == starpu_perf_counter_type_int64);

		id = starpu_perf_counter_type_name_to_id("float");
		STARPU_ASSERT(id == starpu_perf_counter_type_float);

		id = starpu_perf_counter_type_name_to_id("double");
		STARPU_ASSERT(id == starpu_perf_counter_type_double);

		(void)id;
	}

	{
		const char *name;
		
		name = starpu_perf_counter_type_id_to_name(starpu_perf_counter_type_int32);
		STARPU_ASSERT(strcmp(name, "int32") == 0);
		
		name = starpu_perf_counter_type_id_to_name(starpu_perf_counter_type_int64);
		STARPU_ASSERT(strcmp(name, "int64") == 0);
		
		name = starpu_perf_counter_type_id_to_name(starpu_perf_counter_type_float);
		STARPU_ASSERT(strcmp(name, "float") == 0);
		
		name = starpu_perf_counter_type_id_to_name(starpu_perf_counter_type_double);
		STARPU_ASSERT(strcmp(name, "double") == 0);

		(void)name;
	}

	printf("programmatically get counters per scope\n");
	print_scope(starpu_perf_counter_scope_global);
	print_scope(starpu_perf_counter_scope_per_worker);
	print_scope(starpu_perf_counter_scope_per_codelet);
	printf("\n");

	printf("list available counters per scope\n");
	starpu_perf_counter_list_avail(starpu_perf_counter_scope_global);
	starpu_perf_counter_list_avail(starpu_perf_counter_scope_per_worker);
	starpu_perf_counter_list_avail(starpu_perf_counter_scope_per_codelet);
	printf("\n");

	printf("list all available counters\n");
	starpu_perf_counter_list_all_avail();
	printf("\n");

	starpu_shutdown();

	return 0;
}
