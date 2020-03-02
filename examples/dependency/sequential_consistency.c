/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_codeletA(void *descr[], void *args);
void cpu_codeletB(void *descr[], void *args);
void cpu_codeletC(void *descr[], void *args);

struct starpu_codelet clA =
{
	.cpu_funcs = {cpu_codeletA},
	.cpu_funcs_name = {"cpu_codeletA"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codeletA"
};

struct starpu_codelet clB =
{
	.cpu_funcs = {cpu_codeletB},
	.cpu_funcs_name = {"cpu_codeletB"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codeletB"
};

struct starpu_codelet clC =
{
	.cpu_funcs = {cpu_codeletC},
	.cpu_funcs_name = {"cpu_codeletC"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codeletC"
};

void cpu_codeletA(void *descr[], void *args)
{
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	starpu_data_handle_t value_handle;
	starpu_tag_t tagHoldC;
	int ret;
	unsigned char handle_sequential_consistency[] = {0};

	FPRINTF(stderr, "[Task A] Value = %d\n", *val);

	starpu_codelet_unpack_args(args, &value_handle, &tagHoldC);

	// With several data, one would need to use a dynamically
	// allocated array for the sequential consistency,
	// the array could be freed immediately after calling
	// starpu_task_insert()

	ret = starpu_task_insert(&clB,
				 STARPU_RW, value_handle,
				 STARPU_CALLBACK_WITH_ARG_NFREE, starpu_tag_notify_from_apps, tagHoldC,
				 STARPU_HANDLES_SEQUENTIAL_CONSISTENCY, handle_sequential_consistency,
				 STARPU_NAME, "taskB",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	*val *= 2;
}

void cpu_codeletB(void *descr[], void *args)
{
	(void)args;
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	FPRINTF(stderr, "[Task B] Value = %d\n", *val);
	STARPU_ASSERT_MSG(*val == 24, "Incorrect value %d (expected 24)\n", *val);
	*val += 1;
}

void cpu_codeletC(void *descr[], void *args)
{
	(void)args;
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	FPRINTF(stderr, "[Task C] Value = %d\n", *val);
	STARPU_ASSERT_MSG(*val == 25, "Incorrect value %d (expected 25)\n", *val);
	*val *= 2;
}

/*
 * Submit taskA and hold it
 * Submit taskC and hold it
 * Release taskA
 * Execute taskA       --> submit taskB
 * Execute taskB       --> callback: release taskC
 *
 * All three tasks use the same data in RW, taskB is submitted after
 * taskC, so taskB should normally only execute after taskC but as the
 * sequential consistency for (taskB, data) is unset, taskB can
 * execute straightaway
 */
int main(void)
{
        int value=12;
	int ret;
	starpu_data_handle_t value_handle;
	starpu_tag_t tagHoldA = 42;
	starpu_tag_t tagHoldC = 84;
	starpu_tag_t tagA = 421;
	starpu_tag_t tagC = 842;

	struct starpu_conf conf;

	if (sizeof(starpu_tag_t) > sizeof(void*))
	{
		// Can't pass a tag_t through callback arg :/
		return 77;
	}

	starpu_conf_init(&conf);
	conf.nmic = 0;
	conf.nmpi_ms = 0;

        ret = starpu_init(&conf);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		return 77;
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() < 1)
	{
		FPRINTF(stderr, "This application requires at least 1 cpu worker\n");
		starpu_shutdown();
		return 77;
	}

	starpu_variable_data_register(&value_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));

	starpu_tag_declare_deps_array(tagA, 1, &tagHoldA);
	starpu_tag_declare_deps_array(tagC, 1, &tagHoldC);

	ret = starpu_task_insert(&clA,
				 STARPU_TAG, tagA,
				 STARPU_RW, value_handle,
				 STARPU_VALUE, &value_handle, sizeof(starpu_data_handle_t),
				 STARPU_VALUE, &tagHoldC, sizeof(starpu_tag_t),
				 STARPU_NAME, "taskA",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_insert(&clC,
				 STARPU_TAG, tagC,
				 STARPU_RW, value_handle,
				 STARPU_NAME, "taskC",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// Release taskA (we want to make sure it will execute after taskC has been submitted)
	starpu_tag_notify_from_apps(tagHoldA);

	starpu_data_unregister(value_handle);

	STARPU_ASSERT_MSG(value == 50, "Incorrect value %d (expected 50)\n", value);

	starpu_shutdown();

	FPRINTF(stderr, "Value = %d\n", value);

	return ret;
}
