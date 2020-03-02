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

/* This checks that adding an end dependency for an already-terminated task
 * works */

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define INIT 12

void cpu_codelet2(void *descr[], void *args)
{
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(void)args;
	STARPU_ASSERT(*val == INIT);
	starpu_sleep(0.1);
	STARPU_ASSERT(*val == INIT);
	*val *= 2;
}

struct starpu_codelet cl2 =
{
	.cpu_funcs = {cpu_codelet2},
	.cpu_funcs_name = {"cpu_codelet2"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codelet2"
};

void cpu_codelet(void *descr[], void *args)
{
	(void)args;
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	STARPU_ASSERT(*val == 2*INIT);
	starpu_sleep(0.1);
	STARPU_ASSERT(*val == 2*INIT);
	*val *= 2;
}

struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_codelet},
	.cpu_funcs_name = {"cpu_codelet"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codelet"
};

int main(void)
{
        int value=INIT;
	int ret;
	starpu_data_handle_t value_handle;
	struct starpu_conf conf;
	struct starpu_task *task, *task2;

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

	task = starpu_task_build(&cl,
				 STARPU_RW, value_handle,
				 0);
	STARPU_ASSERT(task);
	task->detach = 0;

	task2 = starpu_task_build(&cl2,
				 STARPU_RW, value_handle,
				 0);
	STARPU_ASSERT(task2);
	task2->detach = 0;
	task2->destroy = 0;

	ret = starpu_task_submit(task2);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_wait(task2);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

	starpu_task_declare_end_deps(task, 1, task2);
	starpu_task_destroy(task2);

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

	starpu_data_unregister(value_handle);

	STARPU_ASSERT(value == 2*2*INIT);

        starpu_shutdown();

	FPRINTF(stderr, "Value = %d\n", value);

	return ret;
}
