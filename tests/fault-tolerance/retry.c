/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This tests the fault tolerance interface: it submits a tasks which repeatedly
 * fails until being eventually successful
 */

#include <starpu.h>
#include "../helper.h"

/* This task fakes some repeated errors	 */
static int retry;
void cpu_increment(void *descr[], void *arg)
{
	(void)arg;
	unsigned *var = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *var2 = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	FPRINTF(stderr,"computing\n");
	*var2 = *var + 1;
	if (retry < 10)
	{
		FPRINTF(stderr,"failing\n");
		retry++;
		/* Fake failure */
		starpu_task_ft_failed(starpu_task_get_current());
	}
	else
		FPRINTF(stderr,"succeed\n");
}

static struct starpu_codelet my_codelet =
{
	.cpu_funcs = {cpu_increment},
	//.cpu_funcs_name = {"cpu_increment"},
	.modes = { STARPU_R, STARPU_W },
	.nbuffers = 2
};

/* This implements the retry strategy
 * (Identical to the default implementation: just retry) */
static void check_ft(void *arg)
{
	struct starpu_task *meta_task = arg;
	struct starpu_task *current_task = starpu_task_get_current();
	struct starpu_task *new_task;
	int ret;

	if (!current_task->failed)
	{
		FPRINTF(stderr,"didn't fail, release main task\n");
		starpu_task_ft_success(meta_task);
		return;
	}

	FPRINTF(stderr,"failed, try again\n");

	new_task = starpu_task_ft_create_retry(meta_task, current_task, check_ft);

	/* Here we could e.g. force the task to use only a CPU implementation
	 * known to be failsafe */

	ret = starpu_task_submit_nodeps(new_task);
	STARPU_ASSERT(!ret);
}

int main(void)
{
	int x = 12;
	int y = 1;
	starpu_data_handle_t h_x, h_y;
	int ret, ret1;

	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		/* TODO _submit_job_take_data_deps */
		return STARPU_TEST_SKIPPED;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&h_x, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&h_y, STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(y));

	retry = 0;
	ret1 = starpu_task_insert(&my_codelet,
				  STARPU_PROLOGUE_CALLBACK, starpu_task_ft_prologue,
				  STARPU_PROLOGUE_CALLBACK_ARG_NFREE, check_ft,
				  STARPU_R, h_x,
				  STARPU_W, h_y,
				  0);
	if (ret1 != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret1, "starpu_task_insert");
	starpu_task_wait_for_all();

	starpu_data_unregister(h_x);
	starpu_data_unregister(h_y);

	starpu_shutdown();

	if (x != 12)
		ret = 1;
	FPRINTF(stderr, "Value x = %d (expected 12)\n", x);

	if (ret1 != -ENODEV)
	{
		if (y != 13)
			ret = 1;
		FPRINTF(stderr, "Value y = %d (expected 13)\n", y);
	}

	STARPU_RETURN(ret);
}
