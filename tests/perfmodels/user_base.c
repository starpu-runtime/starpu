/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_scheduler.h>
#include <unistd.h>
#include "../helper.h"

/*
 * Test using a user-provided base for the perfmodel
 */

void func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	usleep(1000);
}

size_t get_size_base(struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	return 3;
};

uint32_t get_footprint(struct starpu_task *task)
{
	uint32_t orig = starpu_task_data_footprint(task);
	return starpu_hash_crc32c_be(42, orig);
};

static struct starpu_perfmodel rb_model =
{
	.type = STARPU_REGRESSION_BASED,
	.symbol = "user_base_valid_model_regression_based",
	.size_base = get_size_base,
};

static struct starpu_perfmodel nlrb_model =
{
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "user_base_valid_model_non_linear_regression_based",
	.size_base = get_size_base,
};

static struct starpu_perfmodel hb_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "user_base_valid_model_history_based",
	.size_base = get_size_base,
};

static struct starpu_perfmodel hb_model_foot =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "user_base_valid_model_history_based_footprint",
	.footprint = get_footprint,
};

static struct starpu_codelet mycodelet =
{
	.cuda_funcs = {func},
	.opencl_funcs = {func},
	.cpu_funcs = {func},
	.cpu_funcs_name = {"func"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static int submit(struct starpu_codelet *codelet, struct starpu_perfmodel *model)
{
	int nloops = 123;
	int loop;
	starpu_data_handle_t handle;
	int ret;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.sched_policy_name = "eager";
	conf.calibrate = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	codelet->model = model;

        starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, 100, sizeof(int));
	for (loop = 0; loop < nloops; loop++)
	{
		ret = starpu_task_insert(codelet, STARPU_W, handle, 0);
		if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
        starpu_data_unregister(handle);
	starpu_shutdown();
	return EXIT_SUCCESS;
}

int main(void)
{
	int ret;

	/* Use a linear regression model */
	ret = submit(&mycodelet, &rb_model);
	if (ret) return ret;

	/* Use a non-linear regression model */
	ret = submit(&mycodelet, &nlrb_model);
	if (ret) return ret;

	/* Use a history model */
	ret = submit(&mycodelet, &hb_model);
	if (ret) return ret;

	/* Use a history model with footprints*/
	ret = submit(&mycodelet, &hb_model_foot);
	if (ret) return ret;

	return EXIT_SUCCESS;
}
