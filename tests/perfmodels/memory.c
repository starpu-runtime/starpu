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
#include <core/perfmodel/perfmodel.h>
#include "../helper.h"

/*
 * Test providing the memory perfmodel function
 */

void func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_perfmodel my_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "my_model",
};

static struct starpu_codelet my_codelet =
{
	.cpu_funcs = {func},
	.cpu_funcs_name = {"func"},
	.model = &my_model
};

double cuda_cost_function(struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	t;
	a;
	return (double)i;
}

int main(void)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_perfmodel_init(&my_model);
	starpu_perfmodel_set_per_devices_cost_function(&my_model, 0, cuda_cost_function, STARPU_CUDA_WORKER, 0, 1, -1);

	ret = starpu_task_insert(&my_codelet, 0);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();
	starpu_shutdown();

	return EXIT_SUCCESS;
}
