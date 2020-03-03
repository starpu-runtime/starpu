/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

/*
 * Test the starpu_perfmodel_update_history function
 */

static struct starpu_perfmodel model =
{
	.type = STARPU_REGRESSION_BASED,
	.symbol = "feed"
};

static struct starpu_perfmodel nl_model =
{
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "nlfeed"
};

static struct starpu_codelet cl =
{
	.model = &model,
	.nbuffers = 1,
	.modes = {STARPU_W}
};

int main(void)
{
	struct starpu_task task;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	 if (starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) < 2)
	 {
		 starpu_shutdown();
		 return STARPU_TEST_SKIPPED;
	 }

	starpu_task_init(&task);
	task.cl = &cl;

	int size;
	for (size = 1024; size < 16777216; size *= 2)
	{
		float measured_fast, measured_slow;
		starpu_data_handle_t handle;
		starpu_vector_data_register(&handle, -1, 0, size, sizeof(float));
		task.handles[0] = handle;

		/* Simulate Fast GPU. In real applications this would be
		 * replaced by fetching from actual measurement */
		measured_fast = 0.002+size*0.00000001;
		measured_slow = 0.001+size*0.0000001;

		struct starpu_perfmodel_arch arch;
		arch.ndevices = 1;
		arch.devices = (struct starpu_perfmodel_device*)malloc(sizeof(struct starpu_perfmodel_device));
		arch.devices[0].type = STARPU_CUDA_WORKER;
		arch.devices[0].ncores = 0;
		/* Simulate Fast GPU */
		arch.devices[0].devid = 0;
		starpu_perfmodel_update_history(&model, &task, &arch, 0, 0, measured_fast);
		starpu_perfmodel_update_history(&nl_model, &task, &arch, 0, 0, measured_fast);

		/* Simulate Slow GPU */
		arch.devices[0].devid = 1;
		starpu_perfmodel_update_history(&model, &task, &arch, 0, 0, measured_slow);
		starpu_perfmodel_update_history(&nl_model, &task, &arch, 0, 0, measured_slow);
		starpu_task_clean(&task);
		starpu_data_unregister(handle);
	}

	starpu_shutdown();

	return EXIT_SUCCESS;
}
