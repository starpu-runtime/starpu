/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2024-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdlib.h>
#include <unistd.h>

#include <starpu.h>
#include "../helper.h"

/*
 * This tests the hardware energy counters monitoring feature (see \ref
 * HardwareEnergyMonitoring in the documentation). It enables the feature
 * through the STARPU_ENERGY_READER environment variable, forces a short
 * sampling interval so that at least a few samples get taken while the
 * CPU workers are busy, and then runs enough tasks for some time to let
 * the sampling logic run in the driver loop.
 */

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	usleep(1000);
}

static struct starpu_codelet dummy_codelet =
    {
	.cpu_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.nbuffers = 0};

int main(void)
{
	int ret;
	unsigned i;

	setenv("STARPU_ENERGY_READER", "1", 1);
	setenv("STARPU_ENERGY_PKG_INTERVAL", "10", 1);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned ntasks = 50 * starpu_cpu_worker_get_count();
	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_insert(&dummy_codelet, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();

	starpu_shutdown();

	return EXIT_SUCCESS;
}
