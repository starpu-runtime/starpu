/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <starpu.h>
#include "../helper.h"

/*
 * Test binding the main thread to its dedicated core, making one less CPU core
 * available to StarPU.
 */

int main(void)
{
	int ret;
	struct starpu_conf conf;
	int ncpus;
	unsigned active_bindid;
	unsigned passive_bindid1;
	unsigned passive_bindid2;

	/* First get the number of cores */
	starpu_conf_init(&conf);
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ncpus = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	starpu_shutdown();

	/* Check we have enough of them */
	if (ncpus <= 2) return STARPU_TEST_SKIPPED;

	/* Now re-initialize with two cores less */
	starpu_conf_init(&conf);
	conf.reserve_ncpus = 2;
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Make sure StarPU uses two core less */
	STARPU_ASSERT_MSG(starpu_worker_get_count_by_type(STARPU_CPU_WORKER) == ncpus-2, "Expected %d CPUs, got %d\n", ncpus-2, starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
	FPRINTF(stderr, "CPUS: %d as expected\n", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));

	/* Check we can grab a whole core */
	active_bindid = starpu_get_next_bindid(STARPU_THREAD_ACTIVE, NULL, 0);
	starpu_bind_thread_on(active_bindid, STARPU_THREAD_ACTIVE, "main");

	/* Check we can request for an additional shared core */
	passive_bindid1 = starpu_get_next_bindid(0, NULL, 0);
	passive_bindid2 = starpu_get_next_bindid(0, NULL, 0);
	STARPU_ASSERT(passive_bindid1 != active_bindid);
	STARPU_ASSERT(passive_bindid1 == passive_bindid2);
	starpu_bind_thread_on(passive_bindid1, 0, "main");
	starpu_bind_thread_on(passive_bindid2, 0, "main");

	starpu_shutdown();

	return EXIT_SUCCESS;
}
