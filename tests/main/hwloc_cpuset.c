/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test workers hwloc cpusets
 */

int main(void)
{
	int status = 0;

#ifdef STARPU_HAVE_HWLOC
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.nmpi_ms = 0;

	int ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	int nworkers = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	if (nworkers != 0)
	{
		hwloc_cpuset_t accumulator_cpuset = hwloc_bitmap_alloc();
		hwloc_cpuset_t temp_cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_zero(accumulator_cpuset);
		status = 0;
		int workerids[nworkers];
		starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, workerids, nworkers);
		int i;
		for (i=0; i<nworkers; i++)
		{
			hwloc_cpuset_t cpuset = starpu_worker_get_hwloc_cpuset(workerids[i]);
			/* check that the worker cpuset is not empty */
			if (hwloc_bitmap_iszero(cpuset))
			{
				status = EXIT_FAILURE;
				hwloc_bitmap_free(cpuset);
				break;
			}
			hwloc_bitmap_zero(temp_cpuset);
			/* check that the worker cpuset does not overlap other workers cpusets */
			hwloc_bitmap_and(temp_cpuset, accumulator_cpuset, cpuset);
			if (!hwloc_bitmap_iszero(temp_cpuset))
			{
				status = EXIT_FAILURE;
				hwloc_bitmap_free(cpuset);
				break;
			}

			hwloc_bitmap_or(accumulator_cpuset, accumulator_cpuset, cpuset);

			/* the cpuset returned by starpu_worker_get_hwloc_cpuset() must be freed */
			hwloc_bitmap_free(cpuset);
		}
		hwloc_bitmap_free(temp_cpuset);
		hwloc_bitmap_free(accumulator_cpuset);
	}
	else
	{
		status = STARPU_TEST_SKIPPED;
	}

	starpu_shutdown();
#else
	status = STARPU_TEST_SKIPPED;
#endif

	return status;
}
