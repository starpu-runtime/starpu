/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/workers.h>

void _starpu_sched_find_worker_combinations(struct starpu_machine_topology_s *topology)
{
//#ifdef STARPU_HAVE_HWLOC
//#error TODO !
//#else
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	/* We put the id of all CPU workers in this array */
	int cpu_workers[STARPU_NMAXWORKERS];
	unsigned ncpus = 0;

	unsigned i;
	for (i = 0; i < topology->nworkers; i++)
	{
		if (config->workers[i].perf_arch == STARPU_CPU_DEFAULT)
			cpu_workers[ncpus++] = i;
	}
	
	unsigned size;
	for (size = 2; size <= ncpus; size *= 2)
	{
		unsigned first_cpu;
		for (first_cpu = 0; first_cpu < ncpus; first_cpu += size)
		{
			if (first_cpu + size <= ncpus)
			{
				int workerids[size];

				for (i = 0; i < size; i++)
					workerids[i] = cpu_workers[first_cpu + i];

				/* We register this combination */
				int ret;
				ret = starpu_combined_worker_assign_workerid(size, workerids); 
				STARPU_ASSERT(ret >= 0);
			}
		}
	}
//#endif
}
