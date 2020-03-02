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
#include <starpu.h>
#include <starpurm.h>

static void disp_cpuset(const char * name, hwloc_cpuset_t cpuset)
{
	int strl = hwloc_bitmap_snprintf(NULL, 0, cpuset);
	char str[strl+1];
	hwloc_bitmap_snprintf(str, strl+1, cpuset);
	printf(". %s: %s\n", name, str);
}

int main(int argc, char *argv[])
{
	starpurm_initialize();
	int cpu_id = starpurm_get_device_type_id("cpu");
	const int nb_cpu_units = starpurm_get_nb_devices_by_type(cpu_id);
	if (nb_cpu_units < 1)
	{
		starpurm_shutdown();
		return 77;
	}
	hwloc_cpuset_t cpuset;
	cpuset = starpurm_get_device_worker_cpuset(cpu_id, 0);
	disp_cpuset("worker cpuset", cpuset);
	hwloc_bitmap_free(cpuset);

	cpuset = starpurm_get_global_cpuset();
	disp_cpuset("global cpuset", cpuset);
	hwloc_bitmap_free(cpuset);

	cpuset = starpurm_get_selected_cpuset();
	disp_cpuset("selected cpuset", cpuset);
	hwloc_bitmap_free(cpuset);

	cpuset = starpurm_get_all_cpu_workers_cpuset();
	disp_cpuset("all cpu workers cpuset", cpuset);
	hwloc_bitmap_free(cpuset);

	cpuset = starpurm_get_all_device_workers_cpuset();
	disp_cpuset("all device workers cpuset", cpuset);
	hwloc_bitmap_free(cpuset);

	starpurm_shutdown();

	return 0;
}
