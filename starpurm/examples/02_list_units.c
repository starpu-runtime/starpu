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

/* This example lists the CPU and device units detected and managed by
 * StarPURM. */

#include <stdio.h>
#include <starpurm.h>

int main(int argc, char *argv[])
{
	int ids[4];
	int i;
	starpurm_initialize();
	ids[0] = starpurm_get_device_type_id("cpu");
	ids[1] = starpurm_get_device_type_id("opencl");
	ids[2] = starpurm_get_device_type_id("cuda");
	ids[3] = starpurm_get_device_type_id("mic");

	for (i=0; i<4; i++)
	{
		const int id = ids[i];
		if (id == -1)
			continue;
		const int nb_units = starpurm_get_nb_devices_by_type(id);
		printf("%s: %d units\n", starpurm_get_device_type_name(id), nb_units);
		int j;
		for (j=0; j<nb_units; j++)
		{
			hwloc_cpuset_t cpuset = starpurm_get_device_worker_cpuset(id, j);
			int strl = hwloc_bitmap_snprintf(NULL, 0, cpuset);
			char str[strl+1];
			hwloc_bitmap_snprintf(str, strl+1, cpuset);
			printf(". %d: %s\n", j, str);
			hwloc_bitmap_free(cpuset);
		}
	}
	starpurm_shutdown();
	return 0;
}
