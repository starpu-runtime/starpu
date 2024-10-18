/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	(void)argc;
	(void)argv;
	int type_ids[3];
	int type;
	starpurm_initialize();
	type_ids[0] = starpurm_device_type_name_to_id("cpu");
	type_ids[1] = starpurm_device_type_name_to_id("opencl");
	type_ids[2] = starpurm_device_type_name_to_id("cuda");

	for (type=0; type<3; type++)
	{
		const int type_id = type_ids[type];
		if (type_id == -1)
			continue;
		const int nb_devices = starpurm_get_nb_devices_by_type(type_id);
		printf("%s: %d devices\n", starpurm_device_type_id_to_name(type_id), nb_devices);
		int rank;
		for (rank=0; rank<nb_devices; rank++)
		{
			int device_id = starpurm_get_device_id_from_rank(type_id, rank);
			int worker_devid = starpurm_get_device_worker_devid(device_id);
			hwloc_cpuset_t cpuset = starpurm_get_device_worker_cpuset(device_id);
			int strl = hwloc_bitmap_snprintf(NULL, 0, cpuset);
			char str[strl+1];
			hwloc_bitmap_snprintf(str, strl+1, cpuset);
			printf(". %d (worker devid %d): %s\n", rank, worker_devid, str);
			hwloc_bitmap_free(cpuset);
		}
	}
	starpurm_shutdown();
	return 0;
}
