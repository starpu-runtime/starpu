/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* This example shows a basic StarPU vector scale app on top of StarPURM with a nVidia CUDA kernel */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu.h>
#include <starpurm.h>

static int rm_cpu_type_id = -1;
static int rm_nb_cpu_units = 0;

static void test1();
static void init_rm_infos(void);

static int global_nb_tasks = 100;
static const int nb_random_tests = 10;

/* vector scale codelet */
static void work_func(void *cl_buffers[], void *cl_arg)
{
	double timestamp = starpu_timing_now();
	double timestamp2;
	do
	{
		timestamp2 = starpu_timing_now();
	}
	while ((timestamp2 - timestamp) < 1e6);
}

static struct starpu_codelet work_cl =
{
	.cpu_funcs = {work_func},
};

/* main routines */
static void test1()
{
	int i;
	for (i=0; i<global_nb_tasks; i++)
	{
		int ret = starpu_task_insert(&work_cl,
				0);
		assert(ret == 0);
	}
	starpu_task_wait_for_all();
}

static void init_rm_infos(void)
{
	int cpu_type = starpurm_get_device_type_id("cpu");
	int nb_cpu_units = starpurm_get_nb_devices_by_type(cpu_type);
	if (nb_cpu_units < 1)
	{
		/* No CPU unit available. */
		exit(77);
	}

	rm_cpu_type_id = cpu_type;
	rm_nb_cpu_units = nb_cpu_units;
}

static void disp_selected_cpuset(void)
{
	hwloc_cpuset_t selected_cpuset = starpurm_get_selected_cpuset();
	int strl = hwloc_bitmap_snprintf(NULL, 0, selected_cpuset);
	char str[strl+1];
	hwloc_bitmap_snprintf(str, strl+1, selected_cpuset);
	printf("selected cpuset = %s\n", str);
}

int main(int argc, char *argv[])
{
	srandom(time(NULL));
	int drs_enabled;
	if (argc > 1)
	{
		global_nb_tasks = atoi(argv[1]);
	}
	starpurm_initialize();
	init_rm_infos();
	printf("using default units\n");
	disp_selected_cpuset();
	test1();

	if (rm_nb_cpu_units > 1)
	{
		const int nb_cpus = rm_nb_cpu_units;
		const int half_nb_cpus = nb_cpus/2;
		printf("nb_cpu_units = %d\n", nb_cpus);

		starpurm_set_drs_enable(NULL);
		drs_enabled = starpurm_drs_enabled_p();
		assert(drs_enabled != 0);

		printf("withdrawing %d cpus from StarPU\n", half_nb_cpus);
		starpurm_withdraw_cpus_from_starpu(NULL, half_nb_cpus);
		disp_selected_cpuset();
		test1();

		printf("assigning %d cpus to StarPU\n", half_nb_cpus);
		starpurm_assign_cpus_to_starpu(NULL, half_nb_cpus);
		disp_selected_cpuset();
		test1();

		int i;
		for (i=0; i<nb_random_tests; i++)
		{
			int some_cpus = 1+ random()%nb_cpus;
			printf("assigning exactly %d cpus to StarPU\n", some_cpus);
			starpurm_withdraw_all_cpus_from_starpu(NULL);
			starpurm_assign_cpus_to_starpu(NULL, some_cpus);
			disp_selected_cpuset();
			test1();
		}

		starpurm_set_drs_disable(NULL);
		drs_enabled = starpurm_drs_enabled_p();
		assert(drs_enabled == 0);
	}

	starpurm_shutdown();
	return 0;
}
