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

#ifdef STARPU_NON_BLOCKING_DRIVERS
int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	return 77;
}
#else
static int rm_cpu_type_id = -1;
static int rm_nb_cpu_units = 0;

#if defined (STARPU_QUICK_CHECK)
static int global_nb_tasks_1 = 20;
static const int nb_random_tests_1 = 5;
static int global_nb_tasks_2 = 10;
static const int nb_random_tests_2 = 2;
#elif defined (STARPU_LONG_CHECK)
static int global_nb_tasks_1 = 200;
static const int nb_random_tests_1 = 20;
static int global_nb_tasks_2 = 100;
static const int nb_random_tests_2 = 10;
#else
static int global_nb_tasks_1 = 50;
static const int nb_random_tests_1 = 5;
static int global_nb_tasks_2 = 10;
static const int nb_random_tests_2 = 8;
#endif

/* vector scale codelet */
static void work_func(void *cl_buffers[], void *cl_arg)
{
	(void)cl_buffers;
	(void)cl_arg;

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
static void test_1()
{
	int i;
	for (i=0; i<global_nb_tasks_1; i++)
	{
		int ret = starpu_task_insert(&work_cl,
				0);
		assert(ret == 0);
	}
	starpu_task_wait_for_all();
}

static void test_2()
{
	int i;
	for (i=0; i<global_nb_tasks_2; i++)
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

static hwloc_cpuset_t gen_random_cpuset(void)
{
	const int nb_cpus = rm_nb_cpu_units;
	hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
	do
	{
		hwloc_bitmap_clr_range(cpuset, 0, -1);
		int j;
		for (j=0; j<nb_cpus; j++)
		{
			if (random()%2) {
				hwloc_bitmap_set(cpuset, j);
			}
		}
	}
	while(hwloc_bitmap_iszero(cpuset));
	return cpuset;
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
		global_nb_tasks_1 = atoi(argv[1]);
		if (argc > 2)
		{
			global_nb_tasks_2 = atoi(argv[2]);
		}
		else
		{
			global_nb_tasks_2 = global_nb_tasks_1 / 10;
			if (global_nb_tasks_2 < 1)
			{
				global_nb_tasks_2 = 1;
			}
		}
	}
	starpurm_initialize();
	init_rm_infos();
	printf("using default units\n");
	disp_selected_cpuset();
	test_1();

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
		test_1();

		printf("assigning %d cpus to StarPU\n", half_nb_cpus);
		starpurm_assign_cpus_to_starpu(NULL, half_nb_cpus);
		disp_selected_cpuset();
		test_1();

		int i;
		for (i=0; i<nb_random_tests_1; i++)
		{
			int some_cpus = 1+ random()%nb_cpus;
			printf("assigning exactly %d cpus to StarPU\n", some_cpus);
			starpurm_withdraw_all_cpus_from_starpu(NULL);
			starpurm_assign_cpus_to_starpu(NULL, some_cpus);
			disp_selected_cpuset();
			test_1();
		}

		for (i=0; i<nb_random_tests_1; i++)
		{
			int some_cpus = random()%nb_cpus;
			starpurm_assign_all_cpus_to_starpu(NULL);
			printf("withdrawing exactly %d cpus from StarPU\n", some_cpus);
			starpurm_withdraw_cpus_from_starpu(NULL, some_cpus);
			disp_selected_cpuset();
			test_1();
		}

		for (i=0; i<nb_random_tests_2; i++)
		{
			int some_cpuid = random()%nb_cpus;
			printf("assigning cpuid %d to StarPU\n", some_cpuid);
			starpurm_withdraw_all_cpus_from_starpu(NULL);
			starpurm_assign_cpu_to_starpu(NULL, some_cpuid);
			disp_selected_cpuset();
			test_2();
		}

		for (i=0; i<nb_random_tests_2; i++)
		{
			int some_cpuid = random()%nb_cpus;
			starpurm_assign_all_cpus_to_starpu(NULL);
			printf("withdrawing cpuid %d from StarPU\n", some_cpuid);
			starpurm_withdraw_cpu_from_starpu(NULL, some_cpuid);
			disp_selected_cpuset();
			test_2();
		}

		for (i=0; i<nb_random_tests_2; i++)
		{
			hwloc_cpuset_t some_cpu_mask = gen_random_cpuset();

			{
				int strl = hwloc_bitmap_snprintf(NULL, 0, some_cpu_mask);
				char str[strl+1];
				hwloc_bitmap_snprintf(str, strl+1, some_cpu_mask);
				printf("assigning cpu mask %s to StarPU\n", str);
			}
			starpurm_withdraw_all_cpus_from_starpu(NULL);
			starpurm_assign_cpu_mask_to_starpu(NULL, some_cpu_mask);
			disp_selected_cpuset();
			test_2();
			hwloc_bitmap_free(some_cpu_mask);
		}

		for (i=0; i<nb_random_tests_2; i++)
		{
			hwloc_cpuset_t some_cpu_mask = gen_random_cpuset();

			{
				int strl = hwloc_bitmap_snprintf(NULL, 0, some_cpu_mask);
				char str[strl+1];
				hwloc_bitmap_snprintf(str, strl+1, some_cpu_mask);
				printf("withdrawing cpu mask %s from StarPU\n", str);
			}
			starpurm_assign_all_cpus_to_starpu(NULL);
			starpurm_withdraw_cpu_mask_from_starpu(NULL, some_cpu_mask);
			disp_selected_cpuset();
			test_2();
			hwloc_bitmap_free(some_cpu_mask);
		}

		starpurm_set_drs_disable(NULL);
		drs_enabled = starpurm_drs_enabled_p();
		assert(drs_enabled == 0);
	}

	starpurm_shutdown();
	return 0;
}
#endif
