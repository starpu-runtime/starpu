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

/* This example shows a basic StarPU vector scale app on top of StarPURM */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu.h>
#include <starpurm.h>

static int rm_cpu_type_id = -1;
static int rm_nb_cpu_units = 0;

static void usage(void);
static void test1(const int N);
static void test2(const int N, const int task_mult);
static void init_rm_infos(void);

/* vector scale codelet */
static void vector_scale_func(void *cl_buffers[], void *cl_arg)
{
	double scalar = -1.0;
	int n = STARPU_VECTOR_GET_NX(cl_buffers[0]);
	double *vector = (double *)STARPU_VECTOR_GET_PTR(cl_buffers[0]);
	int i;
	starpu_codelet_unpack_args(cl_arg, &scalar);

	int workerid = starpu_worker_get_id();
	hwloc_cpuset_t worker_cpuset = starpu_worker_get_hwloc_cpuset(workerid);
	hwloc_cpuset_t check_cpuset = starpurm_get_selected_cpuset();
	{
		int strl1 = hwloc_bitmap_snprintf(NULL, 0, worker_cpuset);
		char str1[strl1+1];
		hwloc_bitmap_snprintf(str1, strl1+1, worker_cpuset);
		int strl2 = hwloc_bitmap_snprintf(NULL, 0, check_cpuset);
		char str2[strl2+1];
		hwloc_bitmap_snprintf(str2, strl2+1, check_cpuset);
		printf("worker[%03d] - task: vector=%p, n=%d, scalar=%lf, worker cpuset = %s, selected cpuset = %s\n", workerid, vector, n, scalar, str1, str2);
	}
	hwloc_bitmap_and(check_cpuset, check_cpuset, worker_cpuset);
	assert(!hwloc_bitmap_iszero(check_cpuset));
	hwloc_bitmap_free(check_cpuset);
	hwloc_bitmap_free(worker_cpuset);

	for (i = 0; i < n; i++)
	{
		vector[i] *= scalar;
	}
}

static struct starpu_codelet vector_scale_cl =
{
	.cpu_funcs = {vector_scale_func},
	.nbuffers = 1
};

/* main routines */
static void usage(void)
{
	fprintf(stderr, "usage: 05_vector_scale [VECTOR_SIZE]\n");
	exit(1);
}

static void test1(const int N)
{
	double *vector = NULL;
	const double scalar = 2.0;
	starpu_data_handle_t vector_handle;
	int ret;
	
	vector = malloc(N * sizeof(*vector));
	{
		int i;
		for (i = 0; i < N; i++)
		{
			vector[i] = i;
		}
	}
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, N, sizeof(*vector));

	ret = starpu_task_insert(&vector_scale_cl, 
			STARPU_RW, vector_handle,
			STARPU_VALUE, &scalar, sizeof(scalar),
			0);
	assert(ret == 0);
	starpu_task_wait_for_all();

	starpu_data_unregister(vector_handle);
	{
		int i;
		for (i = 0; i < N; i++)
		{
			double d_i = i;
			if (vector[i] != d_i*scalar)
			{
				fprintf(stderr, "%s: check_failed\n", __func__);
				exit(1);
			}
		}
	}
	free(vector);
}

static void test2(const int N, const int task_mult)
{
	double *vector = NULL;
	const double scalar = 3.0;
	starpu_data_handle_t vector_handle;
	int ret;
	
	vector = malloc(N * sizeof(*vector));
	{
		int i;
		for (i = 0; i < N; i++)
		{
			vector[i] = i;
		}
	}
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, N, sizeof(*vector));
	struct starpu_data_filter partition_filter =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = rm_nb_cpu_units * task_mult
	};

	starpu_data_partition(vector_handle, &partition_filter);

	{
		int i;
		for (i = 0; i < rm_nb_cpu_units*task_mult; i++)
		{
			starpu_data_handle_t sub_vector_handle = starpu_data_get_sub_data(vector_handle, 1, i);
			ret = starpu_task_insert(&vector_scale_cl, 
					STARPU_RW, sub_vector_handle,
					STARPU_VALUE, &scalar, sizeof(scalar),
					0);
			assert(ret == 0);
		}
	}
	starpu_task_wait_for_all();
	starpu_data_unpartition(vector_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(vector_handle);
	{
		int i;
		for (i = 0; i < N; i++)
		{
			double d_i = i;
			if (vector[i] != d_i*scalar)
			{
				fprintf(stderr, "%s: check_failed\n", __func__);
				exit(1);
			}
		}
	}
	free(vector);
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

int main(int argc, char *argv[])
{
	int param_N = 1000000;
	int drs_enabled;
	if (argc > 1)
	{
		param_N = atoi(argv[1]);
		if (param_N < 1)
		{
			usage();
		}
	}

	starpurm_initialize();
	init_rm_infos();
	test1(param_N);
	test2(param_N, 1);
	test2(param_N, 10);
	test2(param_N, 100);

	if (rm_nb_cpu_units > 1)
	{
		const int half_nb_cpus = rm_nb_cpu_units/2;
		starpurm_set_drs_enable(NULL);
		drs_enabled = starpurm_drs_enabled_p();
		assert(drs_enabled != 0);

		printf("withdrawing %d cpus from StarPU\n", half_nb_cpus);
		starpurm_withdraw_cpus_from_starpu(NULL, half_nb_cpus);
		test2(param_N, 1);
		test2(param_N, 10);
		test2(param_N, 100);

		printf("assigning %d cpus to StarPU\n", half_nb_cpus);
		starpurm_assign_cpus_to_starpu(NULL, half_nb_cpus);
		test2(param_N, 1);
		test2(param_N, 10);
		test2(param_N, 100);

		int i;
		for (i = rm_nb_cpu_units-1; i > 0; i--)
		{
			starpurm_set_max_parallelism(NULL, i);
			test2(param_N, 10);
		}

		printf("withdrawing all cpus from StarPU\n");
		starpurm_withdraw_all_cpus_from_starpu(NULL);
		printf("assigning %d cpus to StarPU\n", rm_nb_cpu_units);
		starpurm_assign_cpus_to_starpu(NULL, rm_nb_cpu_units);
		test2(param_N, 1);
		test2(param_N, 10);
		test2(param_N, 100);

		starpurm_set_drs_disable(NULL);
		drs_enabled = starpurm_drs_enabled_p();
		assert(drs_enabled == 0);
	}

	starpurm_shutdown();
	return 0;
}
