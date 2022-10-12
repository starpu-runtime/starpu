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
static int rm_cuda_type_id = -1;
static int rm_nb_cpu_units = 0;
static int rm_nb_cuda_units = 0;

static void usage(void);
static void test1(const int N);
static void test2(const int N, const int task_mult);
static void init_rm_infos(void);

/* vector scale codelet */
static void vector_scale_func(void *cl_buffers[], void *cl_arg)
{
	float scalar = -1.0;
	int n = STARPU_VECTOR_GET_NX(cl_buffers[0]);
	float *vector = (float *)STARPU_VECTOR_GET_PTR(cl_buffers[0]);
	int i;
	starpu_codelet_unpack_args(cl_arg, &scalar);

	{
		int workerid = starpu_worker_get_id();
		hwloc_cpuset_t worker_cpuset = starpu_worker_get_hwloc_cpuset(workerid);
		hwloc_cpuset_t check_cpuset = starpurm_get_selected_cpuset();
#if 0
		{
			int strl1 = hwloc_bitmap_snprintf(NULL, 0, worker_cpuset);
			char str1[strl1+1];
			hwloc_bitmap_snprintf(str1, strl1+1, worker_cpuset);
			int strl2 = hwloc_bitmap_snprintf(NULL, 0, check_cpuset);
			char str2[strl2+1];
			hwloc_bitmap_snprintf(str2, strl2+1, check_cpuset);
			printf("worker[%03d] - task: vector=%p, n=%d, scalar=%lf, worker cpuset = %s, selected cpuset = %s\n", workerid, vector, n, scalar, str1, str2);
		}
#endif
		hwloc_bitmap_and(check_cpuset, check_cpuset, worker_cpuset);
		assert(!hwloc_bitmap_iszero(check_cpuset));
		hwloc_bitmap_free(check_cpuset);
		hwloc_bitmap_free(worker_cpuset);
	}

	for (i = 0; i < n; i++)
	{
		vector[i] *= scalar;
	}
}

extern void vector_scale_cuda_func(void *cl_buffers[], void *cl_arg);

static struct starpu_codelet vector_scale_cl =
{
	.cpu_funcs = {vector_scale_func},
	.cuda_funcs = {vector_scale_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
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
	float *vector = NULL;
	const float scalar = 2.0;
	starpu_data_handle_t vector_handle;
	int ret;

	starpu_malloc((void **)&vector, N * sizeof(*vector));
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
			float d_i = i;
			if (vector[i] != d_i*scalar)
			{
				fprintf(stderr, "%s: check_failed, vector[%d]: %f != %f\n", __func__, i, vector[i], d_i*scalar);
				exit(1);
			}
		}
	}
	starpu_free_noflag(vector, N * sizeof(*vector));
}

static void test2(const int N, const int task_mult)
{
	float *vector = NULL;
	const float scalar = 3.0;
	starpu_data_handle_t vector_handle;
	int ret;

	starpu_malloc((void **)&vector, N * sizeof(*vector));
	{
		int i;
		for (i = 0; i < N; i++)
		{
			vector[i] = i;
		}
	}
	const int nparts = (rm_nb_cpu_units+rm_nb_cuda_units) * task_mult;
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, N, sizeof(*vector));
	struct starpu_data_filter partition_filter =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = nparts
	};

	starpu_data_partition(vector_handle, &partition_filter);

	{
		int i;
		for (i = 0; i < nparts; i++)
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
			float d_i = i;
			if (vector[i] != d_i*scalar)
			{
				fprintf(stderr, "%s: check_failed, vector[%d]: %f != %f\n", __func__, i, vector[i], d_i*scalar);
				exit(1);
			}
		}
	}
	starpu_free_noflag(vector, N * sizeof(*vector));
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

	int cuda_type = starpurm_get_device_type_id("cuda");
	int nb_cuda_units = starpurm_get_nb_devices_by_type(cuda_type);
	if (nb_cuda_units < 1)
	{
		/* No CUDA unit available. */
		exit(77);
	}

	rm_cpu_type_id = cpu_type;
	rm_cuda_type_id = cuda_type;
	rm_nb_cpu_units = nb_cpu_units;
	rm_nb_cuda_units = nb_cuda_units;
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
	printf("using default units\n");
	disp_selected_cpuset();
	test1(param_N);
	test2(param_N, 1);
	test2(param_N, 10);
	test2(param_N, 100);

	if (rm_nb_cpu_units > 1 && rm_nb_cuda_units > 1)
	{
		int nb_cpus = rm_nb_cpu_units;
		const int nb_cudas = rm_nb_cuda_units;
		const int cuda_type = rm_cuda_type_id;
		printf("nb_cpu_units = %d\n", nb_cpus);
		printf("nb_cuda_units = %d\n", nb_cudas);

		/* Keep at least one CPU core */
		nb_cpus--;

		starpurm_set_drs_enable(NULL);
		drs_enabled = starpurm_drs_enabled_p();
		assert(drs_enabled != 0);

		printf("withdrawing %d cpus from StarPU\n", nb_cpus);
		starpurm_withdraw_cpus_from_starpu(NULL, nb_cpus);
		disp_selected_cpuset();
		test2(param_N, 1);
		test2(param_N, 10);
		test2(param_N, 100);

		printf("assigning %d cpus to StarPU\n", nb_cpus);
		starpurm_assign_cpus_to_starpu(NULL, nb_cpus);
		disp_selected_cpuset();
		test2(param_N, 1);
		test2(param_N, 10);
		test2(param_N, 100);

		printf("withdrawing %d cuda devices from StarPU\n", nb_cudas);
		starpurm_withdraw_devices_from_starpu(NULL, cuda_type, nb_cudas);
		disp_selected_cpuset();
		test2(param_N, 1);
		test2(param_N, 10);
		test2(param_N, 100);

		printf("lending %d cuda devices to StarPU\n", nb_cudas);
		starpurm_assign_devices_to_starpu(NULL, cuda_type, nb_cudas);
		disp_selected_cpuset();
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
