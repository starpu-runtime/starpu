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

#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <float.h>
#include <mkl.h>
#include <morse.h>
#include <starpurm.h>
#include <hwloc.h>
#include <pthread.h>

#define CHECK

static int rm_cpu_type_id = -1;
static int rm_cuda_type_id = -1;
static int rm_nb_cpu_units = 0;
static int rm_nb_cuda_units = 0;
static const int nb_random_tests = 10;

static unsigned spawn_pending = 0;
static pthread_mutex_t spawn_pending_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t spawn_pending_cond;

static void _inc_spawn_pending(void)
{
	pthread_mutex_lock(&spawn_pending_mutex);
	assert(spawn_pending < UINT_MAX);
	spawn_pending++;
	pthread_mutex_unlock(&spawn_pending_mutex);
}

static void _dec_spawn_pending(void)
{
	pthread_mutex_lock(&spawn_pending_mutex);
	assert(spawn_pending > 0);
	spawn_pending--;
	if (spawn_pending == 0)
		pthread_cond_broadcast(&spawn_pending_cond);
	pthread_mutex_unlock(&spawn_pending_mutex);
}

static void _wait_pending_spawns(void)
{
	pthread_mutex_lock(&spawn_pending_mutex);
	while (spawn_pending > 0)
		pthread_cond_wait(&spawn_pending_cond, &spawn_pending_mutex);
	pthread_mutex_unlock(&spawn_pending_mutex);
}

static void spawn_callback(void *_arg)
{
	assert(42 == (uintptr_t)_arg);
	_dec_spawn_pending();
}

static void usage(void)
{
	fprintf(stderr, "dgemm: M N K <trans_A=T|N> <trans_B=[T|N]>\n");
	exit(EXIT_FAILURE);
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

	rm_cpu_type_id = cpu_type;
	rm_cuda_type_id = cuda_type;
	rm_nb_cpu_units = nb_cpu_units;
	rm_nb_cuda_units = nb_cuda_units;
}


static void disp_cpuset(hwloc_cpuset_t selected_cpuset)
{
	//hwloc_cpuset_t selected_cpuset = starpurm_get_selected_cpuset();
	int strl = hwloc_bitmap_snprintf(NULL, 0, selected_cpuset);
	char str[strl+1];
	hwloc_bitmap_snprintf(str, strl+1, selected_cpuset);
	printf("%llx: selected cpuset = %s\n", (unsigned long long)pthread_self(), str);
}

struct s_test_args
{
	const int m;
	const int n;
	const int k;
	int transA;
	int transB;
};

static void test(void *_args)
{
	struct s_test_args *args = _args;
	const int m = args->m;
	const int n = args->n;
	const int k = args->k;
	int transA = args->transA;
	int transB = args->transB;
	unsigned rand_seed = (unsigned)time(NULL);
	double *A = malloc(m * k * sizeof(double));
	double *B = malloc(k * n * sizeof(double));
	double *C = calloc(m * n, sizeof(double));
	double *C_test = calloc(m * n, sizeof(double));

	const double alpha = (double)rand_r(&rand_seed) / ((double)rand_r(&rand_seed) + DBL_MIN);
	const double beta  = (double)rand_r(&rand_seed) / ((double)rand_r(&rand_seed) + DBL_MIN);

	int i;
	for (i = 0; i < m; i++)
	{
		int j;
		for (j = 0; j < n; j++)
		{
			A[i*n+j] = (double)rand_r(&rand_seed) / ((double)rand_r(&rand_seed) + DBL_MIN);
			B[i*n+j] = (double)rand_r(&rand_seed) / ((double)rand_r(&rand_seed) + DBL_MIN);
		}
	}

	MORSE_dgemm(transA, transB, m, n, k, alpha, A, k, B, n, beta, C, n);
#ifdef CHECK
	/* Check */
	cblas_dgemm(CblasColMajor,
		    (CBLAS_TRANSPOSE) transA,
		    (CBLAS_TRANSPOSE) transB,
		    m, n, k,
		    alpha, A, k,
		    B, n,
		    beta, C_test, n);

	double C_test_inorm = LAPACKE_dlange(CblasColMajor, 'I', m, n, C_test, n);
	cblas_daxpy(m*n, -1, C, 1, C_test, 1);
	double inorm = LAPACKE_dlange(CblasColMajor, 'I', m, n, C_test, n);
	printf("%llx: ||C_test-C||_I / ||C_test||_I = %e\n", (unsigned long long)pthread_self(), inorm/C_test_inorm);
#endif
	free(A);
	free(B);
	free(C);
	free(C_test);
}

static void select_units(hwloc_cpuset_t selected_cpuset, hwloc_cpuset_t available_cpuset, int offset, int nb)
{
	int first_idx = hwloc_bitmap_first(available_cpuset);
	int last_idx = hwloc_bitmap_last(available_cpuset);
	int count = 0;
	int idx = first_idx;
	while (idx != -1 && idx <= last_idx && count < offset+nb)
	{
		if (hwloc_bitmap_isset(available_cpuset, idx))
		{
			if (count >= offset)
			{
				hwloc_bitmap_set(selected_cpuset, idx);
			}
			count ++;
		}
		idx = hwloc_bitmap_next(available_cpuset, idx);
	}
	assert(count == offset+nb);
}

void spawn_tests(int cpu_offset, int cpu_nb, int cuda_offset, int cuda_nb, void *args)
{
	if (cpu_offset + cpu_nb > rm_nb_cpu_units)
		exit(77);
	if (cuda_offset + cuda_nb > rm_nb_cuda_units)
		exit(77);
	hwloc_cpuset_t cpu_cpuset = starpurm_get_all_cpu_workers_cpuset();
	hwloc_cpuset_t cuda_cpuset = starpurm_get_all_device_workers_cpuset_by_type(rm_cuda_type_id);
	hwloc_cpuset_t sel_cpuset = hwloc_bitmap_alloc();
	assert(sel_cpuset != NULL);

	select_units(sel_cpuset, cpu_cpuset, cpu_offset, cpu_nb);
	select_units(sel_cpuset, cuda_cpuset, cuda_offset, cuda_nb);

	{
		int strl1 = hwloc_bitmap_snprintf(NULL, 0, cpu_cpuset);
		char str1[strl1+1];
		hwloc_bitmap_snprintf(str1, strl1+1, cpu_cpuset);

		int strl2 = hwloc_bitmap_snprintf(NULL, 0, cuda_cpuset);
		char str2[strl2+1];
		hwloc_bitmap_snprintf(str2, strl2+1, cuda_cpuset);
		printf("all cpus cpuset = %s\n", str1);

		int strl3 = hwloc_bitmap_snprintf(NULL, 0, sel_cpuset);
		char str3[strl3+1];
		hwloc_bitmap_snprintf(str3, strl1+3, sel_cpuset);
		printf("spawn on selected cpuset = %s (avail cpu %s, avail cuda %s)\n", str3, str1, str2);
	}

	_inc_spawn_pending();
	starpurm_spawn_kernel_on_cpus_callback(NULL, test, args, sel_cpuset, spawn_callback, (void*)(uintptr_t)42);

	hwloc_bitmap_free(sel_cpuset);
	hwloc_bitmap_free(cpu_cpuset);
	hwloc_bitmap_free(cuda_cpuset);
}

int main(int argc, char const *argv[])
{
	pthread_cond_init(&spawn_pending_cond, NULL);

	int transA = MorseTrans;
	int transB = MorseTrans;

	if (argc < 6 || argc > 6)
		usage();

	int m = atoi(argv[1]);
	if (m < 1)
		usage();
	int n = atoi(argv[2]);
	if (n < 1)
		usage();
	int k = atoi(argv[3]);
	if (k < 1)
		usage();

	if (strcmp(argv[4], "T") == 0)
		transA = MorseTrans;
	else if (strcmp(argv[4], "N") == 0)
		transA = MorseNoTrans;
	else
		usage();

	if (strcmp(argv[5], "T") == 0)
		transB = MorseTrans;
	else if (strcmp(argv[5], "N") == 0)
		transB = MorseNoTrans;
	else
		usage();

	srand(time(NULL));

	struct s_test_args test_args = { .m = m, .n = n, .k = k, .transA = transA, .transB = transB };

	/* Test case */
	starpurm_initialize();
	starpurm_set_drs_enable(NULL);
	init_rm_infos();
	printf("cpu units: %d\n", rm_nb_cpu_units);
	printf("cuda units: %d\n", rm_nb_cuda_units);
	printf("using default units\n");
	disp_cpuset(starpurm_get_selected_cpuset());

	MORSE_Init(rm_nb_cpu_units, rm_nb_cuda_units);
	test(&test_args);
	{
		int cpu_offset = 0;
		int cpu_nb = rm_nb_cpu_units/2;
		if (cpu_nb == 0 && rm_nb_cpu_units > 0)
		{
			cpu_nb = 1;
		}
		int cuda_offset = 0;
		int cuda_nb = rm_nb_cuda_units/2;
		if (cuda_nb == 0 && rm_nb_cuda_units > 0)
		{
			cuda_nb = 1;
		}
		spawn_tests(cpu_offset, cpu_nb, cuda_offset, cuda_nb, &test_args);
	}
	{
		int cpu_offset = rm_nb_cpu_units/2;
		int cpu_nb = cpu_offset;
		if (cpu_nb == 0 && rm_nb_cpu_units > 0)
		{
			cpu_nb = 1;
		}
		int cuda_offset = rm_nb_cuda_units/2;
		int cuda_nb = rm_nb_cuda_units - cuda_offset;
		spawn_tests(cpu_offset, cpu_nb, cuda_offset, cuda_nb, &test_args);
	}
	_wait_pending_spawns();
	MORSE_Finalize();

	starpurm_shutdown();
	pthread_cond_destroy(&spawn_pending_cond);

	return 0;

}
