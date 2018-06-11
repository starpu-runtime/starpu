/* StarPURM --- StarPU Resource Management Layer.
 *
 * Copyright (C) 2017, 2018  Inria
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
#include <hwloc/glibc-sched.h>

#define CHECK

static hwloc_topology_t topology;
static int rm_cpu_type_id = -1;
static int rm_cuda_type_id = -1;
static int rm_nb_cpu_units = 0;
static int rm_nb_cuda_units = 0;
static const int nb_random_tests = 10;

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


static void disp_selected_cpuset(void)
{
	hwloc_cpuset_t selected_cpuset = starpurm_get_selected_cpuset();
	int strl = hwloc_bitmap_snprintf(NULL, 0, selected_cpuset);
	char str[strl+1];
	hwloc_bitmap_snprintf(str, strl+1, selected_cpuset);
	printf("selected cpuset = %s\n", str);
}

int main( int argc, char const *argv[])
{
	int i, j;
	enum DDSS_TRANS transA = MorseTrans;
	enum DDSS_TRANS transB = MorseTrans;
	int ret;

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

	double *A = malloc(m * k * sizeof(double));
	double *B = malloc(k * n * sizeof(double));
	double *C = malloc(m * n * sizeof(double));
	double *C_test = malloc(m * n * sizeof(double));

	double alpha = (double)rand() / (double)rand() + DBL_MIN;
	double beta  = (double)rand() / (double)rand() + DBL_MIN;
 
	// Matrix A, B, C and C_test initialization
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i*n+j] = (double )rand() / (double)rand() + DBL_MIN;
			B[i*n+j] = (double )rand() / (double)rand() + DBL_MIN;
			C[i*n+j] = 0.0;
			C_test[i * n + j] = 0.0;
		}
	}

	/* Test case */
	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);
	starpurm_initialize();
	starpurm_set_drs_enable(NULL);
	init_rm_infos();
	printf("cpu units: %d\n", rm_nb_cpu_units);
	printf("cuda units: %d\n", rm_nb_cuda_units);
	printf("using default units\n");
	disp_selected_cpuset();

	/* GLIBC cpu_mask as supplied by POCL */
	cpu_set_t cpu_mask;
	CPU_ZERO(&cpu_mask);
	CPU_SET (0, &cpu_mask);
	CPU_SET (1, &cpu_mask);
	CPU_SET (2, &cpu_mask);
	CPU_SET (3, &cpu_mask);

	/* Convert GLIBC cpu_mask into HWLOC cpuset */
	hwloc_cpuset_t hwloc_cpuset = hwloc_bitmap_alloc();
	int status = hwloc_cpuset_from_glibc_sched_affinity(topology, hwloc_cpuset, &cpu_mask, sizeof(cpu_set_t));
	assert(status == 0);

	/* Reset any unit previously allocated to StarPU */
	starpurm_withdraw_all_cpus_from_starpu(NULL);
	/* Enforce new cpu mask */
	starpurm_assign_cpu_mask_to_starpu(NULL, hwloc_cpuset);

	/* task function */
	int M = m;
	int N = n;
	int K = k;
	double ALPHA = alpha;
	int LDA = k;
	int LDB = n;
	double BETA = beta;
	int LDC = n;

	MORSE_Init(4, 0);
	int res = MORSE_dgemm(transA, transB, M, N, K,
			ALPHA, A, LDA, B, LDB,
			BETA, C, LDC);
	MORSE_Finalize();

	/* Withdraw all CPU units from StarPU */
	starpurm_withdraw_all_cpus_from_starpu(NULL);

	hwloc_bitmap_free(hwloc_cpuset);

	starpurm_shutdown();

#ifdef CHECK
	/* Check */
	cblas_dgemm( CblasColMajor, 
			( CBLAS_TRANSPOSE ) transA,
			( CBLAS_TRANSPOSE ) transB,
			m, n, k,
			alpha, A, k,
			B, n,
			beta, C_test, n );

	double C_test_fnorm = LAPACKE_dlange(CblasColMajor, 'F', m, n, C_test, n);
	double C_test_inorm = LAPACKE_dlange(CblasColMajor, 'I', m, n, C_test, n);
	cblas_daxpy(m*n, -1, C, 1, C_test, 1);
	double fnorm = LAPACKE_dlange(CblasColMajor, 'F', m, n, C_test, n);
	double inorm = LAPACKE_dlange(CblasColMajor, 'I', m, n, C_test, n);
	fprintf(stdout, "||C_test-C||_F / ||C_test||_F = %e\n", fnorm/C_test_fnorm);
	fprintf(stdout, "||C_test-C||_I / ||C_test||_I = %e\n", inorm/C_test_inorm);
#endif

	return 0;

}
