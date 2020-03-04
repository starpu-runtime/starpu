/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Main body for the LU factorization: matrix initialization and result
 * checking */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <starpu.h>
#include "xlu.h"
#include "xlu_kernels.h"

#ifdef STARPU_HAVE_VALGRIND_H
#include <valgrind/valgrind.h>
#endif

static unsigned long size = 0;
static unsigned nblocks = 0;
static unsigned check = 0;
static unsigned pivot = 0;
static unsigned no_stride = 0;
static unsigned profile = 0;
static unsigned no_prio=0;
unsigned bound = 0;
unsigned bounddeps = 0;
unsigned boundprio = 0;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

TYPE *A, *A_saved;

/* in case we use non-strided blocks */
TYPE **A_blocks;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

#ifndef STARPU_SIMGRID
		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-piv") == 0)
		{
			pivot = 1;
		}

		else if (strcmp(argv[i], "-no-stride") == 0)
		{
			no_stride = 1;
		}
#endif

		else if (strcmp(argv[i], "-profile") == 0)
		{
			profile = 1;
		}

		else if (strcmp(argv[i], "-bound") == 0)
		{
			bound = 1;
		}
		else if (strcmp(argv[i], "-bounddeps") == 0)
		{
			bound = 1;
			bounddeps = 1;
		}
		else if (strcmp(argv[i], "-bounddepsprio") == 0)
		{
			bound = 1;
			bounddeps = 1;
			boundprio = 1;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			fprintf(stderr,"usage: lu [-size n] [-nblocks b] [-piv] [-no-stride] [-profile] [-bound] [-bounddeps] [-bounddepsprio]\n");
			fprintf(stderr,"Default is size %lu and nblocks %u\n", size, nblocks);
			exit(0);
		}
	}
}

static void display_matrix(TYPE *m, unsigned n, unsigned ld, char *str)
{
	(void)m;
	(void)n;
	(void)ld;
	(void)str;
#if 0
	FPRINTF(stderr, "***********\n");
	FPRINTF(stderr, "Display matrix %s\n", str);
	unsigned i,j;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < n; i++)
		{
			FPRINTF(stderr, "%2.2f\t", m[i+j*ld]);
		}
		FPRINTF(stderr, "\n");
	}
	FPRINTF(stderr, "***********\n");
#endif
}

void copy_blocks_into_matrix(void)
{
	unsigned blocksize = (size/nblocks);

	unsigned i, j;
	unsigned bi, bj;
	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		for (j = 0; j < blocksize; j++)
		for (i = 0; i < blocksize; i++)
		{
			A[(i+bi*blocksize) + (j + bj*blocksize)*size] =
				A_blocks[bi+nblocks*bj][i + j * blocksize];
		}

		starpu_free(A_blocks[bi+nblocks*bj]);
	}
}



void copy_matrix_into_blocks(void)
{
	unsigned blocksize = (size/nblocks);

	unsigned i, j;
	unsigned bi, bj;
	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		starpu_malloc((void **)&A_blocks[bi+nblocks*bj], (size_t)blocksize*blocksize*sizeof(TYPE));

		for (j = 0; j < blocksize; j++)
		for (i = 0; i < blocksize; i++)
		{
			A_blocks[bi+nblocks*bj][i + j * blocksize] =
			A[(i+bi*blocksize) + (j + bj*blocksize)*size];
		}
	}
}

static void init_matrix(void)
{
	/* allocate matrix */
#ifdef STARPU_SIMGRID
	A = (void*) 1;
#else
	starpu_malloc_flags((void **)&A, (size_t)size*size*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
#endif
	STARPU_ASSERT(A);

	starpu_srand48((long int)time(NULL));
	/* starpu_srand48(0); */

#ifndef STARPU_SIMGRID
	/* initialize matrix content */
	unsigned long i,j;
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			A[i + j*size] = (TYPE)starpu_drand48();
#ifdef COMPLEX_LU
			/* also randomize the imaginary component for complex number cases */
			A[i + j*size] += (TYPE)(I*starpu_drand48());
#endif
			if (i == j)
			{
				A[i + j*size] += 1;
				A[i + j*size] *= 100;
			}
		}
	}
#endif

}

static void save_matrix(void)
{
	A_saved = malloc((size_t)size*size*sizeof(TYPE));
	STARPU_ASSERT(A_saved);

	memcpy(A_saved, A, (size_t)size*size*sizeof(TYPE));
}

static double frobenius_norm(TYPE *v, unsigned n)
{
	double sum2 = 0.0;

	/* compute sqrt(Sum(|x|^2)) */

	unsigned i,j;
	for (j = 0; j < n; j++)
	for (i = 0; i < n; i++)
	{
		double a = fabsl((double)v[i+n*j]);
		sum2 += a*a;
	}

	return sqrt(sum2);
}

static void pivot_saved_matrix(unsigned *ipiv)
{
	unsigned k;
	for (k = 0; k < size; k++)
	{
		if (k != ipiv[k])
		{
	/*		FPRINTF(stderr, "SWAP %d and %d\n", k, ipiv[k]); */
			CPU_SWAP(size, &A_saved[k*size], 1, &A_saved[ipiv[k]*size], 1);
		}
	}
}

static void check_result(void)
{
	unsigned i,j;
	TYPE *L, *U;

	L = malloc((size_t)size*size*sizeof(TYPE));
	U = malloc((size_t)size*size*sizeof(TYPE));

	memset(L, 0, size*size*sizeof(TYPE));
	memset(U, 0, size*size*sizeof(TYPE));

	/* only keep the lower part */
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < j; i++)
		{
			L[j+i*size] = A[j+i*size];
		}

		/* diag i = j */
		L[j+j*size] = A[j+j*size];
		U[j+j*size] = 1.0;

		for (i = j+1; i < size; i++)
		{
			U[j+i*size] = A[j+i*size];
		}
	}

	display_matrix(L, size, size, "L");
	display_matrix(U, size, size, "U");

	/* now A_err = L, compute L*U */
	CPU_TRMM("R", "U", "N", "U", size, size, 1.0f, U, size, L, size);

	display_matrix(A_saved, size, size, "P A_saved");
	display_matrix(L, size, size, "LU");

	/* compute "LU - A" in L*/
	CPU_AXPY(size*size, -1.0, A_saved, 1, L, 1);
	display_matrix(L, size, size, "Residuals");

#ifdef COMPLEX_LU
	double err = CPU_ASUM(size*size, L, 1);
	int max = CPU_IAMAX(size*size, L, 1);
	TYPE l_max = L[max];

	FPRINTF(stderr, "Avg error : %e\n", err/(size*size));
	FPRINTF(stderr, "Max error : %e\n", sqrt(creal(l_max)*creal(l_max)+cimag(l_max)*cimag(l_max)));
#else
	TYPE err = CPU_ASUM(size*size, L, 1);
	int max = CPU_IAMAX(size*size, L, 1);

	FPRINTF(stderr, "Avg error : %e\n", err/(size*size));
	FPRINTF(stderr, "Max error : %e\n", L[max]);
#endif

	double residual = frobenius_norm(L, size);
	double matnorm = frobenius_norm(A_saved, size);

	FPRINTF(stderr, "||%sA-LU|| / (||A||*N) : %e\n", pivot?"P":"", residual/(matnorm*size));

	if (residual/(matnorm*size) > 1e-5)
		exit(-1);

	free(L);
	free(U);
	free(A_saved);
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	int power = starpu_cpu_worker_get_count() + 32 * starpu_cuda_worker_get_count();
	int power_cbrt = cbrt(power);
#ifndef STARPU_LONG_CHECK
	power_cbrt /= 2;
#endif
	if (power_cbrt < 1)
		power_cbrt = 1;

#ifdef STARPU_QUICK_CHECK
	if (!size)
		size = 320*2*power_cbrt;
	if (!nblocks)
		nblocks = 2*power_cbrt;
#else
	if (!size)
		size = 960*8*power_cbrt;
	if (!nblocks)
		nblocks = 8*power_cbrt;
#endif

	parse_args(argc, argv);

#ifdef STARPU_HAVE_VALGRIND_H
	if (RUNNING_ON_VALGRIND)
		size = 16;
#endif

	starpu_cublas_init();

	init_matrix();

#ifndef STARPU_SIMGRID
	unsigned *ipiv = NULL;
	if (check)
		save_matrix();

	display_matrix(A, size, size, "A");

	if (profile)
		starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	/* Factorize the matrix (in place) */
	if (pivot)
	{
 		ipiv = malloc(size*sizeof(unsigned));
		if (no_stride)
		{
			/* in case the LU decomposition uses non-strided blocks, we _copy_ the matrix into smaller blocks */
			A_blocks = malloc(nblocks*nblocks*sizeof(TYPE *));
			copy_matrix_into_blocks();

			ret = STARPU_LU(lu_decomposition_pivot_no_stride)(A_blocks, ipiv, size, size, nblocks, no_prio);

			copy_blocks_into_matrix();
			free(A_blocks);
		}
		else
		{
			double start;
			double end;

			start = starpu_timing_now();

			ret = STARPU_LU(lu_decomposition_pivot)(A, ipiv, size, size, nblocks, no_prio);

			end = starpu_timing_now();

			double timing = end - start;

			unsigned n = size;
			double flop = (2.0f*n*n*n)/3.0f;
			FPRINTF(stderr, "Synthetic GFlops (TOTAL) : \n");
			FPRINTF(stdout, "%u	%6.2f\n", n, (flop/timing/1000.0f));
		}
	}
	else
#endif
	{
		ret = STARPU_LU(lu_decomposition)(A, size, size, nblocks, no_prio);
	}

	if (profile)
	{
		FPRINTF(stderr, "Setting profile\n");
		starpu_profiling_status_set(STARPU_PROFILING_DISABLE);
		starpu_profiling_bus_helper_display_summary();
	}

	if (bound)
	{
		if (bounddeps)
		{
			FILE *f = fopen("lu.pl", "w");
			starpu_bound_print_lp(f);
			FPRINTF(stderr,"system printed to lu.pl\n");
			fclose(f);
			f = fopen("lu.mps", "w");
			starpu_bound_print_mps(f);
			FPRINTF(stderr,"system printed to lu.mps\n");
			fclose(f);
			f = fopen("lu.dot", "w");
			starpu_bound_print_dot(f);
			FPRINTF(stderr,"system printed to lu.mps\n");
			fclose(f);
		}
	}

#ifndef STARPU_SIMGRID
	if (check)
	{
		FPRINTF(stderr, "Checking result\n");
		if (pivot)
		{
			pivot_saved_matrix(ipiv);
		}

		check_result();
	}

	if (pivot)
		free(ipiv);
#endif

#ifndef STARPU_SIMGRID
	starpu_free_flags(A, (size_t)size*size*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
#endif

	starpu_cublas_shutdown();

	starpu_shutdown();

	if (ret == -ENODEV) return 77; else return 0;
}
