/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2017       Erwan Leria
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

#ifndef TYPE
#error "Do not compile xgemm.c directly, compile sgemm.c or dgemm.c"
#endif

#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <starpu.h>
#include <starpu_fxt.h>

#ifdef STARPU_HAVE_BLAS
#include <common/blas.h>
#endif

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <starpu_cublas_v2.h>
static const TYPE p1_cuda = 1.0;
static const TYPE v0_cuda = 0.0;
#endif

#ifdef STARPU_USE_HIP
#include <hip/hip_runtime.h>
#include <starpu_hipblas.h>
static const TYPE p1_hip = 1.0;
static const TYPE v0_hip = 0.0;
#endif

#ifdef STARPU_QUICK_CHECK
static unsigned niter = 2;
#else
static unsigned niter = 10;
#endif
static unsigned nsleeps = 1;
static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
static unsigned nslicesz = 4;
#if defined(STARPU_QUICK_CHECK) && !defined(STARPU_SIMGRID)
static unsigned xdim = 256;
static unsigned ydim = 256;
static unsigned zdim = 64;
#else
static unsigned xdim = 960*4;
static unsigned ydim = 960*4;
static unsigned zdim = 960*4;
#endif
static unsigned check = 0;
static unsigned bound = 0;
static unsigned print_hostname = 0;
static unsigned tiled = 0;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)

static TYPE *A, *B, *C;
static starpu_data_handle_t A_handle, B_handle, C_handle;

#ifdef STARPU_HAVE_BLAS
static int check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", ydim, xdim, zdim, (TYPE)-1.0f, A, ydim, B, zdim, (TYPE)1.0f, C, ydim);

	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(xdim*ydim, C, 1);

	if (err < EPSILON*xdim*ydim*zdim)
	{
		FPRINTF(stderr, "Results are OK\n");
		return 0;
	}
	else
	{
		int max;
		max = CPU_IAMAX(xdim*ydim, C, 1);

		FPRINTF(stderr, "There were errors ... err = %f\n", err);
		FPRINTF(stderr, "Max error : %e\n", C[max]);
		return 1;
	}
}
#endif

static int clean_problem_data(int enodev)
{
	int ret = enodev;

	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

#ifdef STARPU_HAVE_BLAS
#ifndef STARPU_SIMGRID
	if (!enodev && check)
		ret = check_output();
#endif
#endif

	starpu_free_flags(A, zdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, xdim*zdim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, xdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

	return ret;
}

#ifdef STARPU_USE_CUDA
static void cublas_mult(void *descr[], void *arg, const TYPE *beta);
static void cublas_gemm0(void *descr[], void *arg)
{
	cublas_mult(descr, arg, &v0_cuda);
}

static void cublas_gemm(void *descr[], void *arg)
{
	cublas_mult(descr, arg, &p1_cuda);
}
#endif

#ifdef STARPU_USE_HIP
static void hipblas_mult(void *descr[], void *arg, const TYPE *beta)
{
        (void)arg;
        TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
        TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
        TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

        unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
        unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
        unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

        unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
        unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
        unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

        hipblasStatus_t status = HIPBLAS_GEMM(starpu_hipblas_get_local_handle(),
					      HIPBLAS_OP_N, HIPBLAS_OP_N,
					      nxC, nyC, nyA,
					      &p1_hip, subA, ldA, subB, ldB,
					      beta, subC, ldC);
        if (status != HIPBLAS_STATUS_SUCCESS)
                STARPU_HIPBLAS_REPORT_ERROR(status);
}

static void hipblas_gemm0(void *descr[], void *arg)
{
        hipblas_mult(descr, arg, &v0_hip);
}

static void hipblas_gemm(void *descr[], void *arg)
{
        hipblas_mult(descr, arg, &p1_hip);
}
#endif

#ifdef STARPU_HAVE_BLAS
void cpu_mult(void *descr[], void *arg, TYPE beta);
void cpu_gemm0(void *descr[], void *arg)
{
	cpu_mult(descr, arg, 0.);
}

void cpu_gemm(void *descr[], void *arg)
{
	cpu_mult(descr, arg, 1.);
}
#endif

static struct starpu_perfmodel starpu_gemm_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = STARPU_GEMM_STR(gemm)
};

static void parse_args(int argc, char **argv);
static void init_problem_data(void);
static void partition_mult_data(void);
static int run_data(void);

int main(int argc, char **argv)
{
	parse_args(argc, argv);

	starpu_fxt_autostart_profiling(0);
	int ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_cublas_init();
	starpu_hipblas_init();

	init_problem_data();
	partition_mult_data();

	ret = run_data();
	ret = clean_problem_data(ret);

	starpu_cublas_shutdown();
	starpu_hipblas_shutdown();
	starpu_shutdown();

	return ret;
}
