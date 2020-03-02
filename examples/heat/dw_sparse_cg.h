/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DW_SPARSE_CG_H__
#define __DW_SPARSE_CG_H__

#include <stdio.h>
#include <stdint.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <signal.h>

#include <starpu.h>

#include "../common/blas.h"

#define MAXITER	100000
#define EPSILON	0.0000001f

/* code parameters */
static uint32_t _size = 33554432;
static unsigned _usecpu = 0;
static unsigned _blocks = 512;
static unsigned _grids  = 8;

struct cg_problem
{
	starpu_data_handle_t ds_matrixA;
	starpu_data_handle_t ds_vecx;
	starpu_data_handle_t ds_vecb;
	starpu_data_handle_t ds_vecr;
	starpu_data_handle_t ds_vecd;
	starpu_data_handle_t ds_vecq;

	sem_t *sem;
	
	float alpha;
	float beta;
	float delta_0;
	float delta_old;
	float delta_new;
	float epsilon;

	int i;
	unsigned size;
};

/* some useful functions */
static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			_size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-block") == 0)
		{
			char *argptr;
			_blocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-grid") == 0)
		{
			char *argptr;
			_grids = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu") == 0)
		{
			_usecpu = 1;
		}
	}
}


static void print_results(float *result, unsigned size)
{
	printf("**** RESULTS **** \n");
	unsigned i;

	for (i = 0; i < STARPU_MIN(size, 16); i++)
	{
		printf("%u -> %f\n", i, result[i]);
	}
}

void cpu_codelet_func_1(void *descr[], void *arg);

void cpu_codelet_func_2(void *descr[], void *arg);

void cublas_codelet_func_3(void *descr[], void *arg);
void cpu_codelet_func_3(void *descr[], void *arg);

void cpu_codelet_func_4(void *descr[], void *arg);

void cpu_codelet_func_5(void *descr[], void *arg);
void cublas_codelet_func_5(void *descr[], void *arg);

void cublas_codelet_func_6(void *descr[], void *arg);
void cpu_codelet_func_6(void *descr[], void *arg);

void cublas_codelet_func_7(void *descr[], void *arg);
void cpu_codelet_func_7(void *descr[], void *arg);

void cublas_codelet_func_8(void *descr[], void *arg);
void cpu_codelet_func_8(void *descr[], void *arg);

void cublas_codelet_func_9(void *descr[], void *arg);
void cpu_codelet_func_9(void *descr[], void *arg);

void iteration_cg(void *problem);

void conjugate_gradient(float *nzvalA, float *vecb, float *vecx, uint32_t nnz,
			unsigned nrow, uint32_t *colind, uint32_t *rowptr);

#endif /* __DW_SPARSE_CG_H__ */
