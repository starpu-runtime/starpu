/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <pthread.h>
#include <starpu.h>
#include "../helper.h"
#include <stdio.h>

/*
 * Check the OpenMP nested task support.
 */

#if !defined(STARPU_OPENMP)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
#define	NX	64
int global_vector[NX];

__attribute__((constructor))
static void omp_constructor(void)
{
	int ret = starpu_omp_init();
	if (ret == -EINVAL) exit(STARPU_TEST_SKIPPED);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
}

__attribute__((destructor))
static void omp_destructor(void)
{
	starpu_omp_shutdown();
}

void task_region_h(void *buffers[], void *args)
{
	struct starpu_vector_interface *_vector = buffers[0];
	int nx = STARPU_VECTOR_GET_NX(_vector);
	int *v = (int *)STARPU_VECTOR_GET_PTR(_vector);
	int f = (int)(intptr_t)args;
	int i;

	printf("depth 2 task, entry: vector ptr = %p\n", v);

	for (i = 0; i < nx; i++)
	{
                v[i] += f;
	}

	printf("depth 2 task ending\n");
}

void task_region_g(void *buffers[], void *args)
{
	struct starpu_vector_interface *_vector = buffers[0];

	int nx = STARPU_VECTOR_GET_NX(_vector);
	int *v = (int *)STARPU_VECTOR_GET_PTR(_vector);
	int f = (int)(intptr_t)args;

	printf("depth 1 task, entry: vector ptr = %p\n", v);

	{
		starpu_data_handle_t task_vector_handle;
		int i;

		for (i = 0; i < nx; i++)
		{
			v[i] += f;
		}

		starpu_vector_data_register(&task_vector_handle, STARPU_MAIN_RAM, (uintptr_t)v, NX, sizeof(v[0]));
		printf("depth 1 task, block 1: task_vector_handle = %p\n", task_vector_handle);
	}

	{
		starpu_data_handle_t task_vector_handle;
		struct starpu_omp_task_region_attr attr;
		int i;

		task_vector_handle = starpu_data_lookup(v);
		printf("depth 1 task, block 2: task_vector_handle = %p\n", task_vector_handle);

		memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
		attr.cl.model         = &starpu_perfmodel_nop;
#endif
		attr.cl.flags         = STARPU_CODELET_SIMGRID_EXECUTE;
		attr.cl.cpu_funcs[0]  = task_region_h;
		attr.cl.where         = STARPU_CPU;
		attr.cl.nbuffers      = 1;
		attr.cl.modes[0]      = STARPU_RW;
		attr.handles          = &task_vector_handle;
		attr.cl_arg_size      = sizeof(void *);
		attr.cl_arg_free      = 0;
		attr.if_clause        = 1;
		attr.final_clause     = 0;
		attr.untied_clause    = 1;
		attr.mergeable_clause = 0;

		i = 0;

		attr.cl_arg = (void *)(intptr_t)i++;
		starpu_omp_task_region(&attr);
		attr.cl_arg = (void *)(intptr_t)i++;
		starpu_omp_task_region(&attr);
	}

	starpu_omp_taskwait();
}

void master_g1(void *arg)
{
	starpu_data_handle_t region_vector_handle;
	int i;

	printf("master_g1: vector ptr = %p\n", global_vector);
	for (i = 0; i < NX; i++)
	{
		global_vector[i] = 1;
	}

	starpu_vector_data_register(&region_vector_handle, STARPU_MAIN_RAM, (uintptr_t)global_vector, NX, sizeof(global_vector[0]));
	printf("master_g1: region_vector_handle = %p\n", region_vector_handle);
}

void master_g2(void *arg)
{
	starpu_data_handle_t region_vector_handle;
	struct starpu_omp_task_region_attr attr;
	int i;

	region_vector_handle = starpu_data_lookup(global_vector);
	printf("master_g2: region_vector_handle = %p\n", region_vector_handle);

	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model         = &starpu_perfmodel_nop;
#endif
	attr.cl.flags         = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0]  = task_region_g;
	attr.cl.where         = STARPU_CPU;
	attr.cl.nbuffers      = 1;
	attr.cl.modes[0]      = STARPU_RW;
	attr.handles          = &region_vector_handle;
	attr.cl_arg_size      = sizeof(void *);
	attr.cl_arg_free      = 0;
	attr.if_clause        = 1;
	attr.final_clause     = 0;
	attr.untied_clause    = 1;
	attr.mergeable_clause = 0;

	i = 0;

	attr.cl_arg = (void *)(intptr_t)i++;
	starpu_omp_task_region(&attr);
	attr.cl_arg = (void *)(intptr_t)i++;
	starpu_omp_task_region(&attr);
	attr.cl_arg = (void *)(intptr_t)i++;
	starpu_omp_task_region(&attr);
	attr.cl_arg = (void *)(intptr_t)i++;
	starpu_omp_task_region(&attr);
}

void parallel_region_f(void *buffers[], void *args)
{
	starpu_omp_master(master_g1, NULL);
	starpu_omp_barrier();
	{
		starpu_data_handle_t region_vector_handle;
		region_vector_handle = starpu_data_lookup(global_vector);
		printf("parallel_region block 1: region_vector_handle = %p\n", region_vector_handle);
	}
	starpu_omp_barrier();
	starpu_omp_master(master_g2, NULL);
	starpu_omp_barrier();
	{
		starpu_data_handle_t region_vector_handle;
		region_vector_handle = starpu_data_lookup(global_vector);
		printf("parallel_region block 2: region_vector_handle = %p\n", region_vector_handle);
	}
}

int
main (void)
{
	struct starpu_omp_parallel_region_attr attr;

	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model        = &starpu_perfmodel_nop;
#endif
	attr.cl.flags        = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0] = parallel_region_f;
	attr.cl.where        = STARPU_CPU;
	attr.if_clause       = 1;
	starpu_omp_parallel_region(&attr);
	return 0;
}
#endif
