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
 * Test recursive OpenMP tasks, data dependences, data slice dependences.
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

void task_region_h(void *buffers[], void *_args)
{
	void **args = _args;
	struct starpu_vector_interface *_vector = buffers[0];
	int nx = STARPU_VECTOR_GET_NX(_vector);
	int elemsize = STARPU_VECTOR_GET_ELEMSIZE(_vector);
	int slice_base = STARPU_VECTOR_GET_SLICE_BASE(_vector);
	int *v = (int *)STARPU_VECTOR_GET_PTR(_vector);
	int f = (int)(intptr_t)args[0];
	int imin = (int)(intptr_t)args[1];
	int imax = (int)(intptr_t)args[2];
	int i;

	assert(elemsize == sizeof(v[0]));

	printf("depth 2 task, entry: vector ptr = %p, slice_base = %d, imin = %d, imax = %d\n", v, slice_base, imin, imax);

	for (i = imin; i < imax; i++)
	{
                assert(i-slice_base>=0);
                assert(i-slice_base<NX);
                (v-slice_base)[i] += f;
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
		int i;

		for (i = 0; i < nx; i++)
		{
			v[i] += f;
		}
	}

	{
		const int half_nx = nx/2;

		starpu_data_handle_t vector_slice_1_handle;
		starpu_vector_data_register(&vector_slice_1_handle, STARPU_MAIN_RAM, (uintptr_t)&v[0], half_nx, sizeof(v[0]));
		printf("depth 1 task, block 1: vector_slice_1_handle = %p\n", vector_slice_1_handle);

		starpu_data_handle_t vector_slice_2_handle;
		starpu_vector_data_register(&vector_slice_2_handle, STARPU_MAIN_RAM, (uintptr_t)&v[half_nx], nx-half_nx, sizeof(v[0]));
		/* set slice base */
		starpu_omp_vector_annotate(vector_slice_2_handle, half_nx);
		printf("depth 1 task, block 1: vector_slice_2_handle = %p\n", vector_slice_2_handle);

	}

	void *cl_arg_1[3];
	void *cl_arg_2[3];

	{
		struct starpu_omp_task_region_attr attr;
		const int half_nx = nx/2;
		int i;

		starpu_data_handle_t vector_slice_1_handle = starpu_data_lookup(&v[0]);
		printf("depth 1 task, block 2: vector_slice_1_handle = %p\n", vector_slice_1_handle);

		starpu_data_handle_t vector_slice_2_handle = starpu_data_lookup(&v[half_nx]);
		printf("depth 1 task, block 2: vector_slice_2_handle = %p\n", vector_slice_2_handle);

		memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
		attr.cl.model         = &starpu_perfmodel_nop;
#endif
		attr.cl.flags         = STARPU_CODELET_SIMGRID_EXECUTE;
		attr.cl.cpu_funcs[0]  = task_region_h;
		attr.cl.where         = STARPU_CPU;
		attr.cl.nbuffers      = 1;
		attr.cl.modes[0]      = STARPU_RW;
		attr.cl_arg_size      = 3*sizeof(void *);
		attr.cl_arg_free      = 0;
		attr.if_clause        = 1;
		attr.final_clause     = 0;
		attr.untied_clause    = 1;
		attr.mergeable_clause = 0;

		i = 0;

		cl_arg_1[0] = (void *)(intptr_t)i++;
		cl_arg_1[1] = (void *)(intptr_t)0;
		cl_arg_1[2] = (void *)(intptr_t)half_nx;
		attr.cl_arg           = cl_arg_1;
		attr.handles          = &vector_slice_1_handle;
		starpu_omp_task_region(&attr);

		cl_arg_2[0] = (void *)(intptr_t)i++;
		cl_arg_2[1] = (void *)(intptr_t)half_nx;
		cl_arg_2[2] = (void *)(intptr_t)nx;
		attr.cl_arg           = cl_arg_2;
		attr.handles          = &vector_slice_2_handle;
		starpu_omp_task_region(&attr);
	}

	starpu_omp_taskwait();
}

void master_g1(void *arg)
{
	(void)arg;
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
	(void)arg;
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
	(void)buffers;
	(void)args;
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

	assert(NX >= 2);
	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model        = &starpu_perfmodel_nop;
#endif
	attr.cl.flags         = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0] = parallel_region_f;
	attr.cl.where        = STARPU_CPU;
	attr.if_clause       = 1;
	starpu_omp_parallel_region(&attr);
	return 0;
}
#endif
