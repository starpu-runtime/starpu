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
 * Check executing a CUDA target task.
 */

#if !defined(STARPU_OPENMP) || !defined(STARPU_USE_CUDA)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
#define	NX	64
int global_vector_1[NX];
int global_vector_2[NX];

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

void task_region_g(void *buffers[], void *args)
{
	struct starpu_vector_interface *_vector_1 = buffers[0];
	int nx1 = STARPU_VECTOR_GET_NX(_vector_1);
	int *v1 = (int *)STARPU_VECTOR_GET_PTR(_vector_1);

	struct starpu_vector_interface *_vector_2 = buffers[1];
	int nx2 = STARPU_VECTOR_GET_NX(_vector_2);
	int *v2 = (int *)STARPU_VECTOR_GET_PTR(_vector_2);

	int f = (int)(intptr_t)args;

	STARPU_ASSERT(nx1 == nx2);

	printf("depth 1 task, entry: vector_1 ptr = %p\n", v1);
	printf("depth 1 task, entry: vector_2 ptr = %p\n", v2);
	printf("depth 1 task, entry: f = %d\n", f);

	fprintf(stderr, "cudaMemcpy: -->\n");
	cudaMemcpyAsync(v2,v1,nx1*sizeof(*_vector_1), cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
	fprintf(stderr, "cudaMemcpy: <--\n");
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

void master_g1(void *arg)
{
	(void)arg;
	{
		starpu_data_handle_t region_vector_handle;
		int i;

		printf("master_g1: vector ptr = %p\n", global_vector_1);
		for (i = 0; i < NX; i++)
		{
			global_vector_1[i] = 1;
		}

		starpu_vector_data_register(&region_vector_handle, STARPU_MAIN_RAM, (uintptr_t)global_vector_1, NX, sizeof(global_vector_1[0]));
		printf("master_g1: region_vector_handle = %p\n", region_vector_handle);
	}
	{
		starpu_data_handle_t region_vector_handle;
		int i;

		printf("master_g1: vector ptr = %p\n", global_vector_2);
		for (i = 0; i < NX; i++)
		{
			global_vector_2[i] = 0;
		}

		starpu_vector_data_register(&region_vector_handle, STARPU_MAIN_RAM, (uintptr_t)global_vector_2, NX, sizeof(global_vector_2[0]));
		printf("master_g1: region_vector_handle = %p\n", region_vector_handle);
	}
}

void master_g2(void *arg)
{
	(void)arg;
	starpu_data_handle_t region_vector_handles[2];
	struct starpu_omp_task_region_attr attr;
	int i;

	region_vector_handles[0] = starpu_data_lookup(global_vector_1);
	printf("master_g2: region_vector_handles[0] = %p\n", region_vector_handles[0]);
	region_vector_handles[1] = starpu_data_lookup(global_vector_2);
	printf("master_g2: region_vector_handles[1] = %p\n", region_vector_handles[1]);

	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model         = &starpu_perfmodel_nop;
	attr.cl.flags         = STARPU_CODELET_SIMGRID_EXECUTE;
#endif
	attr.cl.cpu_funcs[0]  = NULL;
	attr.cl.cuda_funcs[0] = task_region_g;
	attr.cl.where         = STARPU_CUDA;
	attr.cl.nbuffers      = 2;
	attr.cl.modes[0]      = STARPU_R;
	attr.cl.modes[1]      = STARPU_W;
	attr.handles          = region_vector_handles;
	attr.cl_arg_size      = sizeof(void *);
	attr.cl_arg_free      = 0;
	attr.if_clause        = 1;
	attr.final_clause     = 0;
	attr.untied_clause    = 1;
	attr.mergeable_clause = 0;

	i = 0;

	attr.cl_arg = (void *)(intptr_t)i;
	starpu_omp_task_region(&attr);
}

void parallel_region_f(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;
	starpu_omp_master(master_g1, NULL);
	starpu_omp_barrier();
	{
		starpu_data_handle_t region_vector_handle_1;
		region_vector_handle_1 = starpu_data_lookup(global_vector_1);
		printf("parallel_region block 1: region_vector_handle_1 = %p\n", region_vector_handle_1);
	}
	{
		starpu_data_handle_t region_vector_handle_2;
		region_vector_handle_2 = starpu_data_lookup(global_vector_2);
		printf("parallel_region block 1: region_vector_handle_2 = %p\n", region_vector_handle_2);
	}
	starpu_omp_barrier();
	starpu_omp_master(master_g2, NULL);
	starpu_omp_barrier();
	{
		starpu_data_handle_t region_vector_handle_1;
		region_vector_handle_1 = starpu_data_lookup(global_vector_1);
		printf("parallel_region block 2: region_vector_handle_1 = %p\n", region_vector_handle_1);
	}
	{
		starpu_data_handle_t region_vector_handle_2;
		region_vector_handle_2 = starpu_data_lookup(global_vector_2);
		printf("parallel_region block 2: region_vector_handle_2 = %p\n", region_vector_handle_2);
	}
}

int
main (void)
{
	struct starpu_omp_parallel_region_attr attr;

	if (starpu_cuda_worker_get_count() < 1)
	{
		return STARPU_TEST_SKIPPED;
	}

	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model        = &starpu_perfmodel_nop;
#endif
	attr.cl.flags        = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0] = parallel_region_f;
	attr.cl.where        = STARPU_CPU;
	attr.if_clause       = 1;
	starpu_omp_parallel_region(&attr);

	int i;
	for (i = 0; i < NX; i++)
	{
		if (global_vector_1[i] != global_vector_2[i])
		{
			fprintf(stderr, "check failed: global_vector_1[%d] = %d, global_vector_2[%d] = %d\n", i, global_vector_1[i], i, global_vector_2[i]);
			return EXIT_FAILURE;
		}
	}
	return 0;
}
#endif
