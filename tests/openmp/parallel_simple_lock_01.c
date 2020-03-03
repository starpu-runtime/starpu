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
 * Check the OpenMP simple lock support.
 */

#if !defined(STARPU_OPENMP)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
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

starpu_omp_lock_t omp_lock;

void locked_func(void)
{
	const int worker_id = starpu_worker_get_id();
	const pthread_t tid = pthread_self();
	printf("[tid %p] task thread = %d -- locked function\n", (void *)tid, worker_id);
}

void master_g1(void *arg)
{
	starpu_omp_init_lock(&omp_lock);
}

void master_g2(void *arg)
{
	starpu_omp_destroy_lock(&omp_lock);
}

void parallel_region_f(void *buffers[], void *args)
{
	const int worker_id = starpu_worker_get_id();
	const pthread_t tid = pthread_self();
	(void) buffers;
	(void) args;
	printf("[tid %p] task thread = %d -- parallel -->\n", (void *)tid, worker_id);
	starpu_omp_master(master_g1, NULL);
	starpu_omp_barrier();

	starpu_omp_set_lock(&omp_lock);
	locked_func();
	starpu_omp_unset_lock(&omp_lock);

	starpu_omp_set_lock(&omp_lock);
	locked_func();
	starpu_omp_unset_lock(&omp_lock);

	starpu_omp_set_lock(&omp_lock);
	locked_func();
	starpu_omp_unset_lock(&omp_lock);

	starpu_omp_set_lock(&omp_lock);
	locked_func();
	starpu_omp_unset_lock(&omp_lock);

	starpu_omp_barrier();
	starpu_omp_master(master_g2, NULL);
	printf("[tid %p] task thread = %d -- parallel <--\n", (void *)tid, worker_id);
}

int
main (void)
{
	struct starpu_omp_parallel_region_attr attr;
	pthread_t tid;
	tid = pthread_self();
	printf("<main>\n");
	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model        = &starpu_perfmodel_nop;
#endif
	attr.cl.flags        = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0] = parallel_region_f;
	attr.cl.where        = STARPU_CPU;
	attr.if_clause       = 1;
	starpu_omp_parallel_region(&attr);
	printf("<main>\n");
	starpu_omp_parallel_region(&attr);
	printf("<main>\n");
	return 0;
}
#endif
