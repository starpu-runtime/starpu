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
 * Check multiple OpenMP parallel for support.
 */

#if !defined(STARPU_OPENMP)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
#define NB_ITERS 4321
#define CHUNK 42
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

void for_g(unsigned long long i, unsigned long long nb_i, void *arg)
{
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d, for [%s] iterations first=%llu:nb=%llu\n", (void *)tid, worker_id, (const char *)arg, i, nb_i);
	for (; nb_i > 0; i++, nb_i--)
	{
		printf("[tid %p] task thread = %d, for [%s] iteration %llu\n", (void *)tid, worker_id, (const char *)arg, i);
	}
}

void parallel_region_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"static chunk", NB_ITERS, CHUNK, starpu_omp_sched_static, 0, 1);
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"static nochunk", NB_ITERS, 0, starpu_omp_sched_static, 0, 1);

	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"dynamic chunk", NB_ITERS, CHUNK, starpu_omp_sched_dynamic, 0, 1);
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"dynamic nochunk", NB_ITERS, 0, starpu_omp_sched_dynamic, 0, 1);

	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"guided chunk", NB_ITERS, CHUNK, starpu_omp_sched_guided, 0, 1);
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"guided nochunk", NB_ITERS, 0, starpu_omp_sched_guided, 0, 1);
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
