/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Check the OpenMP inline single with copyprivate support.
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

void parallel_region_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	int single_worker_id;
	int i;

	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d -- parallel -->\n", (void *)tid, worker_id);

	for (i=0; i<4; i++)
	{
		int *single_data;

		if ((single_data = starpu_omp_single_copyprivate_inline_begin(&single_worker_id)) == NULL)
		{
			printf("[tid %p] task thread = %d -- single\n", (void *)tid, worker_id);
			single_worker_id = worker_id;
		}
		else
		{
			memcpy(&single_worker_id, single_data, sizeof(single_worker_id));
		}
		starpu_omp_single_copyprivate_inline_end();
		printf("[tid %p] task thread = %d -- single_worker_id = %d\n", (void *)tid, worker_id, single_worker_id);
	}

	printf("[tid %p] task thread = %d -- parallel <--\n", (void *)tid, worker_id);
}

int main(void)
{
	struct starpu_omp_parallel_region_attr attr;
	pthread_t tid;
	tid = pthread_self();
	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model        = &starpu_perfmodel_nop;
#endif
	attr.cl.flags        = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0] = parallel_region_f;
	attr.cl.where        = STARPU_CPU;
	attr.if_clause       = 1;
	printf("<main>\n");
	starpu_omp_parallel_region(&attr);
	printf("<main>\n");
	starpu_omp_parallel_region(&attr);
	printf("<main>\n");
	return 0;
}
#endif
