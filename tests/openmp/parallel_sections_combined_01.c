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
 * Check the OpenMP combined parallel sections support.
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

void f(unsigned long long section_num, void *arg)
{
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d, section [%llu: %s]\n", (void *)tid, worker_id, section_num, (const char *)arg);
}

void parallel_region_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	void *section_args[4];
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);

	section_args[0] = (void *)"A";
	section_args[1] = (void *)"B";
	section_args[2] = (void *)"C";
	section_args[3] = (void *)"D";

	starpu_omp_sections_combined(4, f, section_args, 0);

	section_args[0] = (void *)"E";
	section_args[1] = (void *)"F";
	section_args[2] = (void *)"G";
	section_args[3] = (void *)"H";

	starpu_omp_sections_combined(4, f, section_args, 0);

	section_args[0] = (void *)"I";
	section_args[1] = (void *)"J";
	section_args[2] = (void *)"K";
	section_args[3] = (void *)"L";

	starpu_omp_sections_combined(4, f, section_args, 0);

	section_args[0] = (void *)"M";
	section_args[1] = (void *)"N";
	section_args[2] = (void *)"O";
	section_args[3] = (void *)"P";

	starpu_omp_sections_combined(4, f, section_args, 0);
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
