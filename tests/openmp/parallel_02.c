/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  Inria
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

#if !defined(STARPU_OPENMP)
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else
__attribute__((constructor))
static void omp_constructor(void)
{
	int ret = starpu_omp_init();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
}

__attribute__((destructor))
static void omp_destructor(void)
{
	starpu_omp_shutdown();
}

void parallel_region_2_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] parallel region 2: task thread = %d\n", (void *)tid, worker_id);
}

static struct starpu_codelet parallel_region_2_cl =
{
	.cpu_funcs    = { parallel_region_2_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

void parallel_region_1_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] parallel region 1: task thread = %d\n", (void *)tid, worker_id);

	starpu_omp_parallel_region(&parallel_region_2_cl, NULL, NULL);
}

static struct starpu_codelet parallel_region_1_cl =
{
	.cpu_funcs    = { parallel_region_1_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

int
main (int argc, char *argv[]) {
	starpu_omp_parallel_region(&parallel_region_1_cl, NULL, NULL);
	return 0;
}
#endif
