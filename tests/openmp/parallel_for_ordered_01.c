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
#define NB_ITERS 256
#define CHUNK 16
unsigned long long array[NB_ITERS];

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

void ordered_f(unsigned long long i, void *arg)
{
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d, for [%s] iteration (ordered) %llu\n", (void *)tid, worker_id, (const char *)arg, i);
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
		array[i] = 1;
		starpu_omp_ordered(ordered_f, arg, i);
	}
}

void parallel_region_1_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"static chunk", NB_ITERS, CHUNK, starpu_omp_sched_static, 1, 0);
}

static struct starpu_codelet parallel_region_1_cl =
{
	.cpu_funcs    = { parallel_region_1_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

void parallel_region_2_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"static nochunk", NB_ITERS, 0, starpu_omp_sched_static, 1, 0);
}

static struct starpu_codelet parallel_region_2_cl =
{
	.cpu_funcs    = { parallel_region_2_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

void parallel_region_3_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"dynamic chunk", NB_ITERS, CHUNK, starpu_omp_sched_dynamic, 1, 0);
}

static struct starpu_codelet parallel_region_3_cl =
{
	.cpu_funcs    = { parallel_region_3_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

void parallel_region_4_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"dynamic nochunk", NB_ITERS, 0, starpu_omp_sched_dynamic, 1, 0);
}

static struct starpu_codelet parallel_region_4_cl =
{
	.cpu_funcs    = { parallel_region_4_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

void parallel_region_5_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"guided nochunk", NB_ITERS, 0, starpu_omp_sched_guided, 1, 0);
}

static struct starpu_codelet parallel_region_5_cl =
{
	.cpu_funcs    = { parallel_region_5_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

void parallel_region_6_f(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d\n", (void *)tid, worker_id);
	starpu_omp_for(for_g, (void*)"guided nochunk", NB_ITERS, 0, starpu_omp_sched_guided, 1, 0);
}

static struct starpu_codelet parallel_region_6_cl =
{
	.cpu_funcs    = { parallel_region_6_f, NULL },
	.where        = STARPU_CPU,
	.nbuffers     = 0

};

static void clear_array(void)
{
	memset(array, 0, NB_ITERS*sizeof(unsigned long long));
}

static void check_array(void)
{
	unsigned long long i;
	unsigned long long s = 0;
	for (i = 0; i < NB_ITERS; i++)
	{
		s += array[i];
	}
	if (s != NB_ITERS)
	{
		printf("missing iterations\n");
		exit(1);
	}
}
int
main (int argc, char *argv[]) {
	clear_array();
	starpu_omp_parallel_region(&parallel_region_1_cl, NULL, NULL, 0, 0, 1);
	check_array();
	return 0;

	clear_array();
	starpu_omp_parallel_region(&parallel_region_2_cl, NULL, NULL, 0, 0, 1);
	check_array();

	clear_array();
	starpu_omp_parallel_region(&parallel_region_3_cl, NULL, NULL, 0, 0, 1);
	check_array();

	clear_array();
	starpu_omp_parallel_region(&parallel_region_4_cl, NULL, NULL, 0, 0, 1);
	check_array();

	clear_array();
	starpu_omp_parallel_region(&parallel_region_5_cl, NULL, NULL, 0, 0, 1);
	check_array();

	clear_array();
	starpu_omp_parallel_region(&parallel_region_6_cl, NULL, NULL, 0, 0, 1);
	check_array();
	return 0;
}
#endif
