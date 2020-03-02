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
 * Check the OpenMP ordered parallel for support.
 */

#if !defined(STARPU_OPENMP)
int main(void)
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
	if (ret == -EINVAL) exit(STARPU_TEST_SKIPPED);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
}

__attribute__((destructor))
static void omp_destructor(void)
{
	starpu_omp_shutdown();
}

struct s_ordered_arg
{
	const char *msg;
	unsigned long long i;
};

void ordered_f(void *_arg)
{
	struct s_ordered_arg *arg = _arg;
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf("[tid %p] task thread = %d, for [%s] iteration (ordered) %llu\n", (void *)tid, worker_id, arg->msg, arg->i);
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
		struct s_ordered_arg ordered_arg = { arg, i };
		array[i] = 1;
		starpu_omp_ordered(ordered_f, &ordered_arg);
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
main (void)
{
	struct starpu_omp_parallel_region_attr attr;

	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model        = &starpu_perfmodel_nop;
#endif
	attr.cl.flags        = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.where        = STARPU_CPU;
	attr.if_clause       = 1;

	clear_array();
	attr.cl.cpu_funcs[0] = parallel_region_1_f;
	starpu_omp_parallel_region(&attr);
	check_array();

	clear_array();
	attr.cl.cpu_funcs[0] = parallel_region_2_f;
	starpu_omp_parallel_region(&attr);
	check_array();

	clear_array();
	attr.cl.cpu_funcs[0] = parallel_region_3_f;
	starpu_omp_parallel_region(&attr);
	check_array();

	clear_array();
	attr.cl.cpu_funcs[0] = parallel_region_4_f;
	starpu_omp_parallel_region(&attr);
	check_array();

	clear_array();
	attr.cl.cpu_funcs[0] = parallel_region_5_f;
	starpu_omp_parallel_region(&attr);
	check_array();

	clear_array();
	attr.cl.cpu_funcs[0] = parallel_region_6_f;
	starpu_omp_parallel_region(&attr);
	check_array();
	return 0;
}
#endif
