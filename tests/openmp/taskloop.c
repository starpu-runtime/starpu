/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdio.h>
#include "../helper.h"

/*
 * Check the OpenMP orphaned task support.
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

void taskloop_callback(unsigned long long begin_i, unsigned long long end_i)
{
	int worker_id;
	pthread_t tid;
	tid = pthread_self();
	worker_id = starpu_worker_get_id();
	printf ("begin = %llu , end = %llu, %p\n", begin_i, end_i, (void *)starpu_task_get_current());
}

void taskloop_callback_wrapper(void *buffers[], void *_args)
{
	(void) buffers;
	struct starpu_omp_task_region_attr * args = _args;
	taskloop_callback(args->begin_i, args->end_i);
}

int
main (void)
{
	struct starpu_omp_task_region_attr attr;
	memset(&attr, 0, sizeof(attr));
#ifdef STARPU_SIMGRID
	attr.cl.model         = &starpu_perfmodel_nop;
#endif
	attr.cl.flags         = STARPU_CODELET_SIMGRID_EXECUTE;
	attr.cl.cpu_funcs[0]  = taskloop_callback_wrapper;
	attr.cl_arg           = &attr;
	attr.cl.where         = STARPU_CPU;
	attr.if_clause        = 1;
	attr.final_clause     = 0;
	attr.untied_clause    = 1;
	attr.mergeable_clause = 0;
	attr.nogroup_clause   = 0;
	attr.is_loop          = 0;
	attr.collapse         = 0;
	attr.num_tasks        = 5;
	attr.nb_iterations    = 400;
	attr.grainsize        = 130;

	starpu_omp_taskloop_inline_begin(&attr);
	starpu_omp_taskloop_inline_end(&attr);
	return 0;
}
#endif
