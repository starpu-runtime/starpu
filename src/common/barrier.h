/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __COMMON_BARRIER_H__
#define __COMMON_BARRIER_H__

#include <starpu_thread.h>

/** @file */

struct _starpu_barrier
{
	unsigned count;
	unsigned reached_start;
	unsigned reached_exit;
	double reached_flops;
	starpu_pthread_mutex_t mutex;
	starpu_pthread_mutex_t mutex_exit;
	starpu_pthread_cond_t cond;
};

int _starpu_barrier_init(struct _starpu_barrier *barrier, int count);

int _starpu_barrier_destroy(struct _starpu_barrier *barrier);

int _starpu_barrier_wait(struct _starpu_barrier *barrier);

#endif // __COMMON_BARRIER_H__
