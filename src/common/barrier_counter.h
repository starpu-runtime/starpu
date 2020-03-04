/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __BARRIER_COUNTER_H__
#define __BARRIER_COUNTER_H__

/** @file */

#include <common/utils.h>
#include <common/barrier.h>

struct _starpu_barrier_counter
{
	struct _starpu_barrier barrier;
	unsigned min_threshold;
	unsigned max_threshold;
	starpu_pthread_cond_t cond2;
};

int _starpu_barrier_counter_init(struct _starpu_barrier_counter *barrier_c, unsigned count);

int _starpu_barrier_counter_destroy(struct _starpu_barrier_counter *barrier_c);

int _starpu_barrier_counter_wait_for_empty_counter(struct _starpu_barrier_counter *barrier_c);

int _starpu_barrier_counter_wait_until_counter_reaches_down_to_n(struct _starpu_barrier_counter *barrier_c, unsigned n);
int _starpu_barrier_counter_wait_until_counter_reaches_up_to_n(struct _starpu_barrier_counter *barrier_c, unsigned n);

int _starpu_barrier_counter_wait_for_full_counter(struct _starpu_barrier_counter *barrier_c);

int _starpu_barrier_counter_decrement_until_empty_counter(struct _starpu_barrier_counter *barrier_c, double flops);

int _starpu_barrier_counter_increment_until_full_counter(struct _starpu_barrier_counter *barrier_c, double flops);

int _starpu_barrier_counter_increment(struct _starpu_barrier_counter *barrier_c, double flops);

int _starpu_barrier_counter_check(struct _starpu_barrier_counter *barrier_c);

int _starpu_barrier_counter_get_reached_start(struct _starpu_barrier_counter *barrier_c);

double _starpu_barrier_counter_get_reached_flops(struct _starpu_barrier_counter *barrier_c);
#endif
