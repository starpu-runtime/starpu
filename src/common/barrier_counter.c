/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/barrier_counter.h>

int _starpu_barrier_counter_init(struct _starpu_barrier_counter *barrier_c, unsigned count)
{
	_starpu_barrier_init(&barrier_c->barrier, count);
	barrier_c->min_threshold = 0;
	barrier_c->max_threshold = 0;
	STARPU_PTHREAD_COND_INIT(&barrier_c->cond2, NULL);
	return 0;
}

int _starpu_barrier_counter_destroy(struct _starpu_barrier_counter *barrier_c)
{
	_starpu_barrier_destroy(&barrier_c->barrier);
	STARPU_PTHREAD_COND_DESTROY(&barrier_c->cond2);
	return 0;
}


int _starpu_barrier_counter_wait_for_empty_counter(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	int ret;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	ret = barrier->reached_start;
	while (barrier->reached_start > 0)
		STARPU_PTHREAD_COND_WAIT(&barrier->cond, &barrier->mutex);

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return ret;
}

int _starpu_barrier_counter_wait_until_counter_reaches_down_to_n(struct _starpu_barrier_counter *barrier_c, unsigned n)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	while (barrier->reached_start > n)
	{
		if (barrier_c->max_threshold < n)
			barrier_c->max_threshold = n;
		STARPU_PTHREAD_COND_WAIT(&barrier->cond, &barrier->mutex);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_wait_until_counter_reaches_up_to_n(struct _starpu_barrier_counter *barrier_c, unsigned n)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	while (barrier->reached_start < n)
	{
		if (barrier_c->min_threshold > n)
			barrier_c->min_threshold = n;
		STARPU_PTHREAD_COND_WAIT(&barrier_c->cond2, &barrier->mutex);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_wait_for_full_counter(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	while (barrier->reached_start < barrier->count)
		STARPU_PTHREAD_COND_WAIT(&barrier_c->cond2, &barrier->mutex);

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_decrement_until_empty_counter(struct _starpu_barrier_counter *barrier_c, double flops)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	int ret = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	barrier->reached_flops -= flops;
	if (--barrier->reached_start == 0)
	{
		ret = 1;
		STARPU_PTHREAD_COND_BROADCAST(&barrier->cond);
	}
	if (barrier_c->max_threshold && barrier->reached_start == barrier_c->max_threshold)
	{
		/* have those not happy enough tell us how much again */
		barrier_c->max_threshold = 0;
		STARPU_PTHREAD_COND_BROADCAST(&barrier->cond);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return ret;
}

int _starpu_barrier_counter_increment_until_full_counter(struct _starpu_barrier_counter *barrier_c, double flops)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	int ret = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	barrier->reached_flops += flops;
	if(++barrier->reached_start == barrier->count)
	{
		ret = 1;
		STARPU_PTHREAD_COND_BROADCAST(&barrier_c->cond2);
	}
	if (barrier_c->min_threshold && barrier->reached_start == barrier_c->min_threshold)
	{
		/* have those not happy enough tell us how much again */
		barrier_c->min_threshold = 0;
		STARPU_PTHREAD_COND_BROADCAST(&barrier_c->cond2);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return ret;
}

int _starpu_barrier_counter_increment(struct _starpu_barrier_counter *barrier_c, double flops)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	barrier->reached_start++;
	barrier->reached_flops += flops;
	STARPU_PTHREAD_COND_BROADCAST(&barrier_c->cond2);
	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_check(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	if(barrier->reached_start == 0)
		STARPU_PTHREAD_COND_BROADCAST(&barrier->cond);

	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_get_reached_start(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	int ret;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);
	ret = barrier->reached_start;
	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return ret;
}

double _starpu_barrier_counter_get_reached_flops(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	double ret;
	STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);
	ret = barrier->reached_flops;
	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return ret;
}
