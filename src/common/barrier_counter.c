/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  INRIA
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

int _starpu_barrier_counter_init(struct _starpu_barrier_counter *barrier_c, int count)
{
	_starpu_barrier_init(&barrier_c->barrier, count);
	_STARPU_PTHREAD_COND_INIT(&barrier_c->cond2, NULL);
	return 0;
}

int _starpu_barrier_counter_destroy(struct _starpu_barrier_counter *barrier_c)
{
	_starpu_barrier_destroy(&barrier_c->barrier);
	_STARPU_PTHREAD_COND_DESTROY(&barrier_c->cond2);
	return 0;
}


int _starpu_barrier_counter_wait_for_empty_counter(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	_STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	while (barrier->reached_start > 0)
		_STARPU_PTHREAD_COND_WAIT(&barrier->cond, &barrier->mutex);

	_STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_wait_for_full_counter(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	_STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	while (barrier->reached_start < barrier->count)
		_STARPU_PTHREAD_COND_WAIT(&barrier_c->cond2, &barrier->mutex);

	_STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_decrement_until_empty_counter(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	_STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	if (--barrier->reached_start == 0)
		_STARPU_PTHREAD_COND_BROADCAST(&barrier->cond);

	_STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_increment_until_full_counter(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	_STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);
	
	if(++barrier->reached_start == barrier->count)
		_STARPU_PTHREAD_COND_BROADCAST(&barrier_c->cond2);

	_STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

int _starpu_barrier_counter_increment(struct _starpu_barrier_counter *barrier_c)
{
	struct _starpu_barrier *barrier = &barrier_c->barrier;
	_STARPU_PTHREAD_MUTEX_LOCK(&barrier->mutex);

	barrier->reached_start++;
	
	_STARPU_PTHREAD_MUTEX_UNLOCK(&barrier->mutex);
	return 0;
}

