/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <common/barrier.h>

int _starpu_barrier_init(_starpu_barrier_t *barrier, int count)
{
	barrier->count = count;
	barrier->reached = 0;
	pthread_mutex_init(&barrier->mutex,NULL);
	pthread_cond_init(&barrier->cond,NULL);
	return 0;
}

int _starpu_barrier_destroy(_starpu_barrier_t *barrier)
{
	pthread_mutex_destroy(&barrier->mutex);
	pthread_cond_destroy(&barrier->cond);
	return 0;
}

int _starpu_barrier_wait(_starpu_barrier_t *barrier)
{
	pthread_mutex_lock(&barrier->mutex);
	barrier->reached++;
	if (barrier->reached == barrier->count)
	{
		barrier->reached = 0;
		pthread_cond_broadcast(&barrier->cond);
	}
	else
	{
		pthread_cond_wait(&barrier->cond,&barrier->mutex);
	}
	pthread_mutex_unlock(&barrier->mutex);
	return 0;
}
