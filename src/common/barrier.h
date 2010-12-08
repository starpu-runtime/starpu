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

#ifndef __COMMON_BARRIER_H__
#define __COMMON_BARRIER_H__

#include <pthread.h>

typedef struct {
	int count;
	int reached;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
} _starpu_barrier_t;

int _starpu_barrier_init(_starpu_barrier_t *barrier, int count);

int _starpu_barrier_destroy(_starpu_barrier_t *barrier);

int _starpu_barrier_wait(_starpu_barrier_t *barrier);

#if !defined(PTHREAD_BARRIER_SERIAL_THREAD)
#  define PTHREAD_BARRIER_SERIAL_THREAD -1
#  define pthread_barrier_t _starpu_barrier_t
#  define pthread_barrier_init(b,a,c) _starpu_barrier_init(b, c)
#  define pthread_barrier_destroy(b) _starpu_barrier_destroy(b)
#  define pthread_barrier_wait(b) _starpu_barrier_wait(b)
#endif /* !PTHREAD_BARRIER_SERIAL_THREAD */

#endif // __COMMON_BARRIER_H__
