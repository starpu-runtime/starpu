/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <starpu-mutex.h>

void init_mutex(starpu_mutex *m)
{
	/* this is free at first */
	m->taken = 0;
}

inline int take_mutex_try(starpu_mutex *m)
{
	uint32_t prev;
	prev = __sync_lock_test_and_set(&m->taken, 1);
	return (prev == 0)?0:-1;
}

inline void take_mutex(starpu_mutex *m)
{
	uint32_t prev;
	do {
		prev = __sync_lock_test_and_set(&m->taken, 1);
	} while (prev);
}

inline void release_mutex(starpu_mutex *m)
{
	m->taken = 0;
}
