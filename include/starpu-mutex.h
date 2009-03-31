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

#ifndef __MUTEX_H__
#define __MUTEX_H__

#include <stdint.h>

typedef struct mutex_t {
	/* we only have a trivial implementation yet ! */
	volatile uint32_t taken __attribute__ ((aligned(16)));
} mutex;

void init_mutex(mutex *m);
void take_mutex(mutex *m);
int take_mutex_try(mutex *m);
void release_mutex(mutex *m);

#endif
