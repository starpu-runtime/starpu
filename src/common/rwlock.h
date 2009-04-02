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

#ifndef __RWLOCKS_H__
#define __RWLOCKS_H__

#include <stdint.h>
#include <starpu.h>

typedef struct rw_lock_t {
	uint32_t busy;
	uint8_t writer;
	uint16_t readercnt;
} rw_lock;

void init_rw_lock(rw_lock *lock);
void take_rw_lock_write(rw_lock *lock);
void take_rw_lock_read(rw_lock *lock);
int take_rw_lock_write_try(rw_lock *lock);
int take_rw_lock_read_try(rw_lock *lock);
void release_rw_lock(rw_lock *lock);

///* make sure to have the lock before using that function */
//inline uint8_t rw_lock_is_writer(rw_lock *lock);
//unsigned is_rw_lock_referenced(rw_lock *lock);

#endif
