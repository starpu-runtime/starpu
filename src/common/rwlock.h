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

#ifndef __RWLOCKS_H__
#define __RWLOCKS_H__

#include <stdint.h>
#include <starpu.h>

/* Dummy implementation of a RW-lock using a spinlock. */
typedef struct starpu_rw_lock_s {
	uint32_t busy;
	uint8_t writer;
	uint16_t readercnt;
} starpu_rw_lock_t;

/* Initialize the RW-lock */
void _starpu_init_rw_lock(starpu_rw_lock_t *lock);

/* Grab the RW-lock in a write mode */
void _starpu_take_rw_lock_write(starpu_rw_lock_t *lock);

/* Grab the RW-lock in a read mode */
void _starpu_take_rw_lock_read(starpu_rw_lock_t *lock);

/* Try to grab the RW-lock in a write mode. Returns 0 in case of success, -1
 * otherwise. */
int _starpu_take_rw_lock_write_try(starpu_rw_lock_t *lock);

/* Try to grab the RW-lock in a read mode. Returns 0 in case of success, -1
 * otherwise. */
int _starpu_take_rw_lock_read_try(starpu_rw_lock_t *lock);

/* Unlock the RW-lock. */
void _starpu_release_rw_lock(starpu_rw_lock_t *lock);

#endif
