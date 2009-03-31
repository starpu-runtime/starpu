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

/**
 * A dummy implementation of a rw_lock using spinlocks ...
 */ 

#include "rwlock.h"

static void _take_busy_lock(rw_lock *lock)
{
	uint32_t prev;
	do {
		prev = __sync_lock_test_and_set(&lock->busy, 1);
	} while (prev);
}

static void _release_busy_lock(rw_lock *lock)
{
	lock->busy = 0;
}

void init_rw_lock(rw_lock *lock)
{
	STARPU_ASSERT(lock);

	lock->writer = 0;
	lock->readercnt = 0;
	lock->busy = 0;
}


int take_rw_lock_write_try(rw_lock *lock)
{
	_take_busy_lock(lock);
	
	if (lock->readercnt > 0 || lock->writer)
	{
		/* fail to take the lock */
		_release_busy_lock(lock);
		return -1;
	}
	else {
		STARPU_ASSERT(lock->readercnt == 0);
		STARPU_ASSERT(lock->writer == 0);

		/* no one was either writing nor reading */
		lock->writer = 1;
		_release_busy_lock(lock);
		return 0;
	}
}

int take_rw_lock_read_try(rw_lock *lock)
{
	_take_busy_lock(lock);

	if (lock->writer)
	{
		/* there is a writer ... */
		_release_busy_lock(lock);
		return -1;
	}
	else {
		STARPU_ASSERT(lock->writer == 0);

		/* no one is writing */
		/* XXX check wrap arounds ... */
		lock->readercnt++;
		_release_busy_lock(lock);

		return 0;
	}
}



void take_rw_lock_write(rw_lock *lock)
{
	do {
		_take_busy_lock(lock);
		
		if (lock->readercnt > 0 || lock->writer)
		{
			/* fail to take the lock */
			_release_busy_lock(lock);
		}
		else {
			STARPU_ASSERT(lock->readercnt == 0);
			STARPU_ASSERT(lock->writer == 0);
	
			/* no one was either writing nor reading */
			lock->writer = 1;
			_release_busy_lock(lock);
			return;
		}
	} while (1);
}

void take_rw_lock_read(rw_lock *lock)
{
	do {
		_take_busy_lock(lock);

		if (lock->writer)
		{
			/* there is a writer ... */
			_release_busy_lock(lock);
		}
		else {
			STARPU_ASSERT(lock->writer == 0);

			/* no one is writing */
			/* XXX check wrap arounds ... */
			lock->readercnt++;
			_release_busy_lock(lock);

			return;
		}
	} while (1);
}

void release_rw_lock(rw_lock *lock)
{
	_take_busy_lock(lock);
	/* either writer or reader (exactly one !) */
	if (lock->writer) 
	{
		STARPU_ASSERT(lock->readercnt == 0);
		lock->writer = 0;
	}
	else {
		/* reading mode */
		STARPU_ASSERT(lock->writer == 0);
		lock->readercnt--;
	}
	_release_busy_lock(lock);
}
