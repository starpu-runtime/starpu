/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifndef __STARPU_SPINLOCK_H__
#define __STARPU_SPINLOCK_H__

/** @file */

#include <errno.h>
#include <stdint.h>
#include <common/config.h>
#include <common/fxt.h>
#include <common/thread.h>
#include <starpu.h>

#ifdef STARPU_SPINLOCK_CHECK

/* We don't care about performance */

struct _starpu_spinlock
{
	starpu_pthread_mutex_t errcheck_lock;
	const char *last_taker;
};

int _starpu_spin_init(struct _starpu_spinlock *lock);
int _starpu_spin_destroy(struct _starpu_spinlock *lock);

static inline int __starpu_spin_lock(struct _starpu_spinlock *lock, const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED, const char *func STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_LOCKING_SPINLOCK(file, line);
	int ret = starpu_pthread_mutex_lock(&lock->errcheck_lock);
	STARPU_ASSERT(!ret);
	lock->last_taker = func;
	_STARPU_TRACE_SPINLOCK_LOCKED(file, line);
	return ret;
}

static inline void _starpu_spin_checklocked(struct _starpu_spinlock *lock STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(starpu_pthread_mutex_trylock(&lock->errcheck_lock) != 0);
}

static inline int __starpu_spin_trylock(struct _starpu_spinlock *lock, const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED, const char *func STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_TRYLOCK_SPINLOCK(file, line);
	int ret = starpu_pthread_mutex_trylock(&lock->errcheck_lock);
	STARPU_ASSERT(!ret || (ret == EBUSY));
	if (STARPU_LIKELY(!ret))
	{
		lock->last_taker = func;
		_STARPU_TRACE_SPINLOCK_LOCKED(file, line);
	}
	return ret;
}

static inline int __starpu_spin_unlock(struct _starpu_spinlock *lock, const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED, const char *func STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_UNLOCKING_SPINLOCK(file, line);
	int ret = starpu_pthread_mutex_unlock(&lock->errcheck_lock);
	STARPU_ASSERT(!ret);
	_STARPU_TRACE_SPINLOCK_UNLOCKED(file, line);
	return ret;
}
#else

/* We do care about performance, inline as much as possible */

struct _starpu_spinlock
{
	starpu_pthread_spinlock_t lock;
};

static inline int _starpu_spin_init(struct _starpu_spinlock *lock)
{
	int ret = starpu_pthread_spin_init(&lock->lock, 0);
	STARPU_ASSERT(!ret);
	return ret;
}

#define _starpu_spin_destroy(_lock) starpu_pthread_spin_destroy(&(_lock)->lock)

static inline int __starpu_spin_lock(struct _starpu_spinlock *lock, const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED, const char *func STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_LOCKING_SPINLOCK(file, line);
	int ret = starpu_pthread_spin_lock(&lock->lock);
	STARPU_ASSERT(!ret);
	_STARPU_TRACE_SPINLOCK_LOCKED(file, line);
	return ret;
}

#define _starpu_spin_checklocked(_lock) _starpu_pthread_spin_checklocked(&(_lock)->lock)

static inline int __starpu_spin_trylock(struct _starpu_spinlock *lock, const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED, const char *func STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_TRYLOCK_SPINLOCK(file, line);
	int ret = starpu_pthread_spin_trylock(&lock->lock);
	STARPU_ASSERT(!ret || (ret == EBUSY));
	if (STARPU_LIKELY(!ret))
		_STARPU_TRACE_SPINLOCK_LOCKED(file, line);
	return ret;
}

static inline int __starpu_spin_unlock(struct _starpu_spinlock *lock, const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED, const char *func STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_UNLOCKING_SPINLOCK(file, line);
	int ret = starpu_pthread_spin_unlock(&lock->lock);
	STARPU_ASSERT(!ret);
	_STARPU_TRACE_SPINLOCK_UNLOCKED(file, line);
	return ret;
}
#endif

#define _starpu_spin_lock(lock) \
	__starpu_spin_lock(lock, __FILE__, __LINE__, __starpu_func__)
#define _starpu_spin_trylock(lock) \
	__starpu_spin_trylock(lock, __FILE__, __LINE__, __starpu_func__)
#define _starpu_spin_unlock(lock) \
	__starpu_spin_unlock(lock, __FILE__, __LINE__, __starpu_func__)

#define STARPU_SPIN_MAXTRY 10 

#endif // __STARPU_SPINLOCK_H__
