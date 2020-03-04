/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __COMMON_THREAD_H__
#define __COMMON_THREAD_H__

/** @file */

#include <common/utils.h>

#if defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
int _starpu_pthread_spin_do_lock(starpu_pthread_spinlock_t *lock);
#endif

#if defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK)

static inline int _starpu_pthread_spin_init(starpu_pthread_spinlock_t *lock, int pshared STARPU_ATTRIBUTE_UNUSED)
{
	lock->taken = 0;
	return 0;
}
#define starpu_pthread_spin_init _starpu_pthread_spin_init

static inline int _starpu_pthread_spin_destroy(starpu_pthread_spinlock_t *lock STARPU_ATTRIBUTE_UNUSED)
{
	/* we don't do anything */
	return 0;
}
#define starpu_pthread_spin_destroy _starpu_pthread_spin_destroy

static inline int _starpu_pthread_spin_lock(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	while (1)
	{
		if (STARPU_LIKELY(!lock->taken))
		{
			lock->taken = 1;
			return 0;
		}
		/* Give hand to another thread, hopefully the one which has the
		 * spinlock and probably just has also a short-lived mutex. */
		starpu_sleep(0.000001);
		STARPU_UYIELD();
	}
#elif defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
	if (STARPU_LIKELY(STARPU_VAL_COMPARE_AND_SWAP(&lock->taken, 0, 1) == 0))
		/* Got it on first try! */
		return 0;

	return _starpu_pthread_spin_do_lock(lock);
#else /* !SIMGRID && !LINUX */
	uint32_t prev;
	do
	{
		prev = STARPU_TEST_AND_SET(&lock->taken, 1);
		if (STARPU_UNLIKELY(prev))
			STARPU_UYIELD();
	}
	while (STARPU_UNLIKELY(prev));
	return 0;
#endif
}
#define starpu_pthread_spin_lock _starpu_pthread_spin_lock

static inline void _starpu_pthread_spin_checklocked(starpu_pthread_spinlock_t *lock STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	STARPU_ASSERT(lock->taken);
#elif defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
	STARPU_ASSERT(lock->taken == 1 || lock->taken == 2);
#else
	STARPU_ASSERT(lock->taken);
#endif
}

static inline int _starpu_pthread_spin_trylock(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	if (STARPU_UNLIKELY(lock->taken))
		return EBUSY;
	lock->taken = 1;
	return 0;
#elif defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
	unsigned prev;
	prev = STARPU_VAL_COMPARE_AND_SWAP(&lock->taken, 0, 1);
	return (prev == 0)?0:EBUSY;
#else /* !SIMGRID && !LINUX */
	uint32_t prev;
	prev = STARPU_TEST_AND_SET(&lock->taken, 1);
	return (prev == 0)?0:EBUSY;
#endif
}
#define starpu_pthread_spin_trylock _starpu_pthread_spin_trylock

#if defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
void _starpu_pthread_spin_do_unlock(starpu_pthread_spinlock_t *lock);
#endif

static inline int _starpu_pthread_spin_unlock(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	lock->taken = 0;
#elif defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
	STARPU_ASSERT(lock->taken != 0);
	STARPU_SYNCHRONIZE();
	unsigned next = STARPU_ATOMIC_ADD(&lock->taken, -1);
	if (STARPU_LIKELY(next == 0))
		/* Nobody to wake, we are done */
		return 0;
	_starpu_pthread_spin_do_unlock(lock);
#else /* !SIMGRID && !LINUX */
	STARPU_RELEASE(&lock->taken);
#endif
	return 0;
}
#define starpu_pthread_spin_unlock _starpu_pthread_spin_unlock

#else /* defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK) */

static inline void _starpu_pthread_spin_checklocked(starpu_pthread_spinlock_t *lock STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(pthread_spin_trylock((pthread_spinlock_t *)lock) != 0);
}

#endif /* defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK) */


#endif /* __COMMON_THREAD_H__ */


