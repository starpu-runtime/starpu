/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2014  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2013, 2014  CNRS
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

#include <common/starpu_spinlock.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/fxt.h>
#include <common/thread.h>

int _starpu_spin_init(struct _starpu_spinlock *lock)
{
#if defined(STARPU_SPINLOCK_CHECK)
	starpu_pthread_mutexattr_t errcheck_attr;
//	memcpy(&lock->errcheck_lock, PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP, sizeof(PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP));
	int ret;
	ret = starpu_pthread_mutexattr_init(&errcheck_attr);
	STARPU_CHECK_RETURN_VALUE(ret, "pthread_mutexattr_init");

	ret = starpu_pthread_mutexattr_settype(&errcheck_attr, PTHREAD_MUTEX_ERRORCHECK);
	STARPU_ASSERT(!ret);

	ret = starpu_pthread_mutex_init(&lock->errcheck_lock, &errcheck_attr);
	starpu_pthread_mutexattr_destroy(&errcheck_attr);
	return ret;
#else
	int ret = starpu_pthread_spin_init(&lock->lock, 0);
	STARPU_ASSERT(!ret);
	return ret;
#endif
}

int _starpu_spin_destroy(struct _starpu_spinlock *lock STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_SPINLOCK_CHECK)
	return starpu_pthread_mutex_destroy(&lock->errcheck_lock);
#else
	return starpu_pthread_spin_destroy(&lock->lock);
#endif
}

#undef _starpu_spin_lock
int _starpu_spin_lock(struct _starpu_spinlock *lock)
{
#if defined(STARPU_SPINLOCK_CHECK)
	int ret = starpu_pthread_mutex_lock(&lock->errcheck_lock);
	STARPU_ASSERT(!ret);
	return ret;
#else
	int ret = starpu_pthread_spin_lock(&lock->lock);
	STARPU_ASSERT(!ret);
	return ret;
#endif
}

int _starpu_spin_checklocked(struct _starpu_spinlock *lock)
{
#if defined(STARPU_SPINLOCK_CHECK)
	int ret = starpu_pthread_mutex_trylock(&lock->errcheck_lock);
	STARPU_ASSERT(ret != 0);
	return ret == 0;
#else
	return _starpu_pthread_spin_checklocked(&lock->lock);
#endif
}

#undef _starpu_spin_trylock
int _starpu_spin_trylock(struct _starpu_spinlock *lock)
{
#if defined(STARPU_SPINLOCK_CHECK)
	int ret = starpu_pthread_mutex_trylock(&lock->errcheck_lock);
	STARPU_ASSERT(!ret || (ret == EBUSY));
	return ret;
#else
	int ret = starpu_pthread_spin_trylock(&lock->lock);
	STARPU_ASSERT(!ret || (ret == EBUSY));
	return ret;
#endif
}

#undef _starpu_spin_unlock
int _starpu_spin_unlock(struct _starpu_spinlock *lock STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_SPINLOCK_CHECK)
	int ret = starpu_pthread_mutex_unlock(&lock->errcheck_lock);
	STARPU_ASSERT(!ret);
	return ret;
#else
	int ret = starpu_pthread_spin_unlock(&lock->lock);
	STARPU_ASSERT(!ret);
	return ret;
#endif
}
