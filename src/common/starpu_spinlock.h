/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2013  Centre National de la Recherche Scientifique
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

#include <errno.h>
#include <stdint.h>
#include <starpu_thread.h>
#include <common/config.h>

struct _starpu_spinlock
{
#ifdef STARPU_SIMGRID
	int taken;
#elif defined(STARPU_SPINLOCK_CHECK)
	starpu_pthread_mutexattr_t errcheck_attr;
	starpu_pthread_mutex_t errcheck_lock;
#elif defined(HAVE_PTHREAD_SPIN_LOCK)
	_starpu_pthread_spinlock_t lock;
#else
	/* we only have a trivial implementation yet ! */
	uint32_t taken __attribute__ ((aligned(16)));
#endif
#ifdef STARPU_SPINLOCK_CHECK
	const char *last_taker;
#endif
};

int _starpu_spin_init(struct _starpu_spinlock *lock);
int _starpu_spin_destroy(struct _starpu_spinlock *lock);

int _starpu_spin_lock(struct _starpu_spinlock *lock);
#if defined(STARPU_SPINLOCK_CHECK)
#define _starpu_spin_lock(lock) ({ \
	_starpu_spin_lock(lock); \
	(lock)->last_taker = __func__; \
	0; \
})
#endif
int _starpu_spin_trylock(struct _starpu_spinlock *lock);
#if defined(STARPU_SPINLOCK_CHECK)
#define _starpu_spin_trylock(lock) ({ \
	int err = _starpu_spin_trylock(lock); \
	if (!err) \
		(lock)->last_taker = __func__; \
	err; \
})
#endif
int _starpu_spin_checklocked(struct _starpu_spinlock *lock);
int _starpu_spin_unlock(struct _starpu_spinlock *lock);

#endif // __STARPU_SPINLOCK_H__
