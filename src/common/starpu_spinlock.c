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

#include <common/starpu_spinlock.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/fxt.h>
#include <common/thread.h>

#if defined(STARPU_SPINLOCK_CHECK)
int _starpu_spin_init(struct _starpu_spinlock *lock)
{
	starpu_pthread_mutexattr_t errcheck_attr;
	int ret;
	ret = starpu_pthread_mutexattr_init(&errcheck_attr);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_pthread_mutexattr_init");

	ret = starpu_pthread_mutexattr_settype(&errcheck_attr, PTHREAD_MUTEX_ERRORCHECK);
	STARPU_ASSERT(!ret);

	ret = starpu_pthread_mutex_init(&lock->errcheck_lock, &errcheck_attr);
	starpu_pthread_mutexattr_destroy(&errcheck_attr);
	return ret;
}

int _starpu_spin_destroy(struct _starpu_spinlock *lock)
{
	return starpu_pthread_mutex_destroy(&lock->errcheck_lock);
}
#endif
