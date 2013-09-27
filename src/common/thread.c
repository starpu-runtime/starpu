/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include <common/thread.h>
#include <core/simgrid.h>

#ifdef STARPU_SIMGRID
#include <xbt/synchro_core.h>
#endif

#ifdef STARPU_SIMGRID

extern int _starpu_simgrid_thread_start(int argc, char *argv[]);

int starpu_pthread_create_on(char *name, starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg, int where)
{
	struct _starpu_pthread_args *_args = malloc(sizeof(*_args));
	xbt_dynar_t _hosts;
	_args->f = start_routine;
	_args->arg = arg;
	_hosts = MSG_hosts_as_dynar();
	MSG_process_create(name, _starpu_simgrid_thread_start, _args,
			   xbt_dynar_get_as(_hosts, (where), msg_host_t));
	xbt_dynar_free(&_hosts);
	return 0;
}

int starpu_pthread_create(starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)
{
	return starpu_pthread_create_on("", thread, attr, start_routine, arg, 0);
}

int starpu_pthread_join(starpu_pthread_t thread, void **retval)
{
#ifdef STARPU_DEVEL
#warning TODO: use a simgrid_join when it becomes available
#endif
	MSG_process_sleep(1);
	return 0;
}

int starpu_pthread_attr_init(starpu_pthread_attr_t *attr)
{
	return 0;
}

int starpu_pthread_attr_destroy(starpu_pthread_attr_t *attr)
{
	return 0;
}

int starpu_pthread_attr_setdetachstate(starpu_pthread_attr_t *attr, int detachstate)
{
	return 0;
}

int starpu_pthread_mutex_init(starpu_pthread_mutex_t *mutex, const starpu_pthread_mutexattr_t *mutexattr)
{
	*mutex = xbt_mutex_init();
	return 0;
}

int starpu_pthread_mutex_destroy(starpu_pthread_mutex_t *mutex)
{
	if (*mutex)
		xbt_mutex_destroy(*mutex);
	return 0;
}

int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_LOCKING_MUTEX();

	if (!*mutex) STARPU_PTHREAD_MUTEX_INIT(mutex, NULL);

	xbt_mutex_acquire(*mutex);

	_STARPU_TRACE_MUTEX_LOCKED();

	return 0;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_UNLOCKING_MUTEX();

	xbt_mutex_release(*mutex);

	_STARPU_TRACE_MUTEX_UNLOCKED();

	return 0;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_TRYLOCK_MUTEX();

	xbt_mutex_acquire(*mutex);

	_STARPU_TRACE_MUTEX_LOCKED();

	return 0;
}

static int used_key[MAX_TSD];

int starpu_pthread_key_create(starpu_pthread_key_t *key, void (*destr_function) (void *))
{
	unsigned i;

	/* Note: no synchronization here, we are actually monothreaded anyway. */
	for (i = 0; i < MAX_TSD; i++)
		if (!used_key[i])
		{
			used_key[i] = 1;
			break;
		}
	STARPU_ASSERT(i < MAX_TSD);
	*key = i;
	return 0;
}

int starpu_pthread_key_delete(starpu_pthread_key_t key)
{
	used_key[key] = 0;
	return 0;
}

int starpu_pthread_setspecific(starpu_pthread_key_t key, const void *pointer)
{
	void **array = MSG_host_get_data(MSG_host_self());
	array[key] = pointer;
	return 0;
}

void* starpu_pthread_getspecific(starpu_pthread_key_t key)
{
	void **array = MSG_host_get_data(MSG_host_self());
	return array[key];
}

int starpu_pthread_cond_init(starpu_pthread_cond_t *cond, starpu_pthread_condattr_t *cond_attr)
{
	*cond = xbt_cond_init();
	return 0;
}

int starpu_pthread_cond_signal(starpu_pthread_cond_t *cond)
{
	if (!*cond)
		STARPU_PTHREAD_COND_INIT(cond, NULL);
	xbt_cond_signal(*cond);
	return 0;
}

int starpu_pthread_cond_broadcast(starpu_pthread_cond_t *cond)
{
	if (!*cond)
		STARPU_PTHREAD_COND_INIT(cond, NULL);
	xbt_cond_broadcast(*cond);
	return 0;
}

int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_COND_WAIT_BEGIN();

	if (!*cond)
		STARPU_PTHREAD_COND_INIT(cond, NULL);
	xbt_cond_wait(*cond, *mutex);

	_STARPU_TRACE_COND_WAIT_END();

	return 0;
}

int starpu_pthread_cond_destroy(starpu_pthread_cond_t *cond)
{
	if (*cond)
		xbt_cond_destroy(*cond);
	return 0;
}

int starpu_pthread_rwlock_init(starpu_pthread_rwlock_t *restrict rwlock, const starpu_pthread_rwlockattr_t *restrict attr)
{
	return starpu_pthread_mutex_init(rwlock, attr);
}

int starpu_pthread_rwlock_destroy(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_destroy(rwlock);
}

int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_RDLOCKING_RWLOCK();

 	int p_ret = starpu_pthread_mutex_lock(rwlock);

	_STARPU_TRACE_RWLOCK_RDLOCKED();

	return p_ret;
}

int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_WRLOCKING_RWLOCK();

 	int p_ret = starpu_pthread_mutex_lock(rwlock);

	_STARPU_TRACE_RWLOCK_WRLOCKED();

	return p_ret;
}

int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_UNLOCKING_RWLOCK();

 	int p_ret = starpu_pthread_mutex_unlock(rwlock);

	_STARPU_TRACE_RWLOCK_UNLOCKED();

	return p_ret;
}

#elif !defined(_MSC_VER) /* !STARPU_SIMGRID */

int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_LOCKING_MUTEX();

	int p_ret = pthread_mutex_lock(mutex);

	_STARPU_TRACE_MUTEX_LOCKED();

	return p_ret;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_UNLOCKING_MUTEX();

	int p_ret = pthread_mutex_unlock(mutex);

	_STARPU_TRACE_MUTEX_UNLOCKED();

	return p_ret;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	int ret;
	_STARPU_TRACE_TRYLOCK_MUTEX();

	ret = pthread_mutex_trylock(mutex);

	if (!ret)
		_STARPU_TRACE_MUTEX_LOCKED();

	return ret;
}

int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_COND_WAIT_BEGIN();

 	int p_ret = pthread_cond_wait(cond, mutex);

	_STARPU_TRACE_COND_WAIT_END();

	return p_ret;
}

int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_RDLOCKING_RWLOCK();

 	int p_ret = pthread_rwlock_rdlock(rwlock);

	_STARPU_TRACE_RWLOCK_RDLOCKED();

	return p_ret;
}

int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_WRLOCKING_RWLOCK();

 	int p_ret = pthread_rwlock_wrlock(rwlock);

	_STARPU_TRACE_RWLOCK_WRLOCKED();

	return p_ret;
}

int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_UNLOCKING_RWLOCK();

 	int p_ret = pthread_rwlock_unlock(rwlock);

	_STARPU_TRACE_RWLOCK_UNLOCKED();

	return p_ret;
}

#endif /* STARPU_SIMGRID, _MSC_VER */
