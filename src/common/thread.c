/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2016, 2019  Université de Bordeaux
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
#ifdef STARPU_HAVE_XBT_SYNCHRO_H
#include <xbt/synchro.h>
#else
#include <xbt/synchro_core.h>
#endif
#endif

#ifdef STARPU_SIMGRID

extern int _starpu_simgrid_thread_start(int argc, char *argv[]);

int starpu_pthread_create_on(char *name, starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg, msg_host_t host)
{
	char **_args = malloc(3*sizeof(char*));
	asprintf(&_args[0], "%p", start_routine);
	asprintf(&_args[1], "%p", arg);
	_args[2] = NULL;
	if (!host)
		host = MSG_get_host_by_name("MAIN");
	*thread = MSG_process_create_with_arguments(name, _starpu_simgrid_thread_start, calloc(MAX_TSD, sizeof(void*)), host, 2, _args);
	return 0;
}

int starpu_pthread_create(starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)
{
	return starpu_pthread_create_on("", thread, attr, start_routine, arg, NULL);
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
	/* Note: this is actually safe, because simgrid only preempts within
	 * simgrid functions */
	if (!*mutex) {
		/* Here we may get preempted */
		xbt_mutex_t new_mutex = xbt_mutex_init();
		if (!*mutex)
			*mutex = new_mutex;
		else
			/* Somebody already initialized it while we were
			 * calling xbt_mutex_init, this one is now useless */
			xbt_mutex_destroy(new_mutex);
	}
	xbt_mutex_acquire(*mutex);
	return 0;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	xbt_mutex_release(*mutex);
	return 0;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	int ret;
#ifdef HAVE_XBT_MUTEX_TRY_ACQUIRE
	ret = xbt_mutex_try_acquire(*mutex);
#else
	ret = simcall_mutex_trylock((smx_mutex_t)*mutex);
#endif
	ret = ret ? 0 : EBUSY;
	return ret;
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

static void _starpu_pthread_cond_auto_init(starpu_pthread_cond_t *cond)
{
	/* Note: this is actually safe, because simgrid only preempts within
	 * simgrid functions */
	if (!*cond) {
		/* Here we may get preempted */
		xbt_cond_t new_cond = xbt_cond_init();
		if (!*cond)
			*cond = new_cond;
		else
			/* Somebody already initialized it while we were
			 * calling xbt_cond_init, this one is now useless */
			xbt_cond_destroy(new_cond);
	}
}

int starpu_pthread_cond_signal(starpu_pthread_cond_t *cond)
{
	_starpu_pthread_cond_auto_init(cond);
	xbt_cond_signal(*cond);
	return 0;
}

int starpu_pthread_cond_broadcast(starpu_pthread_cond_t *cond)
{
	_starpu_pthread_cond_auto_init(cond);
	xbt_cond_broadcast(*cond);
	return 0;
}

int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	_starpu_pthread_cond_auto_init(cond);
	xbt_cond_wait(*cond, *mutex);
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
	return starpu_pthread_mutex_init(rwlock, NULL);
}

int starpu_pthread_rwlock_destroy(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_destroy(rwlock);
}

int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_lock(rwlock);
}

int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_trylock(rwlock);
}

int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_lock(rwlock);
}

int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_trylock(rwlock);
}

int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock)
{
	return starpu_pthread_mutex_unlock(rwlock);
}
#endif /* STARPU_SIMGRID */

#if defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_BARRIER)
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr, unsigned count)
{
	int ret = starpu_pthread_mutex_init(&barrier->mutex, NULL);
	if (!ret)
		ret = starpu_pthread_cond_init(&barrier->cond, NULL);
	if (!ret)
		ret = starpu_pthread_cond_init(&barrier->cond_destroy, NULL);
	barrier->count = count;
	barrier->done = 0;
	barrier->busy = 0;
	return ret;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	starpu_pthread_mutex_lock(&barrier->mutex);
	while (barrier->busy) {
		starpu_pthread_cond_wait(&barrier->cond_destroy, &barrier->mutex);
	}
	starpu_pthread_mutex_unlock(&barrier->mutex);
	int ret = starpu_pthread_mutex_destroy(&barrier->mutex);
	if (!ret)
		ret = starpu_pthread_cond_destroy(&barrier->cond);
	if (!ret)
		ret = starpu_pthread_cond_destroy(&barrier->cond_destroy);
	return ret;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret = 0;

	starpu_pthread_mutex_lock(&barrier->mutex);
	barrier->done++;
	if (barrier->done == barrier->count)
	{
		barrier->done = 0;
		starpu_pthread_cond_broadcast(&barrier->cond);
		ret = STARPU_PTHREAD_BARRIER_SERIAL_THREAD;
	} else {
		barrier->busy++;
		starpu_pthread_cond_wait(&barrier->cond, &barrier->mutex);
		barrier->busy--;
		starpu_pthread_cond_broadcast(&barrier->cond_destroy);
	}

	starpu_pthread_mutex_unlock(&barrier->mutex);


	return ret;
}

#endif /* defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_BARRIER) */
