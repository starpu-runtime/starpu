/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2014  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
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
#include <core/simgrid.h>
#include <core/workers.h>

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
	*thread = MSG_process_create(name, _starpu_simgrid_thread_start, _args,
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
#if 0 //def HAVE_MSG_PROCESS_JOIN
	MSG_process_join(thread, 100);
#else
	MSG_process_sleep(1);
#endif
	return 0;
}

int starpu_pthread_exit(void *retval)
{
	MSG_process_kill(MSG_process_self());
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

int starpu_pthread_mutexattr_gettype(const starpu_pthread_mutexattr_t *attr, int *type)
{
	return 0;
}

int starpu_pthread_mutexattr_settype(starpu_pthread_mutexattr_t *attr, int type)
{
	return 0;
}

int starpu_pthread_mutexattr_destroy(starpu_pthread_mutexattr_t *attr)
{
	return 0;
}

int starpu_pthread_mutexattr_init(starpu_pthread_mutexattr_t *attr)
{
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
	return starpu_pthread_mutex_init(rwlock, NULL);
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

int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock)
{
	int p_ret = starpu_pthread_mutex_trylock(rwlock);

	if (!p_ret)
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

int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock)
{
	int p_ret =  starpu_pthread_mutex_trylock(rwlock);

	if (!p_ret)
		_STARPU_TRACE_RWLOCK_RDLOCKED();

	return p_ret;
}


int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_UNLOCKING_RWLOCK();

 	int p_ret = starpu_pthread_mutex_unlock(rwlock);

	_STARPU_TRACE_RWLOCK_UNLOCKED();

	return p_ret;
}

#if defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT)
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr, unsigned count)
{
	*barrier = xbt_barrier_init(count);
	return 0;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	if (*barrier)
		xbt_barrier_destroy(*barrier);
	return 0;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	_STARPU_TRACE_BARRIER_WAIT_BEGIN();

	xbt_barrier_wait(*barrier);

	_STARPU_TRACE_BARRIER_WAIT_END();
	return 0;
}
#endif /* defined(STARPU_SIMGRID) */

#endif /* STARPU_SIMGRID */

#if (defined(STARPU_SIMGRID) && !defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT)) || !defined(STARPU_HAVE_PTHREAD_BARRIER)
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr, unsigned count)
{
	int ret = starpu_pthread_mutex_init(&barrier->mutex, NULL);
	if (!ret)
		ret = starpu_pthread_cond_init(&barrier->cond, NULL);
	barrier->count = count;
	barrier->done = 0;
	return ret;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	int ret = starpu_pthread_mutex_destroy(&barrier->mutex);
	if (!ret)
		ret = starpu_pthread_cond_destroy(&barrier->cond);
	return ret;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret = 0;
	_STARPU_TRACE_BARRIER_WAIT_BEGIN();

	starpu_pthread_mutex_lock(&barrier->mutex);
	barrier->done++;
	if (barrier->done == barrier->count)
	{
		barrier->done = 0;
		starpu_pthread_cond_broadcast(&barrier->cond);
		ret = STARPU_PTHREAD_BARRIER_SERIAL_THREAD;
	} else {
		starpu_pthread_cond_wait(&barrier->cond, &barrier->mutex);
	}

	starpu_pthread_mutex_unlock(&barrier->mutex);

	_STARPU_TRACE_BARRIER_WAIT_END();

	return ret;
}
#endif /* defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_BARRIER) */

#if !defined(STARPU_SIMGRID) && !defined(_MSC_VER) /* !STARPU_SIMGRID */
int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_LOCKING_MUTEX();

	int p_ret = pthread_mutex_lock(mutex);
	int workerid = starpu_worker_get_id();
	if(workerid != -1 && _starpu_worker_mutex_is_sched_mutex(workerid, mutex))
		_starpu_worker_set_flag_sched_mutex_locked(workerid, 1);

	_STARPU_TRACE_MUTEX_LOCKED();

	return p_ret;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_UNLOCKING_MUTEX();

	int p_ret = pthread_mutex_unlock(mutex);
	int workerid = starpu_worker_get_id();
	if(workerid != -1 && _starpu_worker_mutex_is_sched_mutex(workerid, mutex))
		_starpu_worker_set_flag_sched_mutex_locked(workerid, 0);

	_STARPU_TRACE_MUTEX_UNLOCKED();

	return p_ret;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	int ret;
	_STARPU_TRACE_TRYLOCK_MUTEX();

	ret = pthread_mutex_trylock(mutex);

	if (!ret)
	{
		int workerid = starpu_worker_get_id();
		if(workerid != -1 && _starpu_worker_mutex_is_sched_mutex(workerid, mutex))
			_starpu_worker_set_flag_sched_mutex_locked(workerid, 1);

		_STARPU_TRACE_MUTEX_LOCKED();
	}

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

int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_RDLOCKING_RWLOCK();

 	int p_ret = pthread_rwlock_tryrdlock(rwlock);

	if (!p_ret)
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

int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock)
{
	_STARPU_TRACE_WRLOCKING_RWLOCK();

 	int p_ret = pthread_rwlock_trywrlock(rwlock);

	if (!p_ret)
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
#endif

#if !defined(STARPU_SIMGRID) && !defined(_MSC_VER) && defined(STARPU_HAVE_PTHREAD_BARRIER)
int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret;
	_STARPU_TRACE_BARRIER_WAIT_BEGIN();

	ret = pthread_barrier_wait(barrier);

	_STARPU_TRACE_BARRIER_WAIT_END();

	return ret;
}
#endif /* STARPU_SIMGRID, _MSC_VER, STARPU_HAVE_PTHREAD_BARRIER */

#if defined(STARPU_SIMGRID) || !defined(HAVE_PTHREAD_SPIN_LOCK)

int starpu_pthread_spin_init(starpu_pthread_spinlock_t *lock, int pshared)
{
	lock->taken = 0;
	return 0;
}

int starpu_pthread_spin_destroy(starpu_pthread_spinlock_t *lock)
{
	/* we don't do anything */
	return 0;
}

int starpu_pthread_spin_lock(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	while (1)
	{
		if (!lock->taken)
		{
			lock->taken = 1;
			return 0;
		}
		/* Give hand to another thread, hopefully the one which has the
		 * spinlock and probably just has also a short-lived mutex. */
		MSG_process_sleep(0.000001);
		STARPU_UYIELD();
	}
#else
	uint32_t prev;
	do
	{
		prev = STARPU_TEST_AND_SET(&lock->taken, 1);
		if (prev)
			STARPU_UYIELD();
	}
	while (prev);
	return 0;
#endif
}

int starpu_pthread_spin_trylock(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	if (lock->taken)
		return EBUSY;
	lock->taken = 1;
	return 0;
#else
	uint32_t prev;
	prev = STARPU_TEST_AND_SET(&lock->taken, 1);
	return (prev == 0)?0:EBUSY;
#endif
}

int starpu_pthread_spin_unlock(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	lock->taken = 0;
	return 0;
#else
	STARPU_RELEASE(&lock->taken);
	return 0;
#endif
}

#endif /* defined(STARPU_SIMGRID) || !defined(HAVE_PTHREAD_SPIN_LOCK) */

int _starpu_pthread_spin_checklocked(starpu_pthread_spinlock_t *lock)
{
#ifdef STARPU_SIMGRID
	STARPU_ASSERT(lock->taken);
	return !lock->taken;
#elif defined(HAVE_PTHREAD_SPIN_LOCK)
	int ret = pthread_spin_trylock((pthread_spinlock_t *)lock);
	STARPU_ASSERT(ret != 0);
	return ret == 0;
#else
	STARPU_ASSERT(lock->taken);
	return !lock->taken;
#endif
}

