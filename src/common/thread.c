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

#include <starpu.h>
#include <core/simgrid.h>
#ifdef STARPU_DEBUG
#include <core/workers.h>
#endif
#include <common/thread.h>
#include <common/fxt.h>
#include <common/timing.h>

#include <errno.h>
#include <limits.h>

#ifdef STARPU_SIMGRID
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
#include <simgrid/mutex.h>
#include <simgrid/cond.h>
#elif defined(STARPU_HAVE_XBT_SYNCHRO_H)
#include <xbt/synchro.h>
#else
#include <xbt/synchro_core.h>
#endif
#include <smpi/smpi.h>
#include <simgrid/simix.h>
#else

#if defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
#include <linux/futex.h>
#include <sys/syscall.h>

/* Private futexes are not so old, cope with old kernels.  */
#ifdef FUTEX_WAIT_PRIVATE
static int _starpu_futex_wait = FUTEX_WAIT_PRIVATE;
static int _starpu_futex_wake = FUTEX_WAKE_PRIVATE;
#else
static int _starpu_futex_wait = FUTEX_WAIT;
static int _starpu_futex_wake = FUTEX_WAKE;
#endif

#endif
#endif /* !STARPU_SIMGRID */

#ifdef STARPU_SIMGRID

extern int _starpu_simgrid_thread_start(int argc, char *argv[]);

int starpu_pthread_equal(starpu_pthread_t t1, starpu_pthread_t t2)
{
	return t1 == t2;
}

starpu_pthread_t starpu_pthread_self(void)
{
#ifdef HAVE_SG_ACTOR_SELF
	return sg_actor_self();
#else
	return MSG_process_self();
#endif
}

int starpu_pthread_create_on(const char *name, starpu_pthread_t *thread, const starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED, void *(*start_routine) (void *), void *arg, starpu_sg_host_t host)
{
	char **_args;
	_STARPU_MALLOC(_args, 3*sizeof(char*));
	asprintf(&_args[0], "%p", start_routine);
	asprintf(&_args[1], "%p", arg);
	_args[2] = NULL;
	if (!host)
		host = _starpu_simgrid_get_host_by_name("MAIN");

	void *tsd;
	_STARPU_CALLOC(tsd, MAX_TSD+1, sizeof(void*));

#ifdef HAVE_SG_ACTOR_INIT
	*thread= sg_actor_init(name, host);
	sg_actor_data_set(*thread, tsd);
	sg_actor_start(*thread, _starpu_simgrid_thread_start, 2, _args);
#else
	*thread = MSG_process_create_with_arguments(name, _starpu_simgrid_thread_start, tsd, host, 2, _args);
#ifdef HAVE_SG_ACTOR_DATA
	sg_actor_data_set(*thread, tsd);
#endif
#endif

#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
#  ifdef HAVE_SG_ACTOR_REF
	sg_actor_ref(*thread);
#  else
	MSG_process_ref(*thread);
#  endif
#endif
	return 0;
}

int starpu_pthread_create(starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)
{
	return starpu_pthread_create_on("", thread, attr, start_routine, arg, NULL);
}

int starpu_pthread_join(starpu_pthread_t thread STARPU_ATTRIBUTE_UNUSED, void **retval STARPU_ATTRIBUTE_UNUSED)
{
#if SIMGRID_VERSION >= 31400
#  ifdef STARPU_HAVE_SIMGRID_ACTOR_H
	sg_actor_join(thread, 1000000);
#  else
	MSG_process_join(thread, 1000000);
#  endif
#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
#  ifdef HAVE_SG_ACTOR_REF
	sg_actor_unref(thread);
#  else
	MSG_process_unref(thread);
#  endif
#endif
#else
	starpu_sleep(1);
#endif
	return 0;
}

int starpu_pthread_exit(void *retval STARPU_ATTRIBUTE_UNUSED)
{
#ifdef HAVE_SG_ACTOR_SELF
	sg_actor_kill(sg_actor_self());
#else
	MSG_process_kill(MSG_process_self());
#endif
	STARPU_ABORT_MSG("MSG_process_kill(MSG_process_self()) returned?!");
}


int starpu_pthread_attr_init(starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_attr_destroy(starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_attr_setdetachstate(starpu_pthread_attr_t *attr STARPU_ATTRIBUTE_UNUSED, int detachstate STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutex_init(starpu_pthread_mutex_t *mutex, const starpu_pthread_mutexattr_t *mutexattr STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	*mutex = sg_mutex_init();
#else
	*mutex = xbt_mutex_init();
#endif
	return 0;
}

int starpu_pthread_mutex_destroy(starpu_pthread_mutex_t *mutex)
{
	if (*mutex)
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
		sg_mutex_destroy(*mutex);
#else
		xbt_mutex_destroy(*mutex);
#endif
	return 0;
}

int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_LOCKING_MUTEX();

	/* Note: this is actually safe, because simgrid only preempts within
	 * simgrid functions */
	if (!*mutex)
	{
		/* Here we may get preempted */
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
		sg_mutex_t new_mutex = sg_mutex_init();
#else
		xbt_mutex_t new_mutex = xbt_mutex_init();
#endif
		if (!*mutex)
			*mutex = new_mutex;
		else
			/* Somebody already initialized it while we were
			 * calling sg_mutex_init, this one is now useless */
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
			sg_mutex_destroy(new_mutex);
#else
			xbt_mutex_destroy(new_mutex);
#endif
	}

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	sg_mutex_lock(*mutex);
#else
	xbt_mutex_acquire(*mutex);
#endif

	_STARPU_TRACE_MUTEX_LOCKED();

	return 0;
}

int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_UNLOCKING_MUTEX();

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	sg_mutex_unlock(*mutex);
#else
	xbt_mutex_release(*mutex);
#endif

	_STARPU_TRACE_MUTEX_UNLOCKED();

	return 0;
}

int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex)
{
	int ret;
	_STARPU_TRACE_TRYLOCK_MUTEX();

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
	ret = sg_mutex_try_lock(*mutex);
#elif defined(HAVE_XBT_MUTEX_TRY_ACQUIRE) || defined(xbt_mutex_try_acquire)
	ret = xbt_mutex_try_acquire(*mutex);
#else
	ret = simcall_mutex_trylock((smx_mutex_t)*mutex);
#endif
	ret = ret ? 0 : EBUSY;

	_STARPU_TRACE_MUTEX_LOCKED();

	return ret;
}

int starpu_pthread_mutexattr_gettype(const starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED, int *type STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutexattr_settype(starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED, int type STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutexattr_destroy(starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

int starpu_pthread_mutexattr_init(starpu_pthread_mutexattr_t *attr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}


/* Indexed by key-1 */
static int used_key[MAX_TSD];

int starpu_pthread_key_create(starpu_pthread_key_t *key, void (*destr_function) (void *) STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i;

	/* Note: no synchronization here, we are actually monothreaded anyway. */
	for (i = 0; i < MAX_TSD; i++)
	{
		if (!used_key[i])
		{
			used_key[i] = 1;
			break;
		}
	}
	STARPU_ASSERT(i < MAX_TSD);
	/* key 0 is for process pointer argument */
	*key = i+1;
	return 0;
}

int starpu_pthread_key_delete(starpu_pthread_key_t key)
{
	used_key[key-1] = 0;
	return 0;
}

/* We need it only when using smpi */
#pragma weak smpi_process_get_user_data
#if !HAVE_DECL_SMPI_PROCESS_SET_USER_DATA && !defined(smpi_process_get_user_data)
extern void *smpi_process_get_user_data();
#endif

int starpu_pthread_setspecific(starpu_pthread_key_t key, const void *pointer)
{
	void **array;
#ifdef HAVE_SG_ACTOR_DATA
	array = sg_actor_data(sg_actor_self());
#else
#if defined(HAVE_SMPI_PROCESS_SET_USER_DATA) || defined(smpi_process_get_user_data)
#if defined(HAVE_MSG_PROCESS_SELF_NAME) || defined(MSG_process_self_name)
	const char *process_name = MSG_process_self_name();
#else
	const char *process_name = SIMIX_process_self_get_name();
#endif
	char *end;
	/* Test whether it is an MPI rank */
	strtol(process_name, &end, 10);
	if (!*end || !strcmp(process_name, "wait for mpi transfer") ||
			(!strcmp(process_name, "main") && _starpu_simgrid_running_smpi()))
		/* Special-case the SMPI process */
		array = smpi_process_get_user_data();
	else
#endif
		array = MSG_process_get_data(MSG_process_self());
#endif
	array[key] = (void*) pointer;
	return 0;
}

void* starpu_pthread_getspecific(starpu_pthread_key_t key)
{
	void **array;
#ifdef HAVE_SG_ACTOR_DATA
	array = sg_actor_data(sg_actor_self());
#else
#if defined(HAVE_SMPI_PROCESS_SET_USER_DATA) || defined(smpi_process_get_user_data)
#if defined(HAVE_MSG_PROCESS_SELF_NAME) || defined(MSG_process_self_name)
	const char *process_name = MSG_process_self_name();
#else
	const char *process_name = SIMIX_process_self_get_name();
#endif
	char *end;
	/* Test whether it is an MPI rank */
	strtol(process_name, &end, 10);
	if (!*end || !strcmp(process_name, "wait for mpi transfer") ||
			(!strcmp(process_name, "main") && _starpu_simgrid_running_smpi()))
		/* Special-case the SMPI processes */
		array = smpi_process_get_user_data();
	else
#endif
		array = MSG_process_get_data(MSG_process_self());
#endif
	if (!array)
		return NULL;
	return array[key];
}

int starpu_pthread_cond_init(starpu_pthread_cond_t *cond, starpu_pthread_condattr_t *cond_attr STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_HAVE_SIMGRID_COND_H
	*cond = sg_cond_init();
#else
	*cond = xbt_cond_init();
#endif
	return 0;
}

static void _starpu_pthread_cond_auto_init(starpu_pthread_cond_t *cond)
{
	/* Note: this is actually safe, because simgrid only preempts within
	 * simgrid functions */
	if (!*cond)
	{
		/* Here we may get preempted */
#ifdef STARPU_HAVE_SIMGRID_COND_H
		sg_cond_t new_cond = sg_cond_init();
#else
		xbt_cond_t new_cond = xbt_cond_init();
#endif
		if (!*cond)
			*cond = new_cond;
		else
			/* Somebody already initialized it while we were
			 * calling xbt_cond_init, this one is now useless */
#ifdef STARPU_HAVE_SIMGRID_COND_H
			sg_cond_destroy(new_cond);
#else
			xbt_cond_destroy(new_cond);
#endif
	}
}

int starpu_pthread_cond_signal(starpu_pthread_cond_t *cond)
{
	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	sg_cond_notify_one(*cond);
#else
	xbt_cond_signal(*cond);
#endif
	return 0;
}

int starpu_pthread_cond_broadcast(starpu_pthread_cond_t *cond)
{
	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	sg_cond_notify_all(*cond);
#else
	xbt_cond_broadcast(*cond);
#endif
	return 0;
}

int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex)
{
	_STARPU_TRACE_COND_WAIT_BEGIN();

	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	sg_cond_wait(*cond, *mutex);
#else
	xbt_cond_wait(*cond, *mutex);
#endif

	_STARPU_TRACE_COND_WAIT_END();

	return 0;
}

int starpu_pthread_cond_timedwait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, const struct timespec *abstime)
{
#if SIMGRID_VERSION >= 31800
	struct timespec now, delta;
	double delay;
	int ret = 0;
	_starpu_clock_gettime(&now);
	delta.tv_sec = abstime->tv_sec - now.tv_sec;
	delta.tv_nsec = abstime->tv_nsec - now.tv_nsec;
	delay = (double) delta.tv_sec + (double) delta.tv_nsec / 1000000000.;

	_STARPU_TRACE_COND_WAIT_BEGIN();

	_starpu_pthread_cond_auto_init(cond);
#ifdef STARPU_HAVE_SIMGRID_COND_H
	ret = sg_cond_wait_for(*cond, *mutex, delay) ? ETIMEDOUT : 0;
#else
	ret = xbt_cond_timedwait(*cond, *mutex, delay) ? ETIMEDOUT : 0;
#endif

	_STARPU_TRACE_COND_WAIT_END();

	return ret;
#else
	STARPU_ASSERT_MSG(0, "simgrid version is too old for this");
#endif
}

int starpu_pthread_cond_destroy(starpu_pthread_cond_t *cond)
{
	if (*cond)
#ifdef STARPU_HAVE_SIMGRID_COND_H
		sg_cond_destroy(*cond);
#else
		xbt_cond_destroy(*cond);
#endif
	return 0;
}

/* TODO: use rwlocks
 * https://gforge.inria.fr/tracker/index.php?func=detail&aid=17213&group_id=12&atid=165
 */
int starpu_pthread_rwlock_init(starpu_pthread_rwlock_t *restrict rwlock, const starpu_pthread_rwlockattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED)
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

#ifdef STARPU_HAVE_SIMGRID_BARRIER_H
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED, unsigned count)
{
	*barrier = sg_barrier_init(count);
	return 0;
}

int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier)
{
	if (*barrier)
		sg_barrier_destroy(*barrier);
	return 0;
}

int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier)
{
	int ret;

	_STARPU_TRACE_BARRIER_WAIT_BEGIN();

	ret = sg_barrier_wait(*barrier);

	_STARPU_TRACE_BARRIER_WAIT_END();
	return ret;
}
#elif defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) || defined(xbt_barrier_init)
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED, unsigned count)
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
	int ret;

	_STARPU_TRACE_BARRIER_WAIT_BEGIN();

	ret = xbt_barrier_wait(*barrier);

	_STARPU_TRACE_BARRIER_WAIT_END();
	return ret;
}
#endif /* defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) */

int starpu_pthread_queue_init(starpu_pthread_queue_t *q)
{
	STARPU_PTHREAD_MUTEX_INIT(&q->mutex, NULL);
	q->queue = NULL;
	q->allocqueue = 0;
	q->nqueue = 0;
	return 0;
}

int starpu_pthread_wait_init(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_INIT(&w->mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&w->cond, NULL);
	w->block = 1;
	return 0;
}

int starpu_pthread_queue_register(starpu_pthread_wait_t *w, starpu_pthread_queue_t *q)
{
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);

	if (q->nqueue == q->allocqueue)
	{
		/* Make room for the new waiter */
		unsigned newalloc;
		newalloc = q->allocqueue * 2;
		if (!newalloc)
			newalloc = 1;
		_STARPU_REALLOC(q->queue, newalloc * sizeof(*(q->queue)));
		q->allocqueue = newalloc;
	}
	q->queue[q->nqueue++] = w;

	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_queue_unregister(starpu_pthread_wait_t *w, starpu_pthread_queue_t *q)
{
	unsigned i;
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	for (i = 0; i < q->nqueue; i++)
	{
		if (q->queue[i] == w)
		{
			memmove(&q->queue[i], &q->queue[i+1], (q->nqueue - i - 1) * sizeof(*(q->queue)));
			break;
		}
	}
	STARPU_ASSERT(i < q->nqueue);
	q->nqueue--;
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_wait_reset(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	w->block = 1;
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	return 0;
}

int starpu_pthread_wait_wait(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	while (w->block == 1)
		STARPU_PTHREAD_COND_WAIT(&w->cond, &w->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	return 0;
}

/* pthread_cond_timedwait not yet available on windows, but we don't run simgrid there anyway */
#ifdef STARPU_SIMGRID
int starpu_pthread_wait_timedwait(starpu_pthread_wait_t *w, const struct timespec *abstime)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	while (w->block == 1)
		STARPU_PTHREAD_COND_TIMEDWAIT(&w->cond, &w->mutex, abstime);
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	return 0;
}
#endif

int starpu_pthread_queue_signal(starpu_pthread_queue_t *q)
{
	starpu_pthread_wait_t *w;
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	if (q->nqueue)
	{
		/* TODO: better try to wake a sleeping one if possible */
		w = q->queue[0];
		STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
		w->block = 0;
		STARPU_PTHREAD_COND_SIGNAL(&w->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_queue_broadcast(starpu_pthread_queue_t *q)
{
	unsigned i;
	starpu_pthread_wait_t *w;
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	for (i = 0; i < q->nqueue; i++)
	{
		w = q->queue[i];
		STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
		w->block = 0;
		STARPU_PTHREAD_COND_SIGNAL(&w->cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	return 0;
}

int starpu_pthread_wait_destroy(starpu_pthread_wait_t *w)
{
	STARPU_PTHREAD_MUTEX_LOCK(&w->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&w->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&w->mutex);
	STARPU_PTHREAD_COND_DESTROY(&w->cond);
	return 0;
}

int starpu_pthread_queue_destroy(starpu_pthread_queue_t *q)
{
	STARPU_ASSERT(!q->nqueue);
	STARPU_PTHREAD_MUTEX_LOCK(&q->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&q->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&q->mutex);
	free(q->queue);
	return 0;
}

#endif /* STARPU_SIMGRID */

#if (defined(STARPU_SIMGRID) && !defined(STARPU_HAVE_SIMGRID_BARRIER_H) && !defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) && !defined(xbt_barrier_init)) || (!defined(STARPU_SIMGRID) && !defined(STARPU_HAVE_PTHREAD_BARRIER))
int starpu_pthread_barrier_init(starpu_pthread_barrier_t *restrict barrier, const starpu_pthread_barrierattr_t *restrict attr STARPU_ATTRIBUTE_UNUSED, unsigned count)
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
	while (barrier->busy)
	{
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
	_STARPU_TRACE_BARRIER_WAIT_BEGIN();

	starpu_pthread_mutex_lock(&barrier->mutex);
	barrier->done++;
	if (barrier->done == barrier->count)
	{
		barrier->done = 0;
		starpu_pthread_cond_broadcast(&barrier->cond);
		ret = STARPU_PTHREAD_BARRIER_SERIAL_THREAD;
	}
	else
	{
		barrier->busy++;
		starpu_pthread_cond_wait(&barrier->cond, &barrier->mutex);
		barrier->busy--;
		starpu_pthread_cond_broadcast(&barrier->cond_destroy);
	}

	starpu_pthread_mutex_unlock(&barrier->mutex);

	_STARPU_TRACE_BARRIER_WAIT_END();

	return ret;
}
#endif /* defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_BARRIER) */

#ifdef STARPU_FXT_LOCK_TRACES
#if !defined(STARPU_SIMGRID) && !defined(_MSC_VER) /* !STARPU_SIMGRID */
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
#endif /* !defined(STARPU_SIMGRID) && !defined(_MSC_VER) */

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

#endif /* STARPU_FXT_LOCK_TRACES */

/* "sched" variants, to be used (through the STARPU_PTHREAD_MUTEX_*LOCK_SCHED
 * macros of course) which record when the mutex is held or not */
int starpu_pthread_mutex_lock_sched(starpu_pthread_mutex_t *mutex)
{
	return starpu_pthread_mutex_lock(mutex);
}

int starpu_pthread_mutex_unlock_sched(starpu_pthread_mutex_t *mutex)
{
	return starpu_pthread_mutex_unlock(mutex);
}

int starpu_pthread_mutex_trylock_sched(starpu_pthread_mutex_t *mutex)
{
	return starpu_pthread_mutex_trylock(mutex);
}

#ifdef STARPU_DEBUG
void starpu_pthread_mutex_check_sched(starpu_pthread_mutex_t *mutex, char *file, int line)
{
	int workerid = starpu_worker_get_id();
	STARPU_ASSERT_MSG(workerid == -1 || !_starpu_worker_mutex_is_sched_mutex(workerid, mutex), "%s:%d is locking/unlocking a sched mutex but not using STARPU_PTHREAD_MUTEX_LOCK_SCHED", file, line);
}
#endif

#if defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(HAVE_PTHREAD_SPIN_LOCK)

#undef starpu_pthread_spin_init
int starpu_pthread_spin_init(starpu_pthread_spinlock_t *lock, int pshared)
{
	return _starpu_pthread_spin_init(lock, pshared);
}

#undef starpu_pthread_spin_destroy
int starpu_pthread_spin_destroy(starpu_pthread_spinlock_t *lock STARPU_ATTRIBUTE_UNUSED)
{
	return _starpu_pthread_spin_destroy(lock);
}

#undef starpu_pthread_spin_lock
int starpu_pthread_spin_lock(starpu_pthread_spinlock_t *lock)
{
	return _starpu_pthread_spin_lock(lock);
}
#endif

#if defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK)

#if !defined(STARPU_SIMGRID) && defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
int _starpu_pthread_spin_do_lock(starpu_pthread_spinlock_t *lock)
{
	if (STARPU_VAL_COMPARE_AND_SWAP(&lock->taken, 0, 1) == 0)
		/* Got it on first try! */
		return 0;

	/* Busy, spin a bit.  */
	unsigned i;
	for (i = 0; i < 128; i++)
	{
		/* Pause a bit before retrying */
		STARPU_UYIELD();
		/* And synchronize with other threads */
		STARPU_SYNCHRONIZE();
		if (!lock->taken)
			/* Holder released it, try again */
			if (STARPU_VAL_COMPARE_AND_SWAP(&lock->taken, 0, 1) == 0)
				/* Got it! */
				return 0;
	}

	/* We have spent enough time with spinning, let's block */
	/* This avoids typical 10ms pauses when the application thread tries to submit tasks. */
	while (1)
	{
		/* Tell releaser to wake us */
		unsigned prev = STARPU_VAL_EXCHANGE(&lock->taken, 2);
		if (prev == 0)
			/* Ah, it just got released and we actually acquired
			 * it!
			 * Note: the sad thing is that we have just written 2,
			 * so will spuriously try to wake a thread on unlock,
			 * but we can not avoid it since we do not know whether
			 * there are other threads sleeping or not.
			 */
			return 0;

		/* Now start sleeping (unless it was released in between)
		 * We are sure to get woken because either
		 * - some thread has not released the lock yet, and lock->taken
		 *   is 2, so it will wake us.
		 * - some other thread started blocking, and will set
		 *   lock->taken back to 2
		 */
		if (syscall(SYS_futex, &lock->taken, _starpu_futex_wait, 2, NULL, NULL, 0))
			if (errno == ENOSYS)
				_starpu_futex_wait = FUTEX_WAIT;
	}
}
#endif

#undef starpu_pthread_spin_trylock
int starpu_pthread_spin_trylock(starpu_pthread_spinlock_t *lock)
{
	return _starpu_pthread_spin_trylock(lock);
}

#undef starpu_pthread_spin_unlock
int starpu_pthread_spin_unlock(starpu_pthread_spinlock_t *lock)
{
	return _starpu_pthread_spin_unlock(lock);
}

#if !defined(STARPU_SIMGRID) && defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
void _starpu_pthread_spin_do_unlock(starpu_pthread_spinlock_t *lock)
{
	/*
	 * Somebody to wake. Clear 'taken' and wake him.
	 * Note that he may not be sleeping yet, but if he is not, we won't
	 * since the value of 'taken' will have changed.
	 */
	lock->taken = 0;
	STARPU_SYNCHRONIZE();
	if (syscall(SYS_futex, &lock->taken, _starpu_futex_wake, 1, NULL, NULL, 0) == -1)
		switch (errno)
		{
			case ENOSYS:
				_starpu_futex_wake = FUTEX_WAKE;
				if (syscall(SYS_futex, &lock->taken, _starpu_futex_wake, 1, NULL, NULL, 0) == -1)
					STARPU_ASSERT_MSG(0, "futex(wake) returned %d!", errno);
				break;
			case 0:
				break;
			default:
				STARPU_ASSERT_MSG(0, "futex returned %d!", errno);
				break;
		}
}

#endif

#endif /* defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK) */

#ifdef STARPU_SIMGRID

int starpu_sem_destroy(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	sg_sem_destroy(*sem);
#else
	MSG_sem_destroy(*sem);
#endif
	return 0;
}

int starpu_sem_init(starpu_sem_t *sem, int pshared, unsigned value)
{
	STARPU_ASSERT_MSG(pshared == 0, "pshared semaphores not supported under simgrid");
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	*sem = sg_sem_init(value);
#else
	*sem = MSG_sem_init(value);
#endif
	return 0;
}

int starpu_sem_post(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	sg_sem_release(*sem);
#else
	MSG_sem_release(*sem);
#endif
	return 0;
}

int starpu_sem_wait(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	sg_sem_acquire(*sem);
#else
	MSG_sem_acquire(*sem);
#endif
	return 0;
}

int starpu_sem_trywait(starpu_sem_t *sem)
{
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	if (sg_sem_would_block(*sem))
#else
	if (MSG_sem_would_block(*sem))
#endif
		return EAGAIN;
	starpu_sem_wait(sem);
	return 0;
}

int starpu_sem_getvalue(starpu_sem_t *sem, int *sval)
{
#if SIMGRID_VERSION > 31300
#  ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
	*sval = sg_sem_get_capacity(*sem);
#  else
	*sval = MSG_sem_get_capacity(*sem);
#  endif
	return 0;
#else
	(void) sem;
	(void) sval;
	STARPU_ABORT_MSG("sigmrid up to 3.13 did not have working MSG_sem_get_capacity");
#endif
}

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

int starpu_sem_wait(starpu_sem_t *sem)
{
	int ret;
	while((ret = sem_wait(sem)) == -1 && errno == EINTR)
		;

	return ret;
}

int starpu_sem_trywait(starpu_sem_t *sem)
{
	int ret;
	while((ret = sem_trywait(sem)) == -1 && errno == EINTR)
		;
	
	return ret;
}

#endif
