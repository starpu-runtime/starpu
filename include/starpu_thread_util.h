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

// The documentation for this file is in doc/doxygen/chapters/api/threads.doxy

#ifndef __STARPU_THREAD_UTIL_H__
#define __STARPU_THREAD_UTIL_H__

#include <starpu_util.h>
#include <starpu_thread.h>
#include <errno.h>

#if !(defined(_MSC_VER) && !defined(BUILDING_STARPU))
/*
 * Encapsulation of the starpu_pthread_create_* functions.
 */

#define STARPU_PTHREAD_CREATE_ON(name, thread, attr, routine, arg, where) do {		    		\
	int p_ret =  starpu_pthread_create_on((name), (thread), (attr), (routine), (arg), (where)); 	\
	if (STARPU_UNLIKELY(p_ret != 0)) {								\
		fprintf(stderr,										\
			"%s:%d starpu_pthread_create_on: %s\n",						\
			__FILE__, __LINE__, strerror(p_ret));						\
		STARPU_ABORT();										\
	}												\
} while (0)

#define STARPU_PTHREAD_CREATE(thread, attr, routine, arg) do {		    	\
	int p_ret =  starpu_pthread_create((thread), (attr), (routine), (arg)); \
	if (STARPU_UNLIKELY(p_ret != 0)) {					\
		fprintf(stderr,							\
			"%s:%d starpu_pthread_create: %s\n",			\
			__FILE__, __LINE__, strerror(p_ret));			\
		STARPU_ABORT();							\
	}									\
} while (0)

#define STARPU_PTHREAD_JOIN(thread, retval) do {		    	\
	int p_ret =  starpu_pthread_join((thread), (retval)); \
	if (STARPU_UNLIKELY(p_ret != 0)) {					\
		fprintf(stderr,							\
			"%s:%d starpu_pthread_join: %s\n",			\
			__FILE__, __LINE__, strerror(p_ret));			\
		STARPU_ABORT();							\
	}									\
} while (0)

/*
 * Encapsulation of the starpu_pthread_mutex_* functions.
 */

#define STARPU_PTHREAD_MUTEX_INIT(mutex, attr) do {                           \
	int p_ret = starpu_pthread_mutex_init((mutex), (attr));                \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_init: %s\n",               \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_MUTEX_DESTROY(mutex) do {                              \
	int p_ret = starpu_pthread_mutex_destroy(mutex);                       \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_destroy: %s\n",            \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while(0)

#ifdef STARPU_DEBUG
#define _STARPU_CHECK_NOT_SCHED_MUTEX(mutex, file, line) \
	starpu_pthread_mutex_check_sched((mutex), file, line)
#else
#define _STARPU_CHECK_NOT_SCHED_MUTEX(mutex, file, line)
#endif

#define STARPU_PTHREAD_MUTEX_LOCK(mutex) do {				      \
	int p_ret = starpu_pthread_mutex_lock(mutex);			      \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_lock: %s\n",               \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
	_STARPU_CHECK_NOT_SCHED_MUTEX(mutex, __FILE__, __LINE__);                                  \
} while (0)

#define STARPU_PTHREAD_MUTEX_LOCK_SCHED(mutex) do {			      \
	int p_ret = starpu_pthread_mutex_lock_sched(mutex);		      \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_lock_sched: %s\n",         \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_MUTEX_TRYLOCK(mutex) \
	_starpu_pthread_mutex_trylock(mutex, __FILE__, __LINE__)
static STARPU_INLINE
int _starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex, char *file, int line)
{
	int p_ret = starpu_pthread_mutex_trylock(mutex);
	if (STARPU_UNLIKELY(p_ret != 0 && p_ret != EBUSY)) {
		fprintf(stderr,
			"%s:%d starpu_pthread_mutex_trylock: %s\n",
			file, line, strerror(p_ret));
		STARPU_ABORT();
	}
	_STARPU_CHECK_NOT_SCHED_MUTEX(mutex, file, line);
	return p_ret;
}

#define STARPU_PTHREAD_MUTEX_TRYLOCK_SCHED(mutex) \
	_starpu_pthread_mutex_trylock_sched(mutex, __FILE__, __LINE__)
static STARPU_INLINE
int _starpu_pthread_mutex_trylock_sched(starpu_pthread_mutex_t *mutex, char *file, int line)
{
	int p_ret = starpu_pthread_mutex_trylock_sched(mutex);
	if (STARPU_UNLIKELY(p_ret != 0 && p_ret != EBUSY)) {
		fprintf(stderr,
			"%s:%d starpu_pthread_mutex_trylock_sched: %s\n",
			file, line, strerror(p_ret));
		STARPU_ABORT();
	}
	return p_ret;
}

#define STARPU_PTHREAD_MUTEX_UNLOCK(mutex) do {                               \
	_STARPU_CHECK_NOT_SCHED_MUTEX(mutex, __FILE__, __LINE__);              \
	int p_ret = starpu_pthread_mutex_unlock(mutex);                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_unlock: %s\n",             \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(mutex) do {                          \
	int p_ret = starpu_pthread_mutex_unlock_sched(mutex);                  \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_unlock_sched: %s\n",       \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

/*
 * Encapsulation of the starpu_pthread_key_* functions.
 */
#define STARPU_PTHREAD_KEY_CREATE(key, destr) do {                            \
	int p_ret = starpu_pthread_key_create((key), (destr));	               \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_key_create: %s\n",               \
			__FILE__, __LINE__, strerror(p_ret));                  \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_KEY_DELETE(key) do {                                   \
	int p_ret = starpu_pthread_key_delete((key));	                       \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_key_delete: %s\n",               \
			__FILE__, __LINE__, strerror(p_ret));                  \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_SETSPECIFIC(key, ptr) do {                             \
	int p_ret = starpu_pthread_setspecific((key), (ptr));	               \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_setspecific: %s\n",              \
			__FILE__, __LINE__, strerror(p_ret));                  \
	};                                                                     \
} while (0)

#define STARPU_PTHREAD_GETSPECIFIC(key) starpu_pthread_getspecific((key))

/*
 * Encapsulation of the starpu_pthread_rwlock_* functions.
 */
#define STARPU_PTHREAD_RWLOCK_INIT(rwlock, attr) do {                          \
	int p_ret = starpu_pthread_rwlock_init((rwlock), (attr));              \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_rwlock_init: %s\n",              \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_RWLOCK_RDLOCK(rwlock) do {                              \
	int p_ret = starpu_pthread_rwlock_rdlock(rwlock);                      \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_rwlock_rdlock: %s\n",            \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_RWLOCK_TRYRDLOCK(rwlock) \
	_starpu_pthread_rwlock_tryrdlock(rwlock, __FILE__, __LINE__)
static STARPU_INLINE
int _starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock, char *file, int line)
{
	int p_ret = starpu_pthread_rwlock_tryrdlock(rwlock);
	if (STARPU_UNLIKELY(p_ret != 0 && p_ret != EBUSY)) {
		fprintf(stderr,
			"%s:%d starpu_pthread_rwlock_tryrdlock: %s\n",
			file, line, strerror(p_ret));
		STARPU_ABORT();
	}
	return p_ret;
}

#define STARPU_PTHREAD_RWLOCK_WRLOCK(rwlock) do {                              \
	int p_ret = starpu_pthread_rwlock_wrlock(rwlock);                      \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_rwlock_wrlock: %s\n",            \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_RWLOCK_TRYWRLOCK(rwlock) \
	_starpu_pthread_rwlock_trywrlock(rwlock, __FILE__, __LINE__)
static STARPU_INLINE
int _starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock, char *file, int line)
{
	int p_ret = starpu_pthread_rwlock_trywrlock(rwlock);
	if (STARPU_UNLIKELY(p_ret != 0 && p_ret != EBUSY)) {
		fprintf(stderr,
			"%s:%d starpu_pthread_rwlock_trywrlock: %s\n",
			file, line, strerror(p_ret));
		STARPU_ABORT();
	}
	return p_ret;
}

#define STARPU_PTHREAD_RWLOCK_UNLOCK(rwlock) do {                              \
	int p_ret = starpu_pthread_rwlock_unlock(rwlock);                      \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_rwlock_unlock: %s\n",            \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_RWLOCK_DESTROY(rwlock) do {                            \
	int p_ret = starpu_pthread_rwlock_destroy(rwlock);                     \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_rwlock_destroy: %s\n",           \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

/*
 * Encapsulation of the starpu_pthread_cond_* functions.
 */
#define STARPU_PTHREAD_COND_INIT(cond, attr) do {                             \
	int p_ret = starpu_pthread_cond_init((cond), (attr));                  \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_cond_init: %s\n",                \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_COND_DESTROY(cond) do {                                \
	int p_ret = starpu_pthread_cond_destroy(cond);                         \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_cond_destroy: %s\n",             \
			__FILE__, __LINE__, strerror(p_ret));                  \
			STARPU_ABORT();                                        \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_COND_SIGNAL(cond) do {                                 \
	int p_ret = starpu_pthread_cond_signal(cond);                          \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_cond_signal: %s\n",              \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_COND_BROADCAST(cond) do {                              \
	int p_ret = starpu_pthread_cond_broadcast(cond);                       \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_cond_broadcast: %s\n",           \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_COND_WAIT(cond, mutex) do {                            \
	int p_ret = starpu_pthread_cond_wait((cond), (mutex));                 \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_cond_wait: %s\n",                \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

/* pthread_cond_timedwait not yet available on windows, but we don't run simgrid there anyway */
#ifdef STARPU_SIMGRID
#define STARPU_PTHREAD_COND_TIMEDWAIT(cond, mutex, abstime) \
	_starpu_pthread_cond_timedwait(cond, mutex, abstime, __FILE__, __LINE__)
static STARPU_INLINE
int _starpu_pthread_cond_timedwait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, const struct timespec *abstime, char *file, int line)
{
	int p_ret = starpu_pthread_cond_timedwait(cond, mutex, abstime);
	if (STARPU_UNLIKELY(p_ret != 0 && p_ret != ETIMEDOUT)) {
		fprintf(stderr,
			"%s:%d starpu_pthread_cond_timedwait: %s\n",
			file, line, strerror(p_ret));
		STARPU_ABORT();
	}
	return p_ret;
}
#endif

/*
 * Encapsulation of the starpu_pthread_barrier_* functions.
 */

#define STARPU_PTHREAD_BARRIER_INIT(barrier, attr, count) do {                \
	int p_ret = starpu_pthread_barrier_init((barrier), (attr), (count));          \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_barrier_init: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_BARRIER_DESTROY(barrier) do {                          \
	int p_ret = starpu_pthread_barrier_destroy((barrier));                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_barrier_destroy: %s\n",                 \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_BARRIER_WAIT(barrier) do {                             	\
	int p_ret = starpu_pthread_barrier_wait((barrier));				\
	if (STARPU_UNLIKELY(!((p_ret == 0) || (p_ret == STARPU_PTHREAD_BARRIER_SERIAL_THREAD)))) { \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_barrier_wait: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
			STARPU_ABORT();                                        \
	}                                                                      \
} while (0)
#endif /* _MSC_VER */

#endif /* __STARPU_THREAD_UTIL_H__ */
