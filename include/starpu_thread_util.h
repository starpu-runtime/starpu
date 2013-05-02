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

#ifndef __STARPU_THREAD_UTIL_H__
#define __STARPU_THREAD_UTIL_H__

#include <starpu.h>

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

#define STARPU_PTHREAD_MUTEX_LOCK(mutex) do {                                 \
	int p_ret = starpu_pthread_mutex_lock(mutex);                          \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_lock: %s\n",               \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_MUTEX_UNLOCK(mutex) do {                               \
	int p_ret = starpu_pthread_mutex_unlock(mutex);                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_mutex_unlock: %s\n",             \
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

#define STARPU_PTHREAD_RWLOCK_WRLOCK(rwlock) do {                              \
	int p_ret = starpu_pthread_rwlock_wrlock(rwlock);                      \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d starpu_pthread_rwlock_wrlock: %s\n",            \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

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

/*
 * Encapsulation of the starpu_pthread_barrier_* functions.
 */

#define STARPU_PTHREAD_BARRIER_INIT(barrier, attr, count) do {                \
	int p_ret = pthread_barrier_init((barrier), (attr), (count));          \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_barrier_init: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_BARRIER_DESTROY(barrier) do {                          \
	int p_ret = pthread_barrier_destroy((barrier));                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_barrier_destroy: %s\n",                 \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define STARPU_PTHREAD_BARRIER_WAIT(barrier) do {                             \
	int p_ret = pthread_barrier_wait(barrier);                             \
	if (STARPU_UNLIKELY(!((p_ret == 0) || (p_ret == PTHREAD_BARRIER_SERIAL_THREAD)))) { \
		fprintf(stderr,                                                \
			"%s:%d pthread_barrier_wait: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
			STARPU_ABORT();                                        \
	}                                                                      \
} while (0)

#endif /* __STARPU_THREAD_UTIL_H__ */
