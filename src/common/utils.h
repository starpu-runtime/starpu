/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <starpu.h>
#include <common/config.h>
#include <sys/stat.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>

#ifdef STARPU_VERBOSE
#  define _STARPU_DEBUG(fmt, args ...) do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%s] " fmt ,__func__ ,##args); fflush(stderr); }} while(0)
#else
#  define _STARPU_DEBUG(fmt, args ...) do { } while (0)
#endif

#ifdef STARPU_VERBOSE0
#  define _STARPU_LOG_IN()             do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%ld][%s] -->\n", pthread_self(), __func__ ); }} while(0)
#  define _STARPU_LOG_OUT()            do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%ld][%s] <--\n", pthread_self(), __func__ ); }} while(0)
#  define _STARPU_LOG_OUT_TAG(outtag)  do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%ld][%s] <-- (%s)\n", pthread_self(), __func__, outtag); }} while(0)
#else
#  define _STARPU_LOG_IN()
#  define _STARPU_LOG_OUT()
#  define _STARPU_LOG_OUT_TAG(outtag)
#endif

#define _STARPU_DISP(fmt, args ...) do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%s] " fmt ,__func__ ,##args); }} while(0)
#define _STARPU_ERROR(fmt, args ...)                                                  \
	do {                                                                          \
                fprintf(stderr, "[starpu][%s] Error: " fmt ,__func__ ,##args);        \
		STARPU_ABORT();                                                            \
	} while (0)


#define _STARPU_IS_ZERO(a) (fpclassify(a) == FP_ZERO)

typedef pthread_mutex_t _starpu_pthread_mutex_t;
int _starpu_mkpath(const char *s, mode_t mode);
void _starpu_mkpath_and_check(const char *s, mode_t mode);
int _starpu_check_mutex_deadlock(_starpu_pthread_mutex_t *mutex);
char *_starpu_get_home_path(void);

/* If FILE is currently on a comment line, eat it.  */
void _starpu_drop_comments(FILE *f);

struct _starpu_job;
/* Returns the symbol associated to that job if any. */
const char *_starpu_job_get_model_name(struct _starpu_job *j);

struct starpu_codelet;
/* Returns the symbol associated to that job if any. */
const char *_starpu_codelet_get_model_name(struct starpu_codelet *cl);

#define _STARPU_PTHREAD_CREATE(thread, attr, routine, arg) do {                \
	int p_ret = pthread_create((thread), (attr), (routine), (arg));	       \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d pthread_create: %s\n",                          \
			__FILE__, __LINE__, strerror(p_ret));                  \
	}                                                                      \
} while (0)

/*
 * Encapsulation of the pthread_key_* functions.
 */
typedef pthread_key_t _starpu_pthread_key_t;
#define _STARPU_PTHREAD_KEY_CREATE(key, destr) do {                            \
	int p_ret = pthread_key_create((key), (destr));	                       \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d pthread_key_create: %s\n",                      \
			__FILE__, __LINE__, strerror(p_ret));                  \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_KEY_DELETE(key) do {                                   \
	int p_ret = pthread_key_delete((key));	                               \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d pthread_key_delete: %s\n",                      \
			__FILE__, __LINE__, strerror(p_ret));                  \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_SETSPECIFIC(key, ptr) do {                             \
	int p_ret = pthread_setspecific((key), (ptr));	                       \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d pthread_setspecific: %s\n",                     \
			__FILE__, __LINE__, strerror(p_ret));                  \
	};                                                                     \
} while (0)

#define _STARPU_PTHREAD_GETSPECIFIC(key) pthread_getspecific((key))

/*
 * Encapsulation of the pthread_mutex_* functions.
 */
#define _STARPU_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define _STARPU_PTHREAD_MUTEX_INIT(mutex, attr) do {                           \
	int p_ret = pthread_mutex_init((mutex), (attr));                       \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_init: %s\n",                      \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_MUTEX_DESTROY(mutex) do {                              \
	int p_ret = pthread_mutex_destroy(mutex);                              \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_destroy: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while(0)

#define _STARPU_PTHREAD_MUTEX_LOCK(mutex) do {                                 \
	int p_ret = pthread_mutex_lock(mutex);                                 \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_lock: %s\n",                      \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_MUTEX_TRYLOCK(mutex) pthread_mutex_trylock(mutex)

#define _STARPU_PTHREAD_MUTEX_UNLOCK(mutex) do {                               \
	int p_ret = pthread_mutex_unlock(mutex);                               \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_unlock: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

typedef pthread_rwlock_t _starpu_pthread_rwlock_t;
/*
 * Encapsulation of the pthread_rwlock_* functions.
 */
#define _STARPU_PTHREAD_RWLOCK_INIT(rwlock, attr) do {                         \
	int p_ret = pthread_rwlock_init((rwlock), (attr));                     \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_init: %s\n",                     \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_RWLOCK_RDLOCK(rwlock) do {                             \
	int p_ret = pthread_rwlock_rdlock(rwlock);                             \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_rdlock: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_RWLOCK_WRLOCK(rwlock) do {                             \
	int p_ret = pthread_rwlock_wrlock(rwlock);                             \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_wrlock: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_RWLOCK_UNLOCK(rwlock) do {                             \
	int p_ret = pthread_rwlock_unlock(rwlock);                             \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_unlock: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_RWLOCK_DESTROY(rwlock) do {                            \
	int p_ret = pthread_rwlock_destroy(rwlock);                            \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_destroy: %s\n",                  \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

typedef pthread_cond_t _starpu_pthread_cond_t;
/*
 * Encapsulation of the pthread_cond_* functions.
 */
#define _STARPU_PTHREAD_COND_INITIALIZER PTHREAD_COND_INITIALIZER
#define _STARPU_PTHREAD_COND_INIT(cond, attr) do {                             \
	int p_ret = pthread_cond_init((cond), (attr));                         \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_init: %s\n",                       \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_COND_DESTROY(cond) do {                                \
	int p_ret = pthread_cond_destroy(cond);                                \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_destroy: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
			STARPU_ABORT();                                        \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_COND_SIGNAL(cond) do {                                 \
	int p_ret = pthread_cond_signal(cond);                                 \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_signal: %s\n",                     \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_COND_BROADCAST(cond) do {                              \
	int p_ret = pthread_cond_broadcast(cond);                              \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_broadcast: %s\n",                  \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_COND_WAIT(cond, mutex) do {                            \
	int p_ret = pthread_cond_wait((cond), (mutex));                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_wait: %s\n",                       \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#include <common/barrier.h>

typedef pthread_barrier_t _starpu_pthread_barrier_t;
/*
 * Encapsulation of the pthread_barrier_* functions.
 */
#define _STARPU_PTHREAD_BARRIER_INIT(barrier, attr, count) do {                \
	int p_ret = pthread_barrier_init((barrier), (attr), (count));          \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_barrier_init: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_BARRIER_DESTROY(barrier) do {                          \
	int p_ret = pthread_barrier_destroy((barrier));                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_barrier_destroy: %s\n",                 \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_BARRIER_WAIT(barrier) do {                             \
	int p_ret = pthread_barrier_wait(barrier);                             \
	if (STARPU_UNLIKELY(!((p_ret == 0) || (p_ret == PTHREAD_BARRIER_SERIAL_THREAD)))) { \
		fprintf(stderr,                                                \
			"%s:%d pthread_barrier_wait: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
			STARPU_ABORT();                                        \
	}                                                                      \
} while (0)

typedef pthread_spinlock_t _starpu_pthread_spinlock_t;
/*
 * Encapsulation of the pthread_spin_* functions.
 */
#define _STARPU_PTHREAD_SPIN_DESTROY(lock) do {                                \
	int p_ret = pthread_spin_destroy(lock);                                \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_spin_destroy: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_SPIN_LOCK(lock) do {                                   \
	int p_ret = pthread_spin_lock(lock);                                   \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_spin_lock: %s\n",                       \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#define _STARPU_PTHREAD_SPIN_UNLOCK(lock) do {                                 \
	int p_ret = pthread_spin_unlock(lock);                                 \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_spin_unlock: %s\n",                     \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)

#endif // __COMMON_UTILS_H__
