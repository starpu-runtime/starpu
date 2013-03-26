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

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <starpu.h>
#include <common/config.h>
#include <sys/stat.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>

#ifdef STARPU_SIMGRID
#include <xbt/synchro_core.h>
#include <msg/msg.h>
#endif

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif

#ifndef VALGRIND_HG_MUTEX_LOCK_PRE
#define VALGRIND_HG_MUTEX_LOCK_PRE(mutex, istrylock) ((void)0)
#endif
#ifndef VALGRIND_HG_MUTEX_LOCK_POST
#define VALGRIND_HG_MUTEX_LOCK_POST(mutex) ((void)0)
#endif
#ifndef VALGRIND_HG_MUTEX_UNLOCK_PRE
#define VALGRIND_HG_MUTEX_UNLOCK_PRE(mutex) ((void)0)
#endif
#ifndef VALGRIND_HG_MUTEX_UNLOCK_POST
#define VALGRIND_HG_MUTEX_UNLOCK_POST(mutex) ((void)0)
#endif
#ifndef DO_CREQ_v_WW
#define DO_CREQ_v_WW(_creqF, _ty1F, _arg1F, _ty2F, _arg2F) ((void)0)
#endif
#ifndef DO_CREQ_v_W
#define DO_CREQ_v_W(_creqF, _ty1F, _arg1F) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE
#define ANNOTATE_HAPPENS_BEFORE(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif
#ifndef ANNOTATE_RWLOCK_ACQUIRED
#define ANNOTATE_RWLOCK_ACQUIRED(lock, is_w) ((void)0)
#endif
#ifndef ANNOTATE_RWLOCK_RELEASED
#define ANNOTATE_RWLOCK_RELEASED(lock, is_w) ((void)0)
#endif

#define _STARPU_VALGRIND_HG_SPIN_LOCK_PRE(lock) \
	DO_CREQ_v_WW(_VG_USERREQ__HG_PTHREAD_SPIN_LOCK_PRE, \
			struct _starpu_spinlock *, lock, long, 0)
#define _STARPU_VALGRIND_HG_SPIN_LOCK_POST(lock) \
	DO_CREQ_v_W(_VG_USERREQ__HG_PTHREAD_SPIN_LOCK_POST, \
			struct _starpu_spinlock *, lock)
#define _STARPU_VALGRIND_HG_SPIN_UNLOCK_PRE(lock) \
	DO_CREQ_v_W(_VG_USERREQ__HG_PTHREAD_SPIN_INIT_OR_UNLOCK_PRE, \
			struct _starpu_spinlock *, lock)
#define _STARPU_VALGRIND_HG_SPIN_UNLOCK_POST(lock) \
	DO_CREQ_v_W(_VG_USERREQ__HG_PTHREAD_SPIN_INIT_OR_UNLOCK_POST, \
			struct _starpu_spinlock *, lock)

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
                fprintf(stderr, "\n\n[starpu][%s] Error: " fmt ,__func__ ,##args);    \
		fprintf(stderr, "\n\n");					      \
		STARPU_ABORT();                                                       \
	} while (0)


#define _STARPU_IS_ZERO(a) (fpclassify(a) == FP_ZERO)

#ifdef STARPU_SIMGRID
typedef xbt_mutex_t _starpu_pthread_mutex_t;
#else
typedef pthread_mutex_t _starpu_pthread_mutex_t;
#endif
int _starpu_mkpath(const char *s, mode_t mode);
void _starpu_mkpath_and_check(const char *s, mode_t mode);
int _starpu_check_mutex_deadlock(_starpu_pthread_mutex_t *mutex);
char *_starpu_get_home_path(void);
void _starpu_gethostname(char *hostname, size_t size);

/* If FILE is currently on a comment line, eat it.  */
void _starpu_drop_comments(FILE *f);

struct _starpu_job;
/* Returns the symbol associated to that job if any. */
const char *_starpu_job_get_model_name(struct _starpu_job *j);

struct starpu_codelet;
/* Returns the symbol associated to that job if any. */
const char *_starpu_codelet_get_model_name(struct starpu_codelet *cl);

struct _starpu_pthread_args {
	void *(*f)(void*);
	void *arg;
};

int _starpu_simgrid_thread_start(int argc, char *argv[]);
#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_CREATE_ON(name, thread, attr, routine, threadarg, where) do {\
	struct _starpu_pthread_args *_args = malloc(sizeof(*_args));           \
	xbt_dynar_t _hosts;                                                    \
	_args->f = routine;                                                    \
	_args->arg = threadarg;                                                \
	_hosts = MSG_hosts_as_dynar();                                         \
	MSG_process_create((name), _starpu_simgrid_thread_start, _args,        \
			xbt_dynar_get_as(_hosts, (where), msg_host_t));        \
	xbt_dynar_free(&_hosts);                                               \
} while (0)
#else
#define _STARPU_PTHREAD_CREATE_ON(name, thread, attr, routine, arg, where) do {\
	int p_ret = pthread_create((thread), (attr), (routine), (arg));	       \
	if (STARPU_UNLIKELY(p_ret != 0)) {                                     \
		fprintf(stderr,                                                \
			"%s:%d pthread_create: %s\n",                          \
			__FILE__, __LINE__, strerror(p_ret));                  \
	}                                                                      \
} while (0)
#endif
#define _STARPU_PTHREAD_CREATE(name, thread, attr, routine, arg)               \
	_STARPU_PTHREAD_CREATE_ON(name, thread, attr, routine, arg, 0)

/*
 * Encapsulation of the pthread_key_* functions.
 */
#ifdef STARPU_SIMGRID
typedef int _starpu_pthread_key_t;
int _starpu_pthread_key_create(_starpu_pthread_key_t *key);
#define _STARPU_PTHREAD_KEY_CREATE(key, destr) _starpu_pthread_key_create(key)
int _starpu_pthread_key_delete(_starpu_pthread_key_t key);
#define _STARPU_PTHREAD_KEY_DELETE(key) _starpu_pthread_key_delete(key)
int _starpu_pthread_setspecific(_starpu_pthread_key_t key, void *ptr);
#define _STARPU_PTHREAD_SETSPECIFIC(key, ptr) _starpu_pthread_setspecific(key, ptr)
void *_starpu_pthread_getspecific(_starpu_pthread_key_t key);
#define _STARPU_PTHREAD_GETSPECIFIC(key) _starpu_pthread_getspecific(key)
#else
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
#endif

/*
 * Encapsulation of the pthread_mutex_* functions.
 */
#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_MUTEX_INITIALIZER NULL
#define _STARPU_PTHREAD_MUTEX_INIT(mutex, attr) do {                           \
	(*mutex) = xbt_mutex_init();                                           \
} while (0)
#else
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
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_MUTEX_DESTROY(mutex) do {                              \
	if (*mutex)                                                            \
		xbt_mutex_destroy((*mutex));                                   \
} while (0)
#else
#define _STARPU_PTHREAD_MUTEX_DESTROY(mutex) do {                              \
	int p_ret = pthread_mutex_destroy(mutex);                              \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_destroy: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while(0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_MUTEX_LOCK(mutex) do {                                 \
	if (!(*mutex)) _STARPU_PTHREAD_MUTEX_INIT((mutex), NULL);              \
	xbt_mutex_acquire((*mutex));                                           \
} while (0)
#else
#define _STARPU_PTHREAD_MUTEX_LOCK(mutex) do {                                 \
	int p_ret = pthread_mutex_lock(mutex);                                 \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_lock: %s\n",                      \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_MUTEX_TRYLOCK(mutex) (xbt_mutex_acquire(*mutex), 0)
#else
#define _STARPU_PTHREAD_MUTEX_TRYLOCK(mutex) pthread_mutex_trylock(mutex)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_MUTEX_UNLOCK(mutex) do {                               \
	xbt_mutex_release((*mutex));                                           \
} while (0)
#else
#define _STARPU_PTHREAD_MUTEX_UNLOCK(mutex) do {                               \
	int p_ret = pthread_mutex_unlock(mutex);                               \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_mutex_unlock: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
typedef xbt_mutex_t _starpu_pthread_rwlock_t;
#else
typedef pthread_rwlock_t _starpu_pthread_rwlock_t;
#endif
/*
 * Encapsulation of the pthread_rwlock_* functions.
 */
#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_RWLOCK_INIT(rwlock, attr) _STARPU_PTHREAD_MUTEX_INIT(rwlock, attr)
#else
#define _STARPU_PTHREAD_RWLOCK_INIT(rwlock, attr) do {                         \
	int p_ret = pthread_rwlock_init((rwlock), (attr));                     \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_init: %s\n",                     \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_RWLOCK_RDLOCK(rwlock) _STARPU_PTHREAD_MUTEX_LOCK(rwlock)
#else
#define _STARPU_PTHREAD_RWLOCK_RDLOCK(rwlock) do {                             \
	int p_ret = pthread_rwlock_rdlock(rwlock);                             \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_rdlock: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_RWLOCK_WRLOCK(rwlock) _STARPU_PTHREAD_MUTEX_LOCK(rwlock)
#else
#define _STARPU_PTHREAD_RWLOCK_WRLOCK(rwlock) do {                             \
	int p_ret = pthread_rwlock_wrlock(rwlock);                             \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_wrlock: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_RWLOCK_UNLOCK(rwlock) _STARPU_PTHREAD_MUTEX_UNLOCK(rwlock)
#else
#define _STARPU_PTHREAD_RWLOCK_UNLOCK(rwlock) do {                             \
	int p_ret = pthread_rwlock_unlock(rwlock);                             \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_unlock: %s\n",                   \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_RWLOCK_DESTROY(rwlock) _STARPU_PTHREAD_MUTEX_DESTROY(rwlock)
#else
#define _STARPU_PTHREAD_RWLOCK_DESTROY(rwlock) do {                            \
	int p_ret = pthread_rwlock_destroy(rwlock);                            \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_rwlock_destroy: %s\n",                  \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
typedef xbt_cond_t _starpu_pthread_cond_t;
#else
typedef pthread_cond_t _starpu_pthread_cond_t;
#endif
/*
 * Encapsulation of the pthread_cond_* functions.
 */
#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_COND_INITIALIZER NULL
#define _STARPU_PTHREAD_COND_INIT(cond, attr) do {                             \
	(*cond) = xbt_cond_init();                                             \
} while (0)
#else
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
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_COND_DESTROY(cond) do {                                \
	if (*cond)                                                             \
		xbt_cond_destroy((*cond));                                     \
} while (0)
#else
#define _STARPU_PTHREAD_COND_DESTROY(cond) do {                                \
	int p_ret = pthread_cond_destroy(cond);                                \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_destroy: %s\n",                    \
			__FILE__, __LINE__, strerror(p_ret));                  \
			STARPU_ABORT();                                        \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_COND_SIGNAL(cond) do {                                 \
	if (!*cond)                                                            \
		_STARPU_PTHREAD_COND_INIT(cond, NULL);                         \
	xbt_cond_signal((*cond));                                              \
} while (0)
#else
#define _STARPU_PTHREAD_COND_SIGNAL(cond) do {                                 \
	int p_ret = pthread_cond_signal(cond);                                 \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_signal: %s\n",                     \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_COND_BROADCAST(cond) do {                              \
	if (!*cond)                                                            \
		_STARPU_PTHREAD_COND_INIT(cond, NULL);                         \
	xbt_cond_broadcast((*cond));                                           \
} while (0)
#else
#define _STARPU_PTHREAD_COND_BROADCAST(cond) do {                              \
	int p_ret = pthread_cond_broadcast(cond);                              \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_broadcast: %s\n",                  \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#ifdef STARPU_SIMGRID
#define _STARPU_PTHREAD_COND_WAIT(cond, mutex) do {                            \
	if (!*cond)                                                            \
		_STARPU_PTHREAD_COND_INIT(cond, NULL);                         \
	xbt_cond_wait((*cond), (*mutex));                                      \
} while (0)
#else
#define _STARPU_PTHREAD_COND_WAIT(cond, mutex) do {                            \
	int p_ret = pthread_cond_wait((cond), (mutex));                        \
	if (STARPU_UNLIKELY(p_ret)) {                                          \
		fprintf(stderr,                                                \
			"%s:%d pthread_cond_wait: %s\n",                       \
			__FILE__, __LINE__, strerror(p_ret));                  \
		STARPU_ABORT();                                                \
	}                                                                      \
} while (0)
#endif

#define _starpu_pthread_barrier_t pthread_barrier_t
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

#ifdef HAVE_PTHREAD_SPIN_LOCK
typedef pthread_spinlock_t _starpu_pthread_spinlock_t;
#endif

#endif // __COMMON_UTILS_H__
