/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
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

int _starpu_mkpath(const char *s, mode_t mode);
int _starpu_check_mutex_deadlock(pthread_mutex_t *mutex);

#define PTHREAD_MUTEX_INIT(mutex, attr) { int ret = pthread_mutex_init((mutex), (attr)); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_mutex_init: %s\n", strerror(ret)); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_DESTROY(mutex) { int ret = pthread_mutex_destroy(mutex); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_mutex_destroy: %s\n", strerror(ret)); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_LOCK(mutex) { int ret = pthread_mutex_lock(mutex); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_mutex_lock : %s", strerror(ret)); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_UNLOCK(mutex) { int ret = pthread_mutex_unlock(mutex); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_mutex_unlock : %s", strerror(ret)); STARPU_ABORT(); }}

#define PTHREAD_RWLOCK_INIT(rwlock, attr) { int ret = pthread_rwlock_init((rwlock), (attr)); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_rwlock_init : %s", strerror(ret)); }}
#define PTHREAD_RWLOCK_RDLOCK(rwlock) { int ret = pthread_rwlock_rdlock(rwlock); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_rwlock_rdlock : %s", strerror(ret)); }}
#define PTHREAD_RWLOCK_WRLOCK(rwlock) { int ret = pthread_rwlock_wrlock(rwlock); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_rwlock_wrlock : %s", strerror(ret)); }}
#define PTHREAD_RWLOCK_UNLOCK(rwlock) { int ret = pthread_rwlock_unlock(rwlock); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_rwlock_unlock : %s", strerror(ret)); }}
#define PTHREAD_RWLOCK_DESTROY(rwlock) { int ret = pthread_rwlock_destroy(rwlock); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_rwlock_destroy : %s", strerror(ret)); }}

#define PTHREAD_COND_INIT(cond, attr) { int ret = pthread_cond_init((cond), (attr)); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_cond_init : %s", strerror(ret)); STARPU_ABORT(); }}
#define PTHREAD_COND_DESTROY(cond) { int ret = pthread_cond_destroy(cond); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_cond_destroy : %s", strerror(ret)); STARPU_ABORT();}}
#define PTHREAD_COND_SIGNAL(cond) { int ret = pthread_cond_signal(cond); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_cond_signal : %s", strerror(ret)); STARPU_ABORT();}}
#define PTHREAD_COND_BROADCAST(cond) { int ret = pthread_cond_broadcast(cond); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_cond_broadcast : %s", strerror(ret)); STARPU_ABORT();}}
#define PTHREAD_COND_WAIT(cond, mutex) { int ret = pthread_cond_wait((cond), (mutex)); if (STARPU_UNLIKELY(ret)) { fprintf(stderr, "pthread_cond_wait : %s", strerror(ret)); STARPU_ABORT();}}

#endif // __COMMON_UTILS_H__
