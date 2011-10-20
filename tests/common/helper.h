/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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

#include <errno.h>

#define STARPU_TEST_SKIPPED 77

//void *ALL_IS_OK = (void *)123456789L;
//void *ALL_IS_NOT_OK = (void *)987654321L;
//
//#define STARPU_CHECK_MALLOC(ptr) {if (!ptr) { fprintf(stderr, "starpu_malloc failed\n"); return 1; }}
//#define STARPU_CHECK_MALLOC_HAS_FAILED(ptr) {if (ptr) { fprintf(stderr, "starpu_malloc should have failed\n"); return 1; }}
#define STARPU_CHECK_RETURN_VALUE(err, message) {if (err < 0) { perror(message); STARPU_ASSERT(0); }}
#define STARPU_CHECK_RETURN_VALUE_IS(err, value, message) {if (err != value) { perror(message); return 1; }}
//
//#define STARPU_CHECK_MALLOC_THREAD(ptr) {if (!ptr) { fprintf(stderr, "starpu_malloc failed\n"); return ALL_IS_NOT_OK; }}
//#define STARPU_CHECK_MALLOC_HAS_FAILED_THREAD(ptr) {if (ptr) { fprintf(stderr, "starpu_malloc should have failed\n"); return ALL_IS_NOT_OK; }}
//#define STARPU_CHECK_RETURN_VALUE_THREAD(err, message) {if (err < 0) { perror(message); return ALL_IS_NOT_OK; }}
//#define STARPU_CHECK_RETURN_VALUE_IS_THREAD(err, value, message) {if (err >= 0 || errno != value) { perror(message); return ALL_IS_NOT_OK; }}

//#define STARPU_TEST_OUTPUT
#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)


#define PTHREAD_MUTEX_INIT(mutex, attr) { int p_ret = pthread_mutex_init((mutex), (attr)); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_mutex_init: %s\n", strerror(p_ret)); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_DESTROY(mutex) { int p_ret = pthread_mutex_destroy(mutex); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_mutex_destroy: %s\n", strerror(p_ret)); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_LOCK(mutex) { int p_ret = pthread_mutex_lock(mutex); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_mutex_lock : %s\n", strerror(p_ret)); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_UNLOCK(mutex) { int p_ret = pthread_mutex_unlock(mutex); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_mutex_unlock : %s\n", strerror(p_ret)); STARPU_ABORT(); }}

#define PTHREAD_RWLOCK_INIT(rwlock, attr) { int p_ret = pthread_rwlock_init((rwlock), (attr)); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_rwlock_init : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_RWLOCK_RDLOCK(rwlock) { int p_ret = pthread_rwlock_rdlock(rwlock); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_rwlock_rdlock : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_RWLOCK_WRLOCK(rwlock) { int p_ret = pthread_rwlock_wrlock(rwlock); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_rwlock_wrlock : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_RWLOCK_UNLOCK(rwlock) { int p_ret = pthread_rwlock_unlock(rwlock); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_rwlock_unlock : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_RWLOCK_DESTROY(rwlock) { int p_ret = pthread_rwlock_destroy(rwlock); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_rwlock_destroy : %s\n", strerror(p_ret)); STARPU_ABORT();}}

#define PTHREAD_COND_INIT(cond, attr) { int p_ret = pthread_cond_init((cond), (attr)); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_cond_init : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_COND_DESTROY(cond) { int p_ret = pthread_cond_destroy(cond); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_cond_destroy : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_COND_SIGNAL(cond) { int p_ret = pthread_cond_signal(cond); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_cond_signal : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_COND_BROADCAST(cond) { int p_ret = pthread_cond_broadcast(cond); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_cond_broadcast : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_COND_WAIT(cond, mutex) { int p_ret = pthread_cond_wait((cond), (mutex)); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_cond_wait : %s\n", strerror(p_ret)); STARPU_ABORT();}}

#define PTHREAD_BARRIER_INIT(barrier, attr, count) { int p_ret = pthread_barrier_init((barrier), (attr), (count)); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_barrier_init : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_BARRIER_DESTROY(barrier) { int p_ret = pthread_barrier_destroy((barrier)); if (STARPU_UNLIKELY(p_ret)) { fprintf(stderr, "pthread_barrier_destroy : %s\n", strerror(p_ret)); STARPU_ABORT();}}
#define PTHREAD_BARRIER_WAIT(barrier) { int p_ret = pthread_barrier_wait(barrier); if (STARPU_UNLIKELY(!((p_ret == 0) || (p_ret == PTHREAD_BARRIER_SERIAL_THREAD)))) { fprintf(stderr, "pthread_barrier_wait : %s\n", strerror(p_ret)); STARPU_ABORT();}}
