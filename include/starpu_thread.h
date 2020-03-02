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

#ifndef __STARPU_THREAD_H__
#define __STARPU_THREAD_H__

#include <starpu_config.h>
#include <starpu_util.h>
#ifdef STARPU_SIMGRID
#include <pthread.h>
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
#include <simgrid/mutex.h>
#include <simgrid/cond.h>
#elif defined(STARPU_HAVE_XBT_SYNCHRO_H)
#include <xbt/synchro.h>
#else
#include <xbt/synchro_core.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_ACTOR_H
#include <simgrid/actor.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
#include <simgrid/semaphore.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
#include <simgrid/mutex.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_COND_H
#include <simgrid/cond.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_BARRIER_H
#include <simgrid/barrier.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_HOST_H
#include <simgrid/host.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_MSG_H
#include <simgrid/msg.h>
#elif defined(STARPU_HAVE_MSG_MSG_H)
#include <msg/msg.h>
#endif
#elif !defined(_MSC_VER) || defined(BUILDING_STARPU)
#include <pthread.h>
#include <semaphore.h>
#endif
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Encapsulation of the pthread_create function.
 */

#ifdef STARPU_SIMGRID

#ifdef STARPU_HAVE_SIMGRID_ACTOR_H
typedef sg_actor_t starpu_pthread_t;
#else
typedef msg_process_t starpu_pthread_t;
#endif
typedef int starpu_pthread_attr_t;

#ifdef STARPU_HAVE_SIMGRID_ACTOR_H
typedef sg_host_t starpu_sg_host_t;
#else
typedef msg_host_t starpu_sg_host_t;
#endif
int starpu_pthread_equal(starpu_pthread_t t1, starpu_pthread_t t2);
starpu_pthread_t starpu_pthread_self(void);
int starpu_pthread_create_on(const char *name, starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg, starpu_sg_host_t host);
int starpu_pthread_create(starpu_pthread_t *thread, const starpu_pthread_attr_t *attr, void *(*start_routine) (void *), void *arg);
starpu_pthread_t _starpu_simgrid_actor_create(const char *name, xbt_main_func_t code, starpu_sg_host_t host, int argc, char *argv[]);
int starpu_pthread_join(starpu_pthread_t thread, void **retval);
int starpu_pthread_exit(void *retval) STARPU_ATTRIBUTE_NORETURN;
int starpu_pthread_attr_init(starpu_pthread_attr_t *attr);
int starpu_pthread_attr_destroy(starpu_pthread_attr_t *attr);
int starpu_pthread_attr_setdetachstate(starpu_pthread_attr_t *attr, int detachstate);

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* STARPU_SIMGRID */

typedef pthread_t starpu_pthread_t;
typedef pthread_attr_t starpu_pthread_attr_t;

#define starpu_pthread_equal pthread_equal
#define starpu_pthread_self pthread_self
#define starpu_pthread_create pthread_create
#define starpu_pthread_create_on(name, thread, attr, routine, arg, where) starpu_pthread_create(thread, attr, routine, arg)
#define starpu_pthread_join pthread_join
#define starpu_pthread_exit pthread_exit
#define starpu_pthread_attr_init pthread_attr_init
#define starpu_pthread_attr_destroy pthread_attr_destroy
#define starpu_pthread_attr_setdetachstate pthread_attr_setdetachstate

#endif /* STARPU_SIMGRID, _MSC_VER */

#ifdef STARPU_HAVE_PTHREAD_SETNAME_NP
#ifdef STARPU_HAVE_DARWIN
#define starpu_pthread_setname(name) pthread_setname_np(name)
#else
#define starpu_pthread_setname(name) pthread_setname_np(pthread_self(), name)
#endif
#else
#define starpu_pthread_setname(name)
#endif

/*
 * Encapsulation of the pthread_mutex_* functions.
 */

#ifdef STARPU_SIMGRID
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
typedef sg_mutex_t starpu_pthread_mutex_t;
#else
typedef xbt_mutex_t starpu_pthread_mutex_t;
#endif
typedef int starpu_pthread_mutexattr_t;

#define STARPU_PTHREAD_MUTEX_INITIALIZER NULL

int starpu_pthread_mutex_init(starpu_pthread_mutex_t *mutex, const starpu_pthread_mutexattr_t *mutexattr);
int starpu_pthread_mutex_destroy(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutexattr_gettype(const starpu_pthread_mutexattr_t *attr, int *type);
int starpu_pthread_mutexattr_settype(starpu_pthread_mutexattr_t *attr, int type);
int starpu_pthread_mutexattr_destroy(starpu_pthread_mutexattr_t *attr);
int starpu_pthread_mutexattr_init(starpu_pthread_mutexattr_t *attr);

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

typedef pthread_mutex_t starpu_pthread_mutex_t;
typedef pthread_mutexattr_t starpu_pthread_mutexattr_t;

#define starpu_pthread_mutex_init pthread_mutex_init
#define starpu_pthread_mutex_destroy pthread_mutex_destroy
#define starpu_pthread_mutexattr_gettype pthread_mutexattr_gettype
#define starpu_pthread_mutexattr_settype pthread_mutexattr_settype
#define starpu_pthread_mutexattr_destroy pthread_mutexattr_destroy
#define starpu_pthread_mutexattr_init pthread_mutexattr_init

#ifdef STARPU_FXT_LOCK_TRACES
int starpu_pthread_mutex_lock(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_unlock(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_trylock(starpu_pthread_mutex_t *mutex);
#else
#define starpu_pthread_mutex_lock pthread_mutex_lock
#define starpu_pthread_mutex_unlock pthread_mutex_unlock
#define starpu_pthread_mutex_trylock pthread_mutex_trylock
#endif

#define STARPU_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

#endif /* STARPU_SIMGRID, _MSC_VER */

#if !defined(_MSC_VER) || defined(BUILDING_STARPU)
int starpu_pthread_mutex_lock_sched(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_unlock_sched(starpu_pthread_mutex_t *mutex);
int starpu_pthread_mutex_trylock_sched(starpu_pthread_mutex_t *mutex);
void starpu_pthread_mutex_check_sched(starpu_pthread_mutex_t *mutex, char *file, int line);
#endif

/*
 * Encapsulation of the pthread_key_* functions.
 */
#ifdef STARPU_SIMGRID

typedef int starpu_pthread_key_t;
int starpu_pthread_key_create(starpu_pthread_key_t *key, void (*destr_function) (void *));
int starpu_pthread_key_delete(starpu_pthread_key_t key);
int starpu_pthread_setspecific(starpu_pthread_key_t key, const void *pointer);
void *starpu_pthread_getspecific(starpu_pthread_key_t key);

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

typedef pthread_key_t starpu_pthread_key_t;

#define starpu_pthread_key_create pthread_key_create
#define starpu_pthread_key_delete pthread_key_delete
#define starpu_pthread_setspecific pthread_setspecific
#define starpu_pthread_getspecific pthread_getspecific

#endif /* STARPU_SIMGRID, _MSC_VER */

/*
 * Encapsulation of the pthread_cond_* functions.
 */

#ifdef STARPU_SIMGRID

#ifdef STARPU_HAVE_SIMGRID_COND_H
typedef sg_cond_t starpu_pthread_cond_t;
#else
typedef xbt_cond_t starpu_pthread_cond_t;
#endif
typedef int starpu_pthread_condattr_t;
#define STARPU_PTHREAD_COND_INITIALIZER NULL

int starpu_pthread_cond_init(starpu_pthread_cond_t *cond, starpu_pthread_condattr_t *cond_attr);
int starpu_pthread_cond_signal(starpu_pthread_cond_t *cond);
int starpu_pthread_cond_broadcast(starpu_pthread_cond_t *cond);
int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex);
int starpu_pthread_cond_timedwait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, const struct timespec *abstime);
int starpu_pthread_cond_destroy(starpu_pthread_cond_t *cond);

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

typedef pthread_cond_t starpu_pthread_cond_t;
typedef pthread_condattr_t starpu_pthread_condattr_t;
#define STARPU_PTHREAD_COND_INITIALIZER PTHREAD_COND_INITIALIZER

#define starpu_pthread_cond_init pthread_cond_init
#define starpu_pthread_cond_signal pthread_cond_signal
#define starpu_pthread_cond_broadcast pthread_cond_broadcast

#ifdef STARPU_FXT_LOCK_TRACES
int starpu_pthread_cond_wait(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex);
#else
#define starpu_pthread_cond_wait pthread_cond_wait
#endif

#define starpu_pthread_cond_timedwait pthread_cond_timedwait
#define starpu_pthread_cond_destroy pthread_cond_destroy

#endif /* STARPU_SIMGRID, _MSC_VER */

/*
 * Encapsulation of the pthread_rwlock_* functions.
 */

#ifdef STARPU_SIMGRID

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
typedef sg_mutex_t starpu_pthread_rwlock_t;
#else
typedef xbt_mutex_t starpu_pthread_rwlock_t;
#endif
typedef int starpu_pthread_rwlockattr_t;

int starpu_pthread_rwlock_init(starpu_pthread_rwlock_t *rwlock, const starpu_pthread_rwlockattr_t *attr);
int starpu_pthread_rwlock_destroy(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock);

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

typedef pthread_rwlock_t starpu_pthread_rwlock_t;
typedef pthread_rwlockattr_t starpu_pthread_rwlockattr_t;

#define starpu_pthread_rwlock_init pthread_rwlock_init
#define starpu_pthread_rwlock_destroy pthread_rwlock_destroy

#ifdef STARPU_FXT_LOCK_TRACES
int starpu_pthread_rwlock_rdlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_tryrdlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_wrlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_trywrlock(starpu_pthread_rwlock_t *rwlock);
int starpu_pthread_rwlock_unlock(starpu_pthread_rwlock_t *rwlock);
#else
#define starpu_pthread_rwlock_rdlock pthread_rwlock_rdlock
#define starpu_pthread_rwlock_tryrdlock pthread_rwlock_tryrdlock
#define starpu_pthread_rwlock_wrlock pthread_rwlock_wrlock
#define starpu_pthread_rwlock_trywrlock pthread_rwlock_trywrlock
#define starpu_pthread_rwlock_unlock pthread_rwlock_unlock
#endif

#endif /* STARPU_SIMGRID, _MSC_VER */

/*
 * Encapsulation of the pthread_barrier_* functions.
 */

#if defined(STARPU_SIMGRID) || (!defined(STARPU_HAVE_PTHREAD_BARRIER) && (!defined(_MSC_VER) || defined(BUILDING_STARPU)))

#if defined(STARPU_SIMGRID) && (defined(STARPU_HAVE_SIMGRID_BARRIER_H) || defined(STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT) || defined(xbt_barrier_init))
#ifdef STARPU_HAVE_SIMGRID_BARRIER_H
typedef sg_bar_t starpu_pthread_barrier_t;
#else
typedef xbt_bar_t starpu_pthread_barrier_t;
#endif
typedef int starpu_pthread_barrierattr_t;
#define STARPU_PTHREAD_BARRIER_SERIAL_THREAD -1
#else
typedef struct {
	starpu_pthread_mutex_t mutex;
	starpu_pthread_cond_t cond;
	starpu_pthread_cond_t cond_destroy;
	unsigned count;
	unsigned done;
	unsigned busy;
} starpu_pthread_barrier_t;
typedef int starpu_pthread_barrierattr_t;
#define STARPU_PTHREAD_BARRIER_SERIAL_THREAD -1
#endif

int starpu_pthread_barrier_init(starpu_pthread_barrier_t *barrier, const starpu_pthread_barrierattr_t *attr, unsigned count);
int starpu_pthread_barrier_destroy(starpu_pthread_barrier_t *barrier);
int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier);

#elif !defined(_MSC_VER) /* STARPU_SIMGRID, !STARPU_HAVE_PTHREAD_BARRIER */

typedef pthread_barrier_t starpu_pthread_barrier_t;
typedef pthread_barrierattr_t starpu_pthread_barrierattr_t;

#define starpu_pthread_barrier_init pthread_barrier_init
#define starpu_pthread_barrier_destroy pthread_barrier_destroy

#ifdef STARPU_FXT_LOCK_TRACES
int starpu_pthread_barrier_wait(starpu_pthread_barrier_t *barrier);
#else
#define starpu_pthread_barrier_wait pthread_barrier_wait
#endif
#define STARPU_PTHREAD_BARRIER_SERIAL_THREAD PTHREAD_BARRIER_SERIAL_THREAD

#endif /* STARPU_SIMGRID, !STARPU_HAVE_PTHREAD_BARRIER, _MSC_VER */

/*
 * Encapsulation of the pthread_spin_* functions.
 */

#if defined(STARPU_SIMGRID) || (defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK)

typedef struct
{
#ifdef STARPU_SIMGRID
	int taken;
#elif defined(STARPU_LINUX_SYS) && defined(STARPU_HAVE_XCHG)
	unsigned taken STARPU_ATTRIBUTE_ALIGNED(16);
#else /* we only have a trivial implementation yet ! */
	uint32_t taken STARPU_ATTRIBUTE_ALIGNED(16);
#endif
} starpu_pthread_spinlock_t;

int starpu_pthread_spin_init(starpu_pthread_spinlock_t *lock, int pshared);
int starpu_pthread_spin_destroy(starpu_pthread_spinlock_t *lock);
int starpu_pthread_spin_lock(starpu_pthread_spinlock_t *lock);
int starpu_pthread_spin_trylock(starpu_pthread_spinlock_t *lock);
int starpu_pthread_spin_unlock(starpu_pthread_spinlock_t *lock);

#elif !defined(_MSC_VER) /* !( defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK)) */

typedef pthread_spinlock_t starpu_pthread_spinlock_t;
#define starpu_pthread_spin_init pthread_spin_init
#define starpu_pthread_spin_destroy pthread_spin_destroy
#define starpu_pthread_spin_lock pthread_spin_lock
#define starpu_pthread_spin_trylock pthread_spin_trylock
#define starpu_pthread_spin_unlock pthread_spin_unlock

#endif /* !( defined(STARPU_SIMGRID) || !defined(STARPU_HAVE_PTHREAD_SPIN_LOCK)) */

/*
 * Other needed pthread definitions
 */

#if defined(_MSC_VER) && !defined(BUILDING_STARPU)
typedef void* starpu_pthread_rwlock_t;
typedef void* starpu_pthread_mutex_t;
typedef void* starpu_pthread_cond_t;
typedef void* starpu_pthread_barrier_t;
#endif /* _MSC_VER */

/*
 * Simgrid-specific register/wait synchronization
 *
 * Producers create a "queue" object, and when they have produced something,
 * they call either queue_signal or queue_broadcast in order to wake either one
 * or all consumers waiting on the queue.
 *
 * starpu_pthread_queue_init(&global_queue1->queue);
 * while (1) {
 * 	element = compute();
 * 	push(element, global_queue1);
 * 	starpu_pthread_queue_signal(global_queue1);
 * }
 * starpu_pthread_queue_destroy(&global_queue1->queue);
 *
 * Consumers create a "wait" object, then queue_register on as many queues they
 * want. In their consumption loop, they wait_reset, then test for availibility
 * on all producers, and if none was available, call wait_wait to actually wait
 * for producers. On termination, consumers have to queue_unregister before
 * destroying the "wait" object:
 *
 * starpu_pthread_wait_t wait;
 *
 * starpu_pthread_wait_init(&wait);
 * starpu_pthread_queue_register(&wait, &global_queue1->queue);
 * starpu_pthread_queue_register(&wait, &global_queue2->queue);
 *
 * while (1) {
 * 	int sleep = 1;
 * 	starpu_pthread_wait_reset(&wait);
 * 	if (global_queue1->navailable)
 * 	{
 * 		work(global_queue1);
 * 		sleep = 0;
 * 	}
 * 	if (global_queue2->navailable)
 * 	{
 * 		work(global_queue2);
 * 		sleep = 0;
 * 	}
 * 	if (sleep)
 * 		starpu_pthread_wait_wait(&wait);
 * }
 * starpu_pthread_queue_unregister(&wait, &global_queue1->queue);
 * starpu_pthread_queue_unregister(&wait, &global_queue2->queue);
 * starpu_pthread_wait_destroy(&wait);
 */

#ifdef STARPU_SIMGRID
typedef struct
{
	starpu_pthread_mutex_t mutex;
	starpu_pthread_cond_t cond;
	unsigned block;
} starpu_pthread_wait_t;

typedef struct
{
	starpu_pthread_mutex_t mutex;
	starpu_pthread_wait_t **queue;
	unsigned allocqueue;
	unsigned nqueue;
} starpu_pthread_queue_t;

int starpu_pthread_queue_init(starpu_pthread_queue_t *q);
int starpu_pthread_queue_signal(starpu_pthread_queue_t *q);
int starpu_pthread_queue_broadcast(starpu_pthread_queue_t *q);
int starpu_pthread_queue_destroy(starpu_pthread_queue_t *q);

int starpu_pthread_wait_init(starpu_pthread_wait_t *w);
int starpu_pthread_queue_register(starpu_pthread_wait_t *w, starpu_pthread_queue_t *q);
int starpu_pthread_queue_unregister(starpu_pthread_wait_t *w, starpu_pthread_queue_t *q);
int starpu_pthread_wait_reset(starpu_pthread_wait_t *w);
int starpu_pthread_wait_wait(starpu_pthread_wait_t *w);
int starpu_pthread_wait_timedwait(starpu_pthread_wait_t *w, const struct timespec *abstime);
int starpu_pthread_wait_destroy(starpu_pthread_wait_t *w);
#endif

/*
 * Encapsulation of the semaphore functions.
 */

#ifdef STARPU_SIMGRID

#ifdef STARPU_HAVE_SIMGRID_SEMAPHORE_H
typedef sg_sem_t starpu_sem_t;
#else
typedef msg_sem_t starpu_sem_t;
#endif
int starpu_sem_destroy(starpu_sem_t *);
int starpu_sem_getvalue(starpu_sem_t *, int *);
int starpu_sem_init(starpu_sem_t *, int, unsigned);
int starpu_sem_post(starpu_sem_t *);
int starpu_sem_trywait(starpu_sem_t *);
int starpu_sem_wait(starpu_sem_t *);

#elif !defined(_MSC_VER) || defined(BUILDING_STARPU) /* !STARPU_SIMGRID */

typedef sem_t starpu_sem_t;
#define starpu_sem_destroy sem_destroy
#define starpu_sem_getvalue sem_getvalue
#define starpu_sem_init sem_init
#define starpu_sem_post sem_post
int starpu_sem_trywait(starpu_sem_t *);
int starpu_sem_wait(starpu_sem_t *);

#endif

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_THREAD_H__ */
