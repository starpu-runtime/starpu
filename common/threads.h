#ifndef __THREADS_H__
#define __THREADS_H__

#ifdef USE_MARCEL
#include <marcel.h>
#else
#include <pthread.h>
#endif

#ifdef USE_MARCEL
typedef marcel_t thread_t;
typedef marcel_attr_t thread_attr_t;
typedef marcel_mutexattr_t thread_mutexattr_t;
typedef marcel_mutex_t thread_mutex_t;
#else
typedef pthread_t thread_t;
typedef pthread_attr_t thread_attr_t;
typedef pthread_mutexattr_t thread_mutexattr_t;
typedef pthread_mutex_t thread_mutex_t;
#endif

void thread_exit(void *);
int thread_join(thread_t, void **);
int thread_mutex_init(thread_mutex_t *,  const  thread_mutexattr_t *);
int thread_mutex_lock(thread_mutex_t *);
int thread_mutex_unlock(thread_mutex_t *);
int thread_create(thread_t  *, thread_attr_t *, void *(*start_routine)(void *), void *);




#endif // __THREADS_H__
