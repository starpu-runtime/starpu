#include "threads.h"

#ifdef USE_MARCEL

void thread_exit(void *retval)
{
	marcel_exit(retval);
}

int  thread_create(thread_t  *  thread, thread_attr_t * attr, void *(*start_routine)(void *), void * arg)
{
	return marcel_create(thread, attr, start_routine, arg);
}

int thread_join(thread_t th, void **thread_return)
{
	return marcel_join(th, thread_return);
}

int thread_mutex_init(thread_mutex_t  *mutex,  const  thread_mutexattr_t *mutexattr)
{
	return marcel_mutex_init(mutex, mutexattr);
}

int thread_mutex_lock(thread_mutex_t *mutex)
{
	return marcel_mutex_lock(mutex);
}

int thread_mutex_unlock(thread_mutex_t *mutex)
{
	return marcel_mutex_unlock(mutex);
}


#else // USE_MARCEL is false 

void thread_exit(void *retval)
{
	pthread_exit(retval);
}

int thread_create(thread_t  *  thread, thread_attr_t * attr, void *
       (*start_routine)(void *), void * arg) {
	return pthread_create(thread, attr, start_routine, arg);
}

int thread_join(thread_t th, void **thread_return)
{
	return pthread_join(th, thread_return);
}

int  thread_mutex_init(thread_mutex_t  *mutex,  const  thread_mutexattr_t *mutexattr)
{
	return pthread_mutex_init(mutex, mutexattr);
}

int thread_mutex_lock(thread_mutex_t *mutex)
{
	return pthread_mutex_lock(mutex);
}

int thread_mutex_unlock(thread_mutex_t *mutex)
{
	return pthread_mutex_unlock(mutex);
}


#endif // USE_MARCEL 
