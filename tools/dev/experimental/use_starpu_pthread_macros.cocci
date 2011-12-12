@pthread_mutex_@
expression E1, E2;
@@
(
- pthread_mutex_init(E1, E2);
+ _STARPU_PTHREAD_MUTEX_INIT(E1, E2);
|
- pthread_mutex_lock(E1);
+ _STARPU_PTHREAD_MUTEX_LOCK(E1);
|
- pthread_mutex_unlock(E1);
+ _STARPU_PTHREAD_MUTEX_UNLOCK(E1);
|
- pthread_mutex_destroy(E1);
+ _STARPU_PTHREAD_MUTEX_DESTROY(E1);
)


@pthread_rwlock_@
expression E;
@@
(
- pthread_rwlock_init(E);
+ _STARPU_PTHREAD_RWLOCK_INIT(E);
|
- pthread_rwlock_rdlock(E);
+ _STARPU_PTHREAD_RWLOCK_RDLOCK(E);
|
- pthread_rwlock_wrlock(E);
+ _STARPU_PTHREAD_RWLOCK_WRLOCK(E);
|
- pthread_rwlock_unlock(E);
+ _STARPU_PTHREAD_RWLOCK_UNLOCK(E);
|
- pthread_rwlock_destroy(E);
+ _STARPU_PTHREAD_RWLOCK_DESTROY(E);
)


@pthread_cond_@
expression E1, E2;
@@
(
- pthread_cond_init(E1, E2);
+ _STARPU_PTHREAD_COND_INIT(E1, E2);
|
- pthread_cond_signal(E1);
+ _STARPU_PTHREAD_COND_SIGNAL(E1);
|
- pthread_cond_broadcast(E1);
+ _STARPU_PTHREAD_COND_BROADCAST(E1);
|
- pthread_cond_wait(E1, E2);
+ _STARPU_PTHREAD_COND_WAIT(E1, E2);
|
- pthread_cond_destroy(E1);
+ _STARPU_PTHREAD_COND_DESTROY(E1);
)


@pthread_barrier_@
expression E1, E2, E3;
@@
(
- pthread_barrier_init(E1, E2, E3);
+ _STARPU_PTHREAD_BARRIER_INIT(E1, E2, E3);
|
- pthread_barrier_wait(E1);
+ _STARPU_PTHREAD_BARRIER_WAIT(E1);
|
- pthread_barrier_destroy(E1);
+ _STARPU_PTHREAD_BARRIER_DESTROY(E1);
)
