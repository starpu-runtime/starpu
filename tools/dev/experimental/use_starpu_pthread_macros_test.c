/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
static void
foo(void)
{
	pthread_create(&th, NULL, f, &arg);

	pthread_mutex_init(&mutex, NULL);
	pthread_mutex_lock(&mutex);
	pthread_mutex_unlock(&mutex);
	pthread_mutex_destroy(&mutex);

	pthread_rwlock_init(&rwlock);
	pthread_rwlock_rdlock(&rwlock);
	pthread_rwlock_wrlock(&rwlock);
	pthread_rwlock_unlock(&rwlock);
	pthread_rwlock_destroy(&rwlock);

	pthread_cond_init(&cond, NULL);
	pthread_cond_signal(&cond);
	pthread_cond_broadcast(&cond);
	pthread_cond_wait(&cond, &mutex);
	pthread_cond_destroy(&cond);

	pthread_barrier_init(&barrier, NULL, 42);
	pthread_barrier_wait(&barrier);
	pthread_barrier_destroy(&barrier);
}
