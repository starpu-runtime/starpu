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

int _starpu_mkpath(const char *s, mode_t mode);
int _starpu_check_mutex_deadlock(pthread_mutex_t *mutex);

#define PTHREAD_MUTEX_LOCK(mutex) { int ret = pthread_mutex_lock(mutex); if (STARPU_UNLIKELY(ret)) { perror("pthread_mutex_lock"); STARPU_ABORT(); }}
#define PTHREAD_MUTEX_UNLOCK(mutex) { int ret = pthread_mutex_unlock(mutex); if (STARPU_UNLIKELY(ret)) { perror("pthread_mutex_unlock"); STARPU_ABORT(); }}

#endif // __COMMON_UTILS_H__
