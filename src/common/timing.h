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

#ifndef TIMING_H
#define TIMING_H

/*
 * -- Initialiser la bibliothèque avec timing_init();
 * -- Mémoriser un timestamp :
 *  tick_t t;
 *  GET_TICK(t);
 * -- Calculer un intervalle en microsecondes :
 *  TIMING_DELAY(t1, t2);
 */

#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>
#include <common/config.h>
#include <starpu.h>

#ifdef USE_SYNC_CLOCK
#include <time.h>
#ifndef _POSIX_C_SOURCE
/* for clock_gettime */
#define _POSIX_C_SOURCE 199309L
#endif

/* we use the usual gettimeofday method */
typedef struct tick_s
{
	struct timespec ts;
} tick_t;
#define GET_TICK(t) clock_gettime(CLOCK_MONOTONIC, &((t).ts))

#else // !USE_SYNC_CLOCK

typedef union u_tick
{
  uint64_t tick;

  struct
  {
    uint32_t low;
    uint32_t high;
  }
  sub;
} tick_t;

#if defined(__i386__) || defined(__pentium__) || defined(__pentiumpro__) || defined(__i586__) || defined(__i686__) || defined(__k6__) || defined(__k7__) || defined(__x86_64__)
#  define GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#else
//#  error "Processeur non-supporté par timing.h"
/* XXX */
//#warning "unsupported processor GET_TICK returns 0"
#  define GET_TICK(t) do {} while(0);
#endif

#endif // USE_SYNC_CLOCK

void __attribute__ ((unused)) timing_init(void);
inline double __attribute__ ((unused)) _starpu_tick2usec(long long t);
inline double __attribute__ ((unused)) timing_delay(tick_t *t1, tick_t *t2);

inline double __attribute__ ((unused)) timing_now(void);

#endif /* TIMING_H */


