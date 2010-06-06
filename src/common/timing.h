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
 * -- Initialiser la bibliothèque avec _starpu_timing_init();
 * -- Mémoriser un timestamp :
 *  starpu_tick_t t;
 *  STARPU_GET_TICK(t);
 * -- Calculer un intervalle en microsecondes :
 *  TIMING_DELAY(t1, t2);
 */

#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>
#include <common/config.h>
#include <starpu.h>

#ifdef HAVE_CLOCK_GETTIME
#include <time.h>
#ifndef _POSIX_C_SOURCE
/* for clock_gettime */
#define _POSIX_C_SOURCE 199309L
#endif

/* we use the usual gettimeofday method */
typedef struct starpu_tick_s
{
	struct timespec ts;
} starpu_tick_t;

#ifdef __linux__
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#endif
/* Modern CPUs' clocks are usually not synchronized so we use a monotonic clock
 * to have consistent timing measurements. The CLOCK_MONOTONIC_RAW clock is not
 * subject to NTP adjustments, but is not available on all systems (in that
 * case we use the CLOCK_MONOTONIC clock instead). */
static inline void starpu_get_tick(starpu_tick_t *t) {
#ifdef CLOCK_MONOTONIC_RAW
	static int raw_supported = 0;
	switch (raw_supported) {
	case -1:
		break;
	case 1:
		clock_gettime(CLOCK_MONOTONIC_RAW, &t->ts);
		return;
	case 0:
		if (clock_gettime(CLOCK_MONOTONIC_RAW, &t->ts)) {
			raw_supported = -1;
			break;
		} else {
			raw_supported = 1;
			return;
		}
	}
#endif
	clock_gettime(CLOCK_MONOTONIC, &t->ts);
}
#define STARPU_GET_TICK(t) starpu_get_tick(&(t))

#else // !HAVE_CLOCK_GETTIME

typedef union starpu_u_tick
{
  uint64_t tick;

  struct
  {
    uint32_t low;
    uint32_t high;
  }
  sub;
} starpu_tick_t;

#if defined(__i386__) || defined(__pentium__) || defined(__pentiumpro__) || defined(__i586__) || defined(__i686__) || defined(__k6__) || defined(__k7__) || defined(__x86_64__)
#  define STARPU_GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#else
//#  error "Processeur non-supporté par timing.h"
/* XXX */
//#warning "unsupported processor STARPU_GET_TICK returns 0"
#  define STARPU_GET_TICK(t) do {} while(0);
#endif

#endif // HAVE_CLOCK_GETTIME

void __attribute__ ((unused)) _starpu_timing_init(void);
inline double __attribute__ ((unused)) _starpu_tick2usec(long long t);
inline double __attribute__ ((unused)) _starpu_timing_delay(starpu_tick_t *t1, starpu_tick_t *t2);

inline double __attribute__ ((unused)) _starpu_timing_now(void);

#endif /* TIMING_H */


