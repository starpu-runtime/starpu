/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <sys/time.h>
#include <starpu.h>
#include <common/config.h>
#include <profiling/profiling.h>
#include <common/timing.h>

#ifdef HAVE_CLOCK_GETTIME
#include <time.h>
#ifndef _POSIX_C_SOURCE
/* for clock_gettime */
#define _POSIX_C_SOURCE 199309L
#endif

#ifdef __linux__
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#endif

static struct timespec reference_start_time_ts;

/* Modern CPUs' clocks are usually not synchronized so we use a monotonic clock
 * to have consistent timing measurements. The CLOCK_MONOTONIC_RAW clock is not
 * subject to NTP adjustments, but is not available on all systems (in that
 * case we use the CLOCK_MONOTONIC clock instead). */
void __starpu_clock_gettime(struct timespec *ts) {
#ifdef CLOCK_MONOTONIC_RAW
	static int raw_supported = 0;
	switch (raw_supported) {
	case -1:
		break;
	case 1:
		clock_gettime(CLOCK_MONOTONIC_RAW, ts);
		return;
	case 0:
		if (clock_gettime(CLOCK_MONOTONIC_RAW, ts)) {
			raw_supported = -1;
			break;
		} else {
			raw_supported = 1;
			return;
		}
	}
#endif
	clock_gettime(CLOCK_MONOTONIC, ts);
}

void _starpu_timing_init(void)
{
	__starpu_clock_gettime(&reference_start_time_ts);
}

void starpu_clock_gettime(struct timespec *ts)
{
	struct timespec absolute_ts;

	/* Read the current time */
	__starpu_clock_gettime(&absolute_ts);

	/* Compute the relative time since initialization */
	starpu_timespec_sub(&absolute_ts, &reference_start_time_ts, ts);
}

#else // !HAVE_CLOCK_GETTIME

#if defined(__i386__) || defined(__pentium__) || defined(__pentiumpro__) || defined(__i586__) || defined(__i686__) || defined(__k6__) || defined(__k7__) || defined(__x86_64__)
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

#define STARPU_GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#define TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define TICK_DIFF(t1, t2) (TICK_RAW_DIFF(t1, t2) - residual)

static starpu_tick_t reference_start_tick;
static double scale = 0.0;
static unsigned long long residual = 0;

static int inited = 0;

void _starpu_timing_init(void)
{
  static starpu_tick_t t1, t2;
  int i;

  if (inited) return;

  residual = (unsigned long long)1 << 63;
  
  for(i = 0; i < 20; i++)
    {
      STARPU_GET_TICK(t1);
      STARPU_GET_TICK(t2);
      residual = STARPU_MIN(residual, TICK_RAW_DIFF(t1, t2));
    }
  
  {
    struct timeval tv1,tv2;
    
    STARPU_GET_TICK(t1);
    gettimeofday(&tv1,0);
    usleep(500000);
    STARPU_GET_TICK(t2);
    gettimeofday(&tv2,0);
    scale = ((tv2.tv_sec*1e6 + tv2.tv_usec) -
	     (tv1.tv_sec*1e6 + tv1.tv_usec)) / 
      (double)(TICK_DIFF(t1, t2));
  }

  STARPU_GET_TICK(reference_start_tick);

  inited = 1;
}

void starpu_clock_gettime(struct timespec *ts)
{
	starpu_tick_t tick_now;

	STARPU_GET_TICK(tick_now);

	uint64_t elapsed_ticks = TICK_DIFF(reference_start_tick, tick_now);

	/* We convert this number into nano-seconds so that we can fill the
	 * timespec structure. */
	uint64_t elapsed_ns = (uint64_t)(((double)elapsed_ticks)*(scale*1000.0));
	
	long tv_nsec = (elapsed_ns % 1000000000);
	time_t tv_sec = (elapsed_ns / 1000000000);

	ts->tv_sec = tv_sec;
	ts->tv_nsec = tv_nsec;
}

#else // !HAVE_CLOCK_GETTIME & no rdtsc
#warning StarPU could not find a timer, clock will always return 0
void _starpu_timing_init(void)
{
}

void starpu_clock_gettime(struct timespec *ts)
{
	timerclear(ts);
}
#endif
#endif // HAVE_CLOCK_GETTIME

/* Returns the time elapsed between start and end in microseconds */
double starpu_timing_timespec_delay_us(struct timespec *start, struct timespec *end)
{
	struct timespec diff;
	
	starpu_timespec_sub(end, start, &diff);

	double us = (diff.tv_sec*1e6) + (diff.tv_nsec*1e-3);

	return us;
}

double starpu_timing_timespec_to_us(struct timespec *ts)
{
	return (1000000.0*ts->tv_sec) + (0.001*ts->tv_nsec);
}

double _starpu_timing_now(void)
{
	struct timespec now;
	starpu_clock_gettime(&now);

	return starpu_timing_timespec_to_us(&now);
}
