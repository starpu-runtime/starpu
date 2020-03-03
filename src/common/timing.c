/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/config.h>
#include <starpu_util.h>
#include <profiling/profiling.h>
#include <common/timing.h>
#include <math.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#ifdef HAVE_SIMGRID_ENGINE_H
#include <simgrid/engine.h>
#endif
#endif

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#ifdef STARPU_SIMGRID
void _starpu_timing_init(void)
{
}

void _starpu_clock_gettime(struct timespec *ts)
{
#ifdef HAVE_SIMGRID_GET_CLOCK
	double now = simgrid_get_clock();
#else
	double now = MSG_get_clock();
#endif
	ts->tv_sec = floor(now);
	ts->tv_nsec = floor((now - ts->tv_sec) * 1000000000);
}

#elif defined(HAVE_CLOCK_GETTIME) && defined(CLOCK_MONOTONIC)
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

static struct timespec _starpu_reference_start_time_ts;

/* Modern CPUs' clocks are usually not synchronized so we use a monotonic clock
 * to have consistent timing measurements.
 */
static void _starpu_clock_readtime(struct timespec *ts)
{
#if 0 /* def CLOCK_MONOTONIC_RAW */
/* The CLOCK_MONOTONIC_RAW clock is not
 * subject to NTP adjustments, but is not available on all systems (in that
 * case we use the CLOCK_MONOTONIC clock instead). */

/* In the distributed case, we *do* want NTP adjustments, to get
 * somehow-coherent traces, so this is disabled */

	static int raw_supported = 0;
	switch (raw_supported)
	{
	case -1:
		break;
	case 1:
		clock_gettime(CLOCK_MONOTONIC_RAW, ts);
		return;
	case 0:
		if (clock_gettime(CLOCK_MONOTONIC_RAW, ts))
		{
			raw_supported = -1;
			break;
		}
		else
		{
			raw_supported = 1;
			return;
		}
	}
#endif
	clock_gettime(CLOCK_MONOTONIC, ts);
}

void _starpu_timing_init(void)
{
	_starpu_clock_gettime(&_starpu_reference_start_time_ts);
}

void _starpu_clock_gettime(struct timespec *ts)
{
	struct timespec absolute_ts;

	/* Read the current time */
	_starpu_clock_readtime(&absolute_ts);

	/* Compute the relative time since initialization */
	starpu_timespec_sub(&absolute_ts, &_starpu_reference_start_time_ts, ts);
}

#else // !HAVE_CLOCK_GETTIME

#if defined(__i386__) || defined(__pentium__) || defined(__pentiumpro__) || defined(__i586__) || defined(__i686__) || defined(__k6__) || defined(__k7__) || defined(__x86_64__)
union starpu_u_tick
{
	uint64_t tick;

	struct
	{
		uint32_t low;
		uint32_t high;
	}
		sub;
};

#define STARPU_GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#define STARPU_TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define STARPU_TICK_DIFF(t1, t2) (STARPU_TICK_RAW_DIFF(t1, t2) - _starpu_residual)

static union starpu_u_tick _starpu_reference_start_tick;
static double _starpu_scale = 0.0;
static unsigned long long _starpu_residual = 0;

static int _starpu_inited = 0;

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
static int mygettimeofday(struct timeval *tv, void *tz)
{
	if (tv)
	{
		FILETIME ft;
		unsigned long long res;
		GetSystemTimeAsFileTime(&ft);
		/* 100-nanosecond intervals since January 1, 1601 */
		res = ft.dwHighDateTime;
		res <<= 32;
		res |= ft.dwLowDateTime;
		res /= 10;
		/* Now we have microseconds */
		res -= (((1970-1601)*365) + 89) * 24ULL * 3600ULL * 1000000ULL;
		/* Now we are based on epoch */
		tv->tv_sec = res / 1000000ULL;
		tv->tv_usec = res % 1000000ULL;
	}
}
#else
#define mygettimeofday(tv,tz) gettimeofday(tv,tz)
#endif

void _starpu_timing_init(void)
{
	static union starpu_u_tick t1, t2;
	int i;

	if (_starpu_inited) return;

	_starpu_residual = (unsigned long long)1 << 63;

	for(i = 0; i < 20; i++)
	{
		STARPU_GET_TICK(t1);
		STARPU_GET_TICK(t2);
		_starpu_residual = STARPU_MIN(_starpu_residual, STARPU_TICK_RAW_DIFF(t1, t2));
	}

	{
		struct timeval tv1,tv2;

		STARPU_GET_TICK(t1);
		mygettimeofday(&tv1,0);
		starpu_sleep(0.5);
		STARPU_GET_TICK(t2);
		mygettimeofday(&tv2,0);
		_starpu_scale = ((tv2.tv_sec*1e6 + tv2.tv_usec) -
				 (tv1.tv_sec*1e6 + tv1.tv_usec)) /
			(double)(STARPU_TICK_DIFF(t1, t2));
	}

	STARPU_GET_TICK(_starpu_reference_start_tick);

	_starpu_inited = 1;
}

void _starpu_clock_gettime(struct timespec *ts)
{
	union starpu_u_tick tick_now;

	STARPU_GET_TICK(tick_now);

	uint64_t elapsed_ticks = STARPU_TICK_DIFF(_starpu_reference_start_tick, tick_now);

	/* We convert this number into nano-seconds so that we can fill the
	 * timespec structure. */
	uint64_t elapsed_ns = (uint64_t)(((double)elapsed_ticks)*(_starpu_scale*1000.0));

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

void _starpu_clock_gettime(struct timespec *ts)
{
	ts->tv_sec = 0;
	ts->tv_nsec = 0;
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

double starpu_timing_now(void)
{
#ifdef STARPU_SIMGRID
#  ifdef HAVE_SIMGRID_GET_CLOCK
	return simgrid_get_clock()*1000000;
#  else
	return MSG_get_clock()*1000000;
#  endif
#else
	struct timespec now;
	_starpu_clock_gettime(&now);

	return starpu_timing_timespec_to_us(&now);
#endif
}
