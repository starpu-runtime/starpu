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

#include "timing.h"

static double reference_start_time;

#ifdef STARPU_HAVE_CLOCK_GETTIME

#define TICK_DIFF(t1, t2) ((long long)((t2).ts.tv_sec*1e9 + (t2).ts.tv_nsec) + \
				- (long long)((t1).ts.tv_sec*1e9) + (long long)(t1).ts.tv_nsec)
#define TIMING_DELAY(t1, t2) _starpu_tick2usec(TICK_DIFF((t1), (t2)))

void _starpu_timing_init(void)
{
	reference_start_time = _starpu_timing_now();
}

inline double _starpu_tick2usec(long long t)
{
  return (double)(t)/1000;
}

inline double _starpu_timing_delay(starpu_tick_t *t1, starpu_tick_t *t2)
{
	double d1, d2;

	d1 = _starpu_tick2usec((t1->ts.tv_sec*1e9) + t1->ts.tv_nsec);
	d2 = _starpu_tick2usec((t2->ts.tv_sec*1e9) + t2->ts.tv_nsec);

	return (d2 - d1);;
}

/* returns the current time in us */
inline double _starpu_timing_now(void)
{
	starpu_tick_t tick_now;
	STARPU_GET_TICK(tick_now);

	double absolute_now = _starpu_tick2usec(((tick_now).ts.tv_sec*1e9) + (tick_now).ts.tv_nsec);

	return (absolute_now - reference_start_time);
}

#else // STARPU_HAVE_CLOCK_GETTIME

#define TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define TICK_DIFF(t1, t2) (TICK_RAW_DIFF(t1, t2) - residual)
#define TIMING_DELAY(t1, t2) _starpu_tick2usec(TICK_DIFF(t1, t2))

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

  reference_start_time = _starpu_timing_now();

  inited = 1;
}

inline double _starpu_tick2usec(long long t)
{
  return (double)(t)*scale;
}

inline double _starpu_timing_delay(starpu_tick_t *t1, starpu_tick_t *t2)
{
	return TIMING_DELAY(*t1, *t2);
}

inline double _starpu_timing_now(void)
{
	starpu_tick_t tick_now;
	STARPU_GET_TICK(tick_now);

	double absolute_now =  _starpu_tick2usec(tick_now.tick);

	return (absolute_now - reference_start_time);

}

#endif // STARPU_HAVE_CLOCK_GETTIME
