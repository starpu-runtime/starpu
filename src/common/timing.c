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

#ifdef UNRELIABLETICKS

#define TICK_RAW_DIFF(t1, t2) (((t2).ts.tv_sec*1e9 + (t2).ts.tv_nsec) + \
				- ((t1).ts.tv_sec*1e9) + (t1).ts.tv_nsec)
#define TICK_DIFF(t1, t2) (TICK_RAW_DIFF(t1, t2))
#define TIMING_DELAY(t1, t2) tick2usec(TICK_DIFF(t1, t2))

static double scale = 0;
static unsigned long long residual = 0;

void timing_init(void)
{
}

inline double tick2usec(long long t)
{
  return (double)(t)/1000;
}

inline double timing_delay(tick_t *t1, tick_t *t2)
{
	return TIMING_DELAY(*t1, *t2);
}

/* returns the current time in us */
inline double timing_now(void)
{
	tick_t tick_now;
	GET_TICK(tick_now);

	return tick2usec(((tick_now).ts.tv_sec*1e9) + (tick_now).ts.tv_usec*1e3);
}



#else // UNRELIABLETICKS

#define TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define TICK_DIFF(t1, t2) (TICK_RAW_DIFF(t1, t2) - residual)
#define TIMING_DELAY(t1, t2) tick2usec(TICK_DIFF(t1, t2))

static double scale = 0.0;
static unsigned long long residual = 0;

static int inited = 0;

void timing_init(void)
{
  static tick_t t1, t2;
  int i;

  if (inited) return;

  residual = (unsigned long long)1 << 63;
  
  for(i = 0; i < 20; i++)
    {
      GET_TICK(t1);
      GET_TICK(t2);
      residual = STARPU_MIN(residual, TICK_RAW_DIFF(t1, t2));
    }
  
  {
    struct timeval tv1,tv2;
    
    GET_TICK(t1);
    gettimeofday(&tv1,0);
    usleep(500000);
    GET_TICK(t2);
    gettimeofday(&tv2,0);
    scale = ((tv2.tv_sec*1e6 + tv2.tv_usec) -
	     (tv1.tv_sec*1e6 + tv1.tv_usec)) / 
      (double)(TICK_DIFF(t1, t2));
  }

  inited = 1;
}

inline double tick2usec(long long t)
{
  return (double)(t)*scale;
}

inline double timing_delay(tick_t *t1, tick_t *t2)
{
	return TIMING_DELAY(*t1, *t2);
}

inline double timing_now(void)
{
	tick_t tick_now;
	GET_TICK(tick_now);

	return tick2usec(tick_now.tick);
}

#endif // UNRELIABLETICKS
