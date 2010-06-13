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
 * _starpu_timing_init must be called prior to using any of these timing
 * functions.
 */

#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>
#include <common/config.h>
#include <starpu.h>

/* Computes result = a + b */
#define _starpu_timespec_add(a, b, result)				\
	do {								\
		(result)->tv_sec = (a)->tv_sec + (b)->tv_sec;		\
   		(result)->tv_nsec = (a)->tv_nsec + (b)->tv_nsec; 	\
		if ((result)->tv_nsec >= 1000000000)			\
		{							\
			++(result)->tv_sec;				\
			(result)->tv_nsec -= 1000000000;		\
		}							\
	} while (0)


/* Computes result = a - b */
#define _starpu_timespec_sub(a, b, result)				\
	do {								\
		(result)->tv_sec = (a)->tv_sec - (b)->tv_sec;		\
   		(result)->tv_nsec = (a)->tv_nsec - (b)->tv_nsec; 	\
		if ((result)->tv_nsec < 0)				\
		{							\
			--(result)->tv_sec;				\
			(result)->tv_nsec += 1000000000;		\
		}							\
	} while (0)

void _starpu_timing_init(void);
void starpu_clock_gettime(struct timespec *ts);
double _starpu_timing_now(void);

/* Returns the time elapsed between start and end in microseconds */
double _starpu_timing_timespec_delay_us(struct timespec *start, struct timespec *end);
double _starpu_timing_timespec_to_us(struct timespec *ts);

#endif /* TIMING_H */


