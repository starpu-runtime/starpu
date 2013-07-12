/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2013  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_PROFILING_H__
#define __STARPU_PROFILING_H__

#include <starpu.h>
#include <errno.h>
#include <time.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define STARPU_PROFILING_DISABLE	0
#define STARPU_PROFILING_ENABLE		1

struct starpu_profiling_task_info
{
	struct timespec submit_time;

	struct timespec push_start_time;
	struct timespec push_end_time;
	struct timespec pop_start_time;
	struct timespec pop_end_time;

	struct timespec acquire_data_start_time;
	struct timespec acquire_data_end_time;

	struct timespec start_time;
	struct timespec end_time;

	struct timespec release_data_start_time;
	struct timespec release_data_end_time;

	struct timespec callback_start_time;
	struct timespec callback_end_time;

	/* TODO add expected length, expected start/end ? */
	int workerid;

	uint64_t used_cycles;
	uint64_t stall_cycles;
	double power_consumed;
};

struct starpu_profiling_worker_info
{
	struct timespec start_time;
	struct timespec total_time;
	struct timespec executing_time;
	struct timespec sleeping_time;
	int executed_tasks;

	uint64_t used_cycles;
	uint64_t stall_cycles;
	double power_consumed;
};

struct starpu_profiling_bus_info
{
	struct timespec start_time;
	struct timespec total_time;
	int long long transferred_bytes;
	int transfer_count;
};

void starpu_profiling_init();
void starpu_profiling_set_id(int new_id);
int starpu_profiling_status_set(int status);
int starpu_profiling_status_get(void);

#ifdef BUILDING_STARPU
#include <common/utils.h>
extern int _starpu_profiling;
#define starpu_profiling_status_get() ({ \
	int __ret; \
	ANNOTATE_HAPPENS_AFTER(&_starpu_profiling); \
	__ret = _starpu_profiling; \
	ANNOTATE_HAPPENS_BEFORE(&_starpu_profiling); \
	__ret; \
})
#endif

int starpu_profiling_worker_get_info(int workerid, struct starpu_profiling_worker_info *worker_info);

int starpu_bus_get_count(void);
int starpu_bus_get_id(int src, int dst);
int starpu_bus_get_src(int busid);
int starpu_bus_get_dst(int busid);

int starpu_bus_get_profiling_info(int busid, struct starpu_profiling_bus_info *bus_info);

/* Some helper functions to manipulate profiling API output */
/* Reset timespec */
static __starpu_inline void starpu_timespec_clear(struct timespec *tsp)
{
	tsp->tv_sec = 0;
	tsp->tv_nsec = 0;
}

/* Computes result = a + b */
static __starpu_inline void starpu_timespec_add(struct timespec *a,
						struct timespec *b,
						struct timespec *result)
{
	result->tv_sec = a->tv_sec + b->tv_sec;
	result->tv_nsec = a->tv_nsec + b->tv_nsec;

	if (result->tv_nsec >= 1000000000)
	{
		++(result)->tv_sec;
		result->tv_nsec -= 1000000000;
	}
}

/* Computes res += b */
static __starpu_inline void starpu_timespec_accumulate(struct timespec *result,
						       struct timespec *a)
{
	result->tv_sec += a->tv_sec;
	result->tv_nsec += a->tv_nsec;

	if (result->tv_nsec >= 1000000000)
	{
		++(result)->tv_sec;
		result->tv_nsec -= 1000000000;
	}
}

/* Computes result = a - b */
static __starpu_inline void starpu_timespec_sub(const struct timespec *a,
						const struct timespec *b,
						struct timespec *result)
{
	result->tv_sec = a->tv_sec - b->tv_sec;
	result->tv_nsec = a->tv_nsec - b->tv_nsec;

	if ((result)->tv_nsec < 0)
	{
		--(result)->tv_sec;
		result->tv_nsec += 1000000000;
	}
}

#define starpu_timespec_cmp(a, b, CMP)                          \
	(((a)->tv_sec == (b)->tv_sec) ? ((a)->tv_nsec CMP (b)->tv_nsec) : ((a)->tv_sec CMP (b)->tv_sec))

double starpu_timing_timespec_delay_us(struct timespec *start, struct timespec *end);
double starpu_timing_timespec_to_us(struct timespec *ts);

void starpu_profiling_bus_helper_display_summary(void);
void starpu_profiling_worker_helper_display_summary(void);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PROFILING_H__ */
