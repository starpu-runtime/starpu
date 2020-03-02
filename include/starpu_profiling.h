/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2020       Federal University of Rio Grande do Sul (UFRGS)
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

/**
   @defgroup API_Profiling Profiling
   @{
*/

/**
   Used when calling the function starpu_profiling_status_set() to disable profiling.
*/
#define STARPU_PROFILING_DISABLE	0
/**
   Used when calling the function starpu_profiling_status_set() to enable profiling.
*/
#define STARPU_PROFILING_ENABLE		1

/**
   Information about the execution of a task. It is accessible from
   the field starpu_task::profiling_info if profiling was enabled.
 */
struct starpu_profiling_task_info
{
	/** Date of task submission (relative to the initialization of StarPU). */
	struct timespec submit_time;

	/** Time when the task was submitted to the scheduler. */
	struct timespec push_start_time;
	/** Time when the scheduler finished with the task submission. */
	struct timespec push_end_time;
	/** Time when the scheduler started to be requested for a task, and eventually gave that task. */
	struct timespec pop_start_time;
	/** Time when the scheduler finished providing the task for execution. */
	struct timespec pop_end_time;

	/** Time when the worker started fetching input data. */
	struct timespec acquire_data_start_time;
	/** Time when the worker finished fetching input data. */
	struct timespec acquire_data_end_time;

	/** Date of task execution beginning (relative to the initialization of StarPU). */
	struct timespec start_time;
	/** Date of task execution termination (relative to the initialization of StarPU). */
	struct timespec end_time;

	/** Time when the worker started releasing data. */
	struct timespec release_data_start_time;
	/** Time when the worker finished releasing data. */
	struct timespec release_data_end_time;

	/** Time when the worker started the application callback for the task. */
	struct timespec callback_start_time;
	/** Time when the worker finished the application callback for the task. */
	struct timespec callback_end_time;

	/* TODO add expected length, expected start/end ? */

	/** Identifier of the worker which has executed the task. */
	int workerid;

	/** Number of cycles used by the task, only available in the MoviSim */
	uint64_t used_cycles;
	/** Number of cycles stalled within the task, only available in the MoviSim */
	uint64_t stall_cycles;
	/** Energy consumed by the task, in Joules */
	double energy_consumed;
};

/**
   Profiling information associated to a worker. The timing is
   provided since the previous call to
   starpu_profiling_worker_get_info()
*/
struct starpu_profiling_worker_info
{
	/** Starting date for the reported profiling measurements. */
	struct timespec start_time;
	/** Duration of the profiling measurement interval. */
	struct timespec total_time;
	/** Time spent by the worker to execute tasks during the profiling measurement interval. */
	struct timespec executing_time;
	/** Time spent idling by the worker during the profiling measurement interval. */
	struct timespec sleeping_time;
	/** Number of tasks executed by the worker during the profiling measurement interval. */
	int executed_tasks;

	/** Number of cycles used by the worker, only available in the MoviSim */
	uint64_t used_cycles;
	/** Number of cycles stalled within the worker, only available in the MoviSim */
	uint64_t stall_cycles;
	/** Energy consumed by the worker, in Joules */
	double energy_consumed;

	double flops;
};

struct starpu_profiling_bus_info
{
	/** Time of bus profiling startup. */
	struct timespec start_time;
	/** Total time of bus profiling. */
	struct timespec total_time;
	/** Number of bytes transferred during profiling. */
	int long long transferred_bytes;
	/** Number of transfers during profiling. */
	int transfer_count;
};

/**
   Reset performance counters and enable profiling if the
   environment variable \ref STARPU_PROFILING is set to a positive value.
*/
void starpu_profiling_init(void);

/**
   Set the ID used for profiling trace filename. Has to be called before starpu_init().
*/
void starpu_profiling_set_id(int new_id);

/**
   Set the profiling status. Profiling is activated
   by passing \ref STARPU_PROFILING_ENABLE in \p status. Passing
   \ref STARPU_PROFILING_DISABLE disables profiling. Calling this function
   resets all profiling measurements. When profiling is enabled, the
   field starpu_task::profiling_info points to a valid structure
   starpu_profiling_task_info containing information about the execution
   of the task. Negative return values indicate an error, otherwise the
   previous status is returned.
*/
int starpu_profiling_status_set(int status);

/**
   Return the current profiling status or a negative value in case
   there was an error.
*/
int starpu_profiling_status_get(void);

#ifdef BUILDING_STARPU
#include <common/utils.h>
#ifdef __GNUC__
extern int _starpu_profiling;
#define starpu_profiling_status_get() ({ \
	int __ret; \
	ANNOTATE_HAPPENS_AFTER(&_starpu_profiling); \
	__ret = _starpu_profiling; \
	ANNOTATE_HAPPENS_BEFORE(&_starpu_profiling); \
	__ret; \
})
#endif
#endif

/**
   Get the profiling info associated to the worker identified by
   \p workerid, and reset the profiling measurements. If the argument \p
   worker_info is <c>NULL</c>, only reset the counters associated to worker
   \p workerid. Upon successful completion, this function returns 0.
   Otherwise, a negative value is returned.
*/
int starpu_profiling_worker_get_info(int workerid, struct starpu_profiling_worker_info *worker_info);

/**
   Return the number of buses in the machine
*/
int starpu_bus_get_count(void);

/**
   Return the identifier of the bus between \p src and \p dst
*/
int starpu_bus_get_id(int src, int dst);

/**
   Return the source point of bus \p busid
*/
int starpu_bus_get_src(int busid);

/**
   Return the destination point of bus \p busid
*/
int starpu_bus_get_dst(int busid);
void starpu_bus_set_direct(int busid, int direct);
int starpu_bus_get_direct(int busid);
void starpu_bus_set_ngpus(int busid, int ngpus);
int starpu_bus_get_ngpus(int busid);

/**
   See _starpu_profiling_bus_helper_display_summary in src/profiling/profiling_helpers.c for a usage example.
   Note that calling starpu_bus_get_profiling_info() resets the counters to zero.
*/
int starpu_bus_get_profiling_info(int busid, struct starpu_profiling_bus_info *bus_info);

/* Some helper functions to manipulate profiling API output */
/* Reset timespec */
static __starpu_inline void starpu_timespec_clear(struct timespec *tsp)
{
	tsp->tv_sec = 0;
	tsp->tv_nsec = 0;
}

#define STARPU_NS_PER_S 1000000000

/* Computes result = a + b */
static __starpu_inline void starpu_timespec_add(struct timespec *a,
						struct timespec *b,
						struct timespec *result)
{
	result->tv_sec = a->tv_sec + b->tv_sec;
	result->tv_nsec = a->tv_nsec + b->tv_nsec;

	if (result->tv_nsec >= STARPU_NS_PER_S)
	{
		++(result)->tv_sec;
		result->tv_nsec -= STARPU_NS_PER_S;
	}
}

/* Computes res += b */
static __starpu_inline void starpu_timespec_accumulate(struct timespec *result,
						       struct timespec *a)
{
	result->tv_sec += a->tv_sec;
	result->tv_nsec += a->tv_nsec;

	if (result->tv_nsec >= STARPU_NS_PER_S)
	{
		++(result)->tv_sec;
		result->tv_nsec -= STARPU_NS_PER_S;
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
		result->tv_nsec += STARPU_NS_PER_S;
	}
}

#define starpu_timespec_cmp(a, b, CMP)                          \
	(((a)->tv_sec == (b)->tv_sec) ? ((a)->tv_nsec CMP (b)->tv_nsec) : ((a)->tv_sec CMP (b)->tv_sec))

/**
   Return the time elapsed between \p start and \p end in microseconds.
*/
double starpu_timing_timespec_delay_us(struct timespec *start, struct timespec *end);

/**
   Convert the given timespec \p ts into microseconds
*/
double starpu_timing_timespec_to_us(struct timespec *ts);

/**
   Display statistics about the bus on \c stderr. if the environment
   variable \ref STARPU_BUS_STATS is defined. The function is called
   automatically by starpu_shutdown().
*/
void starpu_profiling_bus_helper_display_summary(void);

/**
   Display statistic about the workers on \c stderr if the
   environment variable \ref STARPU_WORKER_STATS is defined. The function is
   called automatically by starpu_shutdown().
*/
void starpu_profiling_worker_helper_display_summary(void);

/**
   Display statistics about the current data handles registered
   within StarPU. StarPU must have been configured with the configure
   option \ref enable-memory-stats "--enable-memory-stats" (see \ref
   MemoryFeedback).
*/
void starpu_data_display_memory_stats();

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PROFILING_H__ */
