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

#include <starpu.h>
#include <starpu_profiling.h>
#include <profiling/profiling.h>
#include <core/workers.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <errno.h>

static struct starpu_profiling_worker_info worker_info[STARPU_NMAXWORKERS];
/* TODO: rather use rwlock */
static starpu_pthread_mutex_t worker_info_mutex[STARPU_NMAXWORKERS];

/* In case the worker is still sleeping when the user request profiling info,
 * we need to account for the time elasped while sleeping. */
static unsigned worker_registered_sleeping_start[STARPU_NMAXWORKERS];
static struct timespec sleeping_start_date[STARPU_NMAXWORKERS];

static unsigned worker_registered_executing_start[STARPU_NMAXWORKERS];
static struct timespec executing_start_date[STARPU_NMAXWORKERS];


/* Store the busid of the different (src, dst) pairs. busid_matrix[src][dst]
 * contains the busid of (src, dst) or -1 if the bus was not registered. */
struct node_pair
{
	int src;
	int dst;
	struct starpu_profiling_bus_info *bus_info;
};

static int busid_matrix[STARPU_MAXNODES][STARPU_MAXNODES];
static struct starpu_profiling_bus_info bus_profiling_info[STARPU_MAXNODES][STARPU_MAXNODES];
static struct node_pair busid_to_node_pair[STARPU_MAXNODES*STARPU_MAXNODES];
static char bus_direct[STARPU_MAXNODES*STARPU_MAXNODES];
static int bus_ngpus[STARPU_MAXNODES*STARPU_MAXNODES];
static unsigned busid_cnt = 0;

static void _starpu_bus_reset_profiling_info(struct starpu_profiling_bus_info *bus_info);

/* Clear all the profiling info related to the worker. */
static void _starpu_worker_reset_profiling_info_with_lock(int workerid);

/*
 *	Global control of profiling
 */

/* Disabled by default, unless simulating */
int _starpu_profiling =
#ifdef STARPU_SIMGRID
	1
#else
	0
#endif
	;

void starpu_profiling_init()
{
	_starpu_profiling_init();
}

static void _starpu_profiling_reset_counters()
{
	int worker;
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		_starpu_worker_reset_profiling_info_with_lock(worker);
	}

	int busid;
	int bus_cnt = starpu_bus_get_count();
	for (busid = 0; busid < bus_cnt; busid++)
	{
		struct starpu_profiling_bus_info *bus_info;
		bus_info = busid_to_node_pair[busid].bus_info;
		_starpu_bus_reset_profiling_info(bus_info);
	}
}

int starpu_profiling_status_set(int status)
{
	int worker;
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[worker]);
	}

	ANNOTATE_HAPPENS_AFTER(&_starpu_profiling);
	int prev_value = _starpu_profiling;
	_starpu_profiling = status;
	ANNOTATE_HAPPENS_BEFORE(&_starpu_profiling);

	_STARPU_TRACE_SET_PROFILING(status);

	/* If we enable profiling, we reset the counters. */
	if (status == STARPU_PROFILING_ENABLE)
	{
		_starpu_profiling_reset_counters();
	}

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[worker]);
	}

	return prev_value;
}

void _starpu_profiling_init(void)
{
	int worker;

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		STARPU_PTHREAD_MUTEX_INIT(&worker_info_mutex[worker], NULL);
	}
}

void _starpu_profiling_start(void)
{
	const char *env;
	if ((env = starpu_getenv("STARPU_PROFILING")) && atoi(env))
	{
		starpu_profiling_status_set(STARPU_PROFILING_ENABLE);
	}
}

void _starpu_profiling_terminate(void)
{
	int worker;

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&worker_info_mutex[worker]);
	}
}

/*
 *	Task profiling
 */

struct starpu_profiling_task_info *_starpu_allocate_profiling_info_if_needed(struct starpu_task *task)
{
	struct starpu_profiling_task_info *info = NULL;

	/* If we are benchmarking, we need room for the energy */
	if (starpu_profiling_status_get() || (task->cl && task->cl->energy_model && (task->cl->energy_model->benchmarking || _starpu_get_calibrate_flag())))
	{
		_STARPU_CALLOC(info, 1, sizeof(struct starpu_profiling_task_info));
	}

	return info;
}

/*
 *	Worker profiling
 */

static void _starpu_worker_reset_profiling_info_with_lock(int workerid)
{
	_starpu_clock_gettime(&worker_info[workerid].start_time);

	/* This is computed in a lazy fashion when the application queries
	 * profiling info. */
	starpu_timespec_clear(&worker_info[workerid].total_time);

	starpu_timespec_clear(&worker_info[workerid].executing_time);
	starpu_timespec_clear(&worker_info[workerid].sleeping_time);

	worker_info[workerid].executed_tasks = 0;

	worker_info[workerid].used_cycles = 0;
	worker_info[workerid].stall_cycles = 0;
	worker_info[workerid].energy_consumed = 0;
	worker_info[workerid].flops = 0;

	/* We detect if the worker is already sleeping or doing some
	 * computation */
	enum _starpu_worker_status status = _starpu_worker_get_status(workerid);

	if (status == STATUS_SLEEPING || status == STATUS_SLEEPING_SCHEDULING)
	{
		worker_registered_sleeping_start[workerid] = 1;
		_starpu_clock_gettime(&sleeping_start_date[workerid]);
	}
	else
	{
		worker_registered_sleeping_start[workerid] = 0;
	}

	if (status == STATUS_EXECUTING)
	{
		worker_registered_executing_start[workerid] = 1;
		_starpu_clock_gettime(&executing_start_date[workerid]);
	}
	else
	{
		worker_registered_executing_start[workerid] = 0;
	}
}

void _starpu_worker_restart_sleeping(int workerid)
{
	if (starpu_profiling_status_get())
	{
		struct timespec sleep_start_time;
		_starpu_clock_gettime(&sleep_start_time);

		STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);
		if (worker_registered_sleeping_start[workerid] == 0)
		{
			worker_registered_sleeping_start[workerid] = 1;
			memcpy(&sleeping_start_date[workerid], &sleep_start_time, sizeof(struct timespec));
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
	}
}

void _starpu_worker_stop_sleeping(int workerid)
{
	if (starpu_profiling_status_get())
	{
		struct timespec *sleeping_start, sleep_end_time;

		_starpu_clock_gettime(&sleep_end_time);

		STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);

		if (worker_registered_sleeping_start[workerid] == 1)
		{
			sleeping_start = &sleeping_start_date[workerid];

			/* Perhaps that profiling was enabled while the worker was
			 * already blocked, so we don't measure (end - start), but
			 * (end - max(start,worker_start)) where worker_start is the
			 * date of the previous profiling info reset on the worker */
			struct timespec *worker_start = &worker_info[workerid].start_time;
			if (starpu_timespec_cmp(sleeping_start, worker_start, <))
			{
				/* sleeping_start < worker_start */
				sleeping_start = worker_start;
			}

			struct timespec sleeping_time;
			starpu_timespec_sub(&sleep_end_time, sleeping_start, &sleeping_time);

			starpu_timespec_accumulate(&worker_info[workerid].sleeping_time, &sleeping_time);

			worker_registered_sleeping_start[workerid] = 0;
		}

		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);

	}
}

void _starpu_worker_register_executing_start_date(int workerid, struct timespec *executing_start)
{
	if (starpu_profiling_status_get())
	{
		STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);
		worker_registered_executing_start[workerid] = 1;
		memcpy(&executing_start_date[workerid], executing_start, sizeof(struct timespec));
		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
	}
}

void _starpu_worker_register_executing_end(int workerid)
{
	if (starpu_profiling_status_get())
	{
		STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);
		worker_registered_executing_start[workerid] = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
	}
}


void _starpu_worker_update_profiling_info_executing(int workerid, struct timespec *executing_time, int executed_tasks, uint64_t used_cycles, uint64_t stall_cycles, double energy_consumed, double flops)
{
	if (starpu_profiling_status_get())
	{
		STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);

		if (executing_time)
			starpu_timespec_accumulate(&worker_info[workerid].executing_time, executing_time);

		worker_info[workerid].used_cycles += used_cycles;
		worker_info[workerid].stall_cycles += stall_cycles;
		worker_info[workerid].energy_consumed += energy_consumed;
		worker_info[workerid].executed_tasks += executed_tasks;
		worker_info[workerid].flops += flops;

		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
	}
	else /* Not thread safe, shouldn't be too much a problem */
		worker_info[workerid].executed_tasks += executed_tasks;
}

int starpu_profiling_worker_get_info(int workerid, struct starpu_profiling_worker_info *info)
{
	if (!starpu_profiling_status_get())
	{
		/* Not thread safe, shouldn't be too much a problem */
		info->executed_tasks = worker_info[workerid].executed_tasks;
	}

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_get_worker_struct(workerid)->sched_mutex);
	STARPU_PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);

	if (info)
	{
		/* The total time is computed in a lazy fashion */
		struct timespec now;
		_starpu_clock_gettime(&now);

		/* In case some worker is currently sleeping, we take into
		 * account the time spent since it registered. */
		if (worker_registered_sleeping_start[workerid])
		{
			struct timespec sleeping_time;
			starpu_timespec_sub(&now, &sleeping_start_date[workerid], &sleeping_time);
			starpu_timespec_accumulate(&worker_info[workerid].sleeping_time, &sleeping_time);
		}

		if (worker_registered_executing_start[workerid])
		{
			struct timespec executing_time;
			starpu_timespec_sub(&now, &executing_start_date[workerid], &executing_time);
			starpu_timespec_accumulate(&worker_info[workerid].executing_time, &executing_time);
		}

		/* total_time = now - start_time */
		starpu_timespec_sub(&now, &worker_info[workerid].start_time,
					&worker_info[workerid].total_time);

		memcpy(info, &worker_info[workerid], sizeof(struct starpu_profiling_worker_info));
	}

	_starpu_worker_reset_profiling_info_with_lock(workerid);

	STARPU_PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_get_worker_struct(workerid)->sched_mutex);

	return 0;
}

/* When did the task reach the scheduler  ? */
void _starpu_profiling_set_task_push_start_time(struct starpu_task *task)
{
	if (!starpu_profiling_status_get())
		return;

	struct starpu_profiling_task_info *profiling_info;
	profiling_info = task->profiling_info;

	if (profiling_info)
		_starpu_clock_gettime(&profiling_info->push_start_time);
}

void _starpu_profiling_set_task_push_end_time(struct starpu_task *task)
{
	if (!starpu_profiling_status_get())
		return;

	struct starpu_profiling_task_info *profiling_info;
	profiling_info = task->profiling_info;

	if (profiling_info)
		_starpu_clock_gettime(&profiling_info->push_end_time);
}

/*
 *	Bus profiling
 */

void _starpu_initialize_busid_matrix(void)
{
	int i, j;
	for (j = 0; j < STARPU_MAXNODES; j++)
	for (i = 0; i < STARPU_MAXNODES; i++)
		busid_matrix[i][j] = -1;

	busid_cnt = 0;
}

static void _starpu_bus_reset_profiling_info(struct starpu_profiling_bus_info *bus_info)
{
	_starpu_clock_gettime(&bus_info->start_time);
	bus_info->transferred_bytes = 0;
	bus_info->transfer_count = 0;
}

int _starpu_register_bus(int src_node, int dst_node)
{
	if (starpu_bus_get_id(src_node, dst_node) != -1)
		return -EBUSY;

	int busid = STARPU_ATOMIC_ADD(&busid_cnt, 1) - 1;

	busid_matrix[src_node][dst_node] = busid;

	busid_to_node_pair[busid].src = src_node;
	busid_to_node_pair[busid].dst = dst_node;
	busid_to_node_pair[busid].bus_info = &bus_profiling_info[src_node][dst_node];

	_starpu_bus_reset_profiling_info(&bus_profiling_info[src_node][dst_node]);

	return busid;
}

int starpu_bus_get_count(void)
{
	return busid_cnt;
}

int starpu_bus_get_id(int src, int dst)
{
	return busid_matrix[src][dst];
}

int starpu_bus_get_src(int busid)
{
	return busid_to_node_pair[busid].src;
}

int starpu_bus_get_dst(int busid)
{
	return busid_to_node_pair[busid].dst;
}

void starpu_bus_set_direct(int busid, int direct)
{
	bus_direct[busid] = direct;
}

int starpu_bus_get_direct(int busid)
{
	return bus_direct[busid];
}

void starpu_bus_set_ngpus(int busid, int ngpus)
{
	bus_ngpus[busid] = ngpus;
}

int starpu_bus_get_ngpus(int busid)
{
	struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;
	int ngpus = bus_ngpus[busid];
	if (!ngpus)
		/* Unknown number of GPUs, assume it's shared by all GPUs */
		ngpus = topology->ncudagpus+topology->nopenclgpus;
	return ngpus;
}

int starpu_bus_get_profiling_info(int busid, struct starpu_profiling_bus_info *bus_info)
{
	int src_node = starpu_bus_get_src(busid);
	int dst_node = starpu_bus_get_dst(busid);

	/* XXX protect all this  method with a mutex */
	if (bus_info)
	{
		struct timespec now;
		_starpu_clock_gettime(&now);

		/* total_time = now - start_time */
		starpu_timespec_sub(&now, &bus_profiling_info[src_node][dst_node].start_time,
					  &bus_profiling_info[src_node][dst_node].total_time);

		memcpy(bus_info, &bus_profiling_info[src_node][dst_node], sizeof(struct starpu_profiling_bus_info));
	}

	_starpu_bus_reset_profiling_info(&bus_profiling_info[src_node][dst_node]);

	return 0;
}

void _starpu_bus_update_profiling_info(int src_node, int dst_node, size_t size)
{
	bus_profiling_info[src_node][dst_node].transferred_bytes += size;
	bus_profiling_info[src_node][dst_node].transfer_count++;
//	fprintf(stderr, "PROFILE %d -> %d : %d (cnt %d)\n", src_node, dst_node, size, bus_profiling_info[src_node][dst_node].transfer_count);
}

#undef starpu_profiling_status_get
int starpu_profiling_status_get(void)
{
	int ret;
	ANNOTATE_HAPPENS_AFTER(&_starpu_profiling);
	ret = _starpu_profiling;
	ANNOTATE_HAPPENS_BEFORE(&_starpu_profiling);
	return ret;
}
