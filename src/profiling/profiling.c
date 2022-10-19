/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_PAPI
#include <papi.h>
#endif

#ifdef STARPU_PAPI
static starpu_pthread_mutex_t papi_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static int papi_events[PAPI_MAX_HWCTRS];
static int papi_nevents = 0;
static int warned_component_unavailable = 0;
#endif

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
	unsigned worker;
	for (worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		struct _starpu_worker *worker_struct = _starpu_get_worker_struct(worker);
		STARPU_PTHREAD_MUTEX_LOCK(&worker_struct->sched_mutex);
	}
	for (worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&_starpu_get_worker_struct(worker)->profiling_info_mutex);
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

	for (worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		struct _starpu_worker *worker_struct = _starpu_get_worker_struct(worker);
		STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_get_worker_struct(worker)->profiling_info_mutex);
		STARPU_PTHREAD_MUTEX_UNLOCK(&worker_struct->sched_mutex);
	}

	return prev_value;
}

void _starpu_profiling_init(void)
{
	int workerid;

	for (workerid = 0; workerid < STARPU_NMAXWORKERS; workerid++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		enum _starpu_worker_status_index i;

		memset(&worker->profiling_info, 0, sizeof(worker->profiling_info));
		STARPU_PTHREAD_MUTEX_INIT(&worker->profiling_info_mutex, NULL);

		for (i = 0; i< STATUS_INDEX_NR; i++)
			worker->profiling_registered_start[i] = 0;

		worker->profiling_status = STATUS_UNKNOWN;
	}

#ifdef STARPU_PAPI
	STARPU_PTHREAD_MUTEX_LOCK(&papi_mutex);
	int retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT)
	{
		_STARPU_MSG("Failed init PAPI, error: %s.\n", PAPI_strerror(retval));
	}
	retval = PAPI_thread_init(pthread_self);
	if (retval != PAPI_OK)
	{
		_STARPU_MSG("Failed init PAPI thread, error: %s.\n", PAPI_strerror(retval));
	}

	char *conf_papi_events;
	char *papi_event_name;
	conf_papi_events = starpu_getenv("STARPU_PROF_PAPI_EVENTS");
	papi_nevents = 0;
	if (conf_papi_events != NULL)
	{
		while ((papi_event_name = strtok_r(conf_papi_events, " ,", &conf_papi_events)))
		{
			if (papi_nevents == PAPI_MAX_HWCTRS)
			{
				_STARPU_MSG("Too many requested papi counters, ignoring %s\n", papi_event_name);
				continue;
			}

			_STARPU_DEBUG("Loading PAPI Event: %s\n", papi_event_name);
			retval = PAPI_event_name_to_code((char*)papi_event_name, &papi_events[papi_nevents]);
			if (retval != PAPI_OK)
				_STARPU_MSG("Failed to codify papi event [%s], error: %s.\n", papi_event_name, PAPI_strerror(retval));
			else
				papi_nevents++;
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&papi_mutex);
#endif
}

#ifdef STARPU_PAPI
void _starpu_profiling_papi_task_start_counters(struct starpu_task *task)
{
	if (!starpu_profiling_status_get())
		return;

	struct starpu_profiling_task_info *profiling_info;
	profiling_info = task->profiling_info;
	if (profiling_info && papi_nevents)
	{
		int i;
		profiling_info->papi_event_set = PAPI_NULL;
		STARPU_PTHREAD_MUTEX_LOCK(&papi_mutex);
		PAPI_create_eventset(&profiling_info->papi_event_set);
		for(i=0; i<papi_nevents; i++)
		{
			int ret = PAPI_add_event(profiling_info->papi_event_set, papi_events[i]);
#ifdef PAPI_ECMP_DISABLED
			if (ret == PAPI_ECMP_DISABLED && !warned_component_unavailable)
			{
				_STARPU_MSG("Error while registering Papi event: Component containing event is disabled. Try running `papi_component_avail` to get more information.\n");
				warned_component_unavailable = 1;
			}
#else
			(void)ret;
#endif
			profiling_info->papi_values[i]=0;
		}
		PAPI_reset(profiling_info->papi_event_set);
		PAPI_start(profiling_info->papi_event_set);
		STARPU_PTHREAD_MUTEX_UNLOCK(&papi_mutex);
	}
}

void _starpu_profiling_papi_task_stop_counters(struct starpu_task *task)
{
	if (!starpu_profiling_status_get())
		return;

	struct starpu_profiling_task_info *profiling_info;
	profiling_info = task->profiling_info;

	if (profiling_info && papi_nevents)
	{
		int i;
		STARPU_PTHREAD_MUTEX_LOCK(&papi_mutex);
		PAPI_stop(profiling_info->papi_event_set, profiling_info->papi_values);
		for(i=0; i<papi_nevents; i++)
		{
			_STARPU_TRACE_PAPI_TASK_EVENT(papi_events[i], task, profiling_info->papi_values[i]);
		}
		PAPI_cleanup_eventset(profiling_info->papi_event_set);
		PAPI_destroy_eventset(&profiling_info->papi_event_set);
		STARPU_PTHREAD_MUTEX_UNLOCK(&papi_mutex);
	}
}
#endif

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
		STARPU_PTHREAD_MUTEX_DESTROY(&_starpu_get_worker_struct(worker)->profiling_info_mutex);
	}
#ifdef STARPU_PAPI
	/* free the resources used by PAPI */
	STARPU_PTHREAD_MUTEX_LOCK(&papi_mutex);
	PAPI_shutdown();
	STARPU_PTHREAD_MUTEX_UNLOCK(&papi_mutex);
#endif

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
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct starpu_profiling_worker_info *worker_info = &worker->profiling_info;
	struct timespec now;
	_starpu_clock_gettime(&now);

	worker_info->start_time = now;

	/* This is computed in a lazy fashion when the application queries
	 * profiling info. */
	starpu_timespec_clear(&worker_info->total_time);

	starpu_timespec_clear(&worker_info->executing_time);
	starpu_timespec_clear(&worker_info->sleeping_time);

	worker_info->executed_tasks = 0;

	worker_info->used_cycles = 0;
	worker_info->stall_cycles = 0;
	worker_info->energy_consumed = 0;
	worker_info->flops = 0;

	/* We detect if the worker is already sleeping or doing some
	 * computation */
	enum _starpu_worker_status status = _starpu_worker_get_status(workerid);

	enum _starpu_worker_status_index i;

	for (i = 0; i < STATUS_INDEX_NR; i++)
	{
		if (status & (1 << i))
		{
			worker->profiling_registered_start[i] = 1;
			worker->profiling_registered_start_date[i] = now;
		}
		else
		{
			worker->profiling_registered_start[i] = 0;
		}
		worker->profiling_status = status;
		worker->profiling_status_start_date = now;
	}
}

static void _starpu_worker_time_split_accumulate(struct starpu_profiling_worker_info *worker_info, enum _starpu_worker_status status, struct timespec *delta)
{
	/* We here prioritize where we want to attribute the time spent */

	if (status & STATUS_EXECUTING)
		/* Executing task, this is all we want to know */
		starpu_timespec_accumulate(&worker_info->executing_time, delta);
	else if (status & STATUS_CALLBACK)
		/* Otherwise, callback, that's fine as well */
		starpu_timespec_accumulate(&worker_info->callback_time, delta);
	else if (status & STATUS_WAITING)
		/* Not doing any task or callback, held on waiting for some data */
		starpu_timespec_accumulate(&worker_info->waiting_time, delta);
	else if (status & STATUS_SLEEPING)
		/* Not even waiting for some data, but we don't have any task to do anyway */
		starpu_timespec_accumulate(&worker_info->sleeping_time, delta);
	else if (status & STATUS_SCHEDULING)
		/* We do have tasks to do, but the scheduler takes time */
		starpu_timespec_accumulate(&worker_info->scheduling_time, delta);
	/* And otherwise it's just uncategorized overhead */
}

void _starpu_worker_start_state(int workerid, enum _starpu_worker_status_index index, struct timespec *start_time)
{
	if (starpu_profiling_status_get())
	{
		struct timespec state_start_time;

		if (!start_time)
		{
			_starpu_clock_gettime(&state_start_time);
			start_time = &state_start_time;
		}

		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);

		STARPU_PTHREAD_MUTEX_LOCK(&worker->profiling_info_mutex);
		STARPU_ASSERT(worker->profiling_registered_start[index] == 0);
		worker->profiling_registered_start[index] = 1;
		worker->profiling_registered_start_date[index] = *start_time;

		if (worker->profiling_status != STATUS_UNKNOWN)
		{
			struct starpu_profiling_worker_info *worker_info = &worker->profiling_info;
			struct timespec state_time;
			starpu_timespec_sub(start_time, &worker->profiling_status_start_date, &state_time);
			_starpu_worker_time_split_accumulate(worker_info, worker->profiling_status, &state_time);
		}
		worker->profiling_status = _starpu_worker_get_status(workerid) | (1<<index);
		worker->profiling_status_start_date = *start_time;

		STARPU_PTHREAD_MUTEX_UNLOCK(&worker->profiling_info_mutex);
	}
}

static void _starpu_worker_time_accumulate(struct starpu_profiling_worker_info *worker_info, enum _starpu_worker_status_index index, struct timespec *delta)
{
	switch (index)
	{
	case STATUS_INDEX_EXECUTING:
		starpu_timespec_accumulate(&worker_info->all_executing_time, delta);
		break;
	case STATUS_INDEX_CALLBACK:
		starpu_timespec_accumulate(&worker_info->all_callback_time, delta);
		break;
	case STATUS_INDEX_WAITING:
		starpu_timespec_accumulate(&worker_info->all_waiting_time, delta);
		break;
	case STATUS_INDEX_SLEEPING:
		starpu_timespec_accumulate(&worker_info->all_sleeping_time, delta);
		break;
	case STATUS_INDEX_SCHEDULING:
		starpu_timespec_accumulate(&worker_info->all_scheduling_time, delta);
		break;
	case STATUS_INDEX_INITIALIZING:
		/* no profiling info for init */
		break;
	case STATUS_INDEX_NR:
		STARPU_ASSERT(0);
	}
}

void _starpu_worker_stop_state(int workerid, enum _starpu_worker_status_index index, struct timespec *stop_time)
{
	if (starpu_profiling_status_get())
	{
		struct timespec *state_start, state_end_time;
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		struct starpu_profiling_worker_info *worker_info = &worker->profiling_info;

		if (!stop_time)
		{
			_starpu_clock_gettime(&state_end_time);
			stop_time = &state_end_time;
		}

		STARPU_PTHREAD_MUTEX_LOCK(&worker->profiling_info_mutex);

		STARPU_ASSERT (worker->profiling_registered_start[index] == 1);
		{
			state_start = &worker->profiling_registered_start_date[index];

			/* Perhaps that profiling was enabled while the worker was
			 * already blocked, so we don't measure (end - start), but
			 * (end - max(start,worker_start)) where worker_start is the
			 * date of the previous profiling info reset on the worker */
			struct timespec *worker_start = &worker_info->start_time;
			if (starpu_timespec_cmp(state_start, worker_start, <))
			{
				/* state_start < worker_start */
				state_start = worker_start;
			}

			struct timespec state_time;
			starpu_timespec_sub(stop_time, state_start, &state_time);

			_starpu_worker_time_accumulate(worker_info, index, &state_time);

			worker->profiling_registered_start[index] = 0;
		}

		if (worker->profiling_status != STATUS_UNKNOWN)
		{
			struct timespec state_time;
			starpu_timespec_sub(stop_time, &worker->profiling_status_start_date, &state_time);
			_starpu_worker_time_split_accumulate(worker_info, worker->profiling_status, &state_time);
		}
		worker->profiling_status = _starpu_worker_get_status(workerid) & ~(1<<index);
		worker->profiling_status_start_date = *stop_time;

		STARPU_PTHREAD_MUTEX_UNLOCK(&worker->profiling_info_mutex);

	}
}

void _starpu_worker_update_profiling_info_executing(int workerid, int executed_tasks, uint64_t used_cycles, uint64_t stall_cycles, double energy_consumed, double flops)
{
	struct starpu_profiling_worker_info *worker_info = &_starpu_get_worker_struct(workerid)->profiling_info;

	if (starpu_profiling_status_get())
	{
		STARPU_PTHREAD_MUTEX_LOCK(&_starpu_get_worker_struct(workerid)->profiling_info_mutex);

		worker_info->used_cycles += used_cycles;
		worker_info->stall_cycles += stall_cycles;
		worker_info->energy_consumed += energy_consumed;
		worker_info->executed_tasks += executed_tasks;
		worker_info->flops += flops;

		STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_get_worker_struct(workerid)->profiling_info_mutex);
	}
	else /* Not thread safe, shouldn't be too much a problem */
		worker_info->executed_tasks += executed_tasks;
}

int starpu_profiling_worker_get_info(int workerid, struct starpu_profiling_worker_info *info)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct starpu_profiling_worker_info *worker_info = &worker->profiling_info;

	if (!starpu_profiling_status_get())
	{
		/* Not thread safe, shouldn't be too much a problem */
		info->executed_tasks = worker_info->executed_tasks;
	}

	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&_starpu_get_worker_struct(workerid)->sched_mutex);
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_get_worker_struct(workerid)->profiling_info_mutex);

	if (info)
	{
		/* The total time is computed in a lazy fashion */
		struct timespec now;
		_starpu_clock_gettime(&now);

		enum _starpu_worker_status_index i;

		for (i = 0; i< STATUS_INDEX_NR; i++)
		{
			/* In case some worker is currently doing something, we take into
			 * account the time spent since it registered. */
			if (worker->profiling_registered_start[i])
			{
				struct timespec delta;
				starpu_timespec_sub(&now, &worker->profiling_registered_start_date[i], &delta);
				_starpu_worker_time_accumulate(worker_info, i, &delta);
			}
		}
		if (worker->profiling_status != STATUS_UNKNOWN)
		{
			struct timespec delta;
			starpu_timespec_sub(&now, &worker->profiling_status_start_date, &delta);
			_starpu_worker_time_split_accumulate(worker_info, worker->profiling_status, &delta);
		}

		/* total_time = now - start_time */
		starpu_timespec_sub(&now, &worker_info->start_time,
					&worker_info->total_time);

		*info = *worker_info;
	}

	_starpu_worker_reset_profiling_info_with_lock(workerid);

	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_get_worker_struct(workerid)->profiling_info_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&_starpu_get_worker_struct(workerid)->sched_mutex);

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
		ngpus = topology->ndevices[STARPU_CUDA_WORKER]+topology->ndevices[STARPU_OPENCL_WORKER];
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

		*bus_info = bus_profiling_info[src_node][dst_node];
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
