/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <hwloc.h>
#include <starpu.h>
#include <starpurm.h>
#include <common/config.h>
#include <starpurm_private.h>

/*
 * #define _DEBUG
 */

struct s_starpurm_unit
{
	/* Opaque unit id.
	 * 
	 * For StarPU-RM, this id is used as an index to array starpurm->units[].
	 */
	int id;

	/* Id of the unit type. */
	int type;

	/* Boolean indicating whether the device is currently selected for use by the runtime system. */
	int selected;

	/* StarPU id of the worker driving the device. */
	int workerid;

	/* Cpuset of the StarPU worker. */
	hwloc_cpuset_t worker_cpuset;

	/* Condition variable to notify that a unit is now available to driver a worker waking up. */
	pthread_cond_t unit_available_cond;
};

static struct s_starpurm *_starpurm = NULL;

#if 0
static char *bitmap_to_str(hwloc_bitmap_t bitmap)
{
	int strl = hwloc_bitmap_snprintf(NULL, 0, bitmap);
	char *str = malloc(strl+1);
	hwloc_bitmap_snprintf(str, strl+1, bitmap);
	return str;
}
#endif

#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
enum e_starpurm_event
{
	starpurm_event_code_min              = 0,

	starpurm_event_exit                  = 0,
	starpurm_event_worker_going_to_sleep = 1,
	starpurm_event_worker_waking_up      = 2,
	starpurm_event_unit_available        = 3,

	starpurm_event_code_max              = 3
};

const char *_starpurm_event_to_str(int event_code)
{
	const char *s = NULL;
	switch (event_code)
	{
		case starpurm_event_exit:
			s = "starpurm_event_exit";
			break;
		case starpurm_event_worker_going_to_sleep:
			s = "starpurm_event_worker_going_to_sleep";
			break;
		case starpurm_event_worker_waking_up:
			s = "starpurm_event_worker_waking_up";
			break;
		case starpurm_event_unit_available:
			s = "starpurm_event_unit_available";
			break;
		default:
			s = "<unknown starpurm event>";
			break;
	}
	return s;
}

struct s_starpurm_event
{
	struct s_starpurm_event *next;
	struct s_starpurm_event *prev;
	enum e_starpurm_event code;
	unsigned int workerid;
};

static void _enqueue_event(struct s_starpurm_event *event)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	assert(event->next == NULL);
	assert(event->prev == NULL);
	assert(event->code >= starpurm_event_code_min && event->code <= starpurm_event_code_max);
	pthread_mutex_lock(&rm->event_list_mutex);
	if (rm->event_processing_ended)
	{
		pthread_mutex_unlock(&rm->event_list_mutex);
		return;
	}
	assert((rm->event_list_head == NULL && rm->event_list_tail == NULL)
			|| (rm->event_list_head != NULL && rm->event_list_tail != NULL));
	if (rm->event_list_head == NULL)
	{
		rm->event_list_tail = event;
	}
	else
	{
		rm->event_list_head->prev = event;
	}
	event->next = rm->event_list_head;
	rm->event_list_head = event;
	if (event->code == starpurm_event_exit)
	{
		rm->event_processing_ended = 1;
		int i;
		for (i=0; i<rm->nunits; i++)
		{
			pthread_cond_broadcast(&rm->units[i].unit_available_cond);
		}
	}
#ifdef STARPURM_VERBOSE
	if (event->code != starpurm_event_worker_waking_up)
		fprintf(stderr, "%s: event->code=%d('%s'), workerid=%u\n", __func__, event->code, _starpurm_event_to_str(event->code), event->workerid);
#endif
	pthread_cond_broadcast(&rm->event_list_cond);
#ifdef STARPURM_HAVE_DLB
	if (event->code == starpurm_event_worker_waking_up)
	{
		int unit_id = rm->worker_unit_ids[event->workerid];
		/* if DLB is in use, wait for the unit to become available from the point of view of DLB, before using it */
#ifdef STARPURM_VERBOSE
		fprintf(stderr, "%s: event->code=%d('%s'), workerid=%u - waiting\n", __func__, event->code, _starpurm_event_to_str(event->code), event->workerid);
#endif
		pthread_cond_wait(&rm->units[unit_id].unit_available_cond, &rm->event_list_mutex);
#ifdef STARPURM_VERBOSE
		fprintf(stderr, "%s: event->code=%d('%s'), workerid=%u - wakeup\n", __func__, event->code, _starpurm_event_to_str(event->code), event->workerid);
#endif
	}
#endif
	pthread_mutex_unlock(&rm->event_list_mutex);
}

static struct s_starpurm_event *_dequeue_event_no_lock(void)
{
	struct s_starpurm *rm = _starpurm;
	struct s_starpurm_event *event = NULL;
	if (rm->event_list_tail != NULL)
	{
		event = rm->event_list_tail;
		if (event->prev == NULL)
		{
			rm->event_list_head = NULL;
			rm->event_list_tail = NULL;
		}
		else
		{
			event->prev->next = NULL;
			rm->event_list_tail = event->prev;
		}
		event->prev = NULL;
		event->next = NULL;
	}
	return event;
}

static struct s_starpurm_event *_wait_event_no_lock(void)
{
	struct s_starpurm *rm = _starpurm;
	while (rm->event_list_head == NULL)
	{
		pthread_cond_wait(&rm->event_list_cond, &rm->event_list_mutex);
	}
	struct s_starpurm_event *event = _dequeue_event_no_lock();
	return event;
}

/* unused */
static struct s_starpurm_event *_dequeue_event(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	pthread_mutex_lock(&rm->event_list_mutex);
	struct s_starpurm_event *event = _dequeue_event_no_lock();
	pthread_mutex_unlock(&rm->event_list_mutex);
	return event;
}

/* unused */
static struct s_starpurm_event *_wait_event(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	pthread_mutex_lock(&rm->event_list_mutex);
	struct s_starpurm_event *event = _wait_event_no_lock();
	pthread_mutex_unlock(&rm->event_list_mutex);
	return event;
}

static void _enqueue_exit_event(void)
{
	struct s_starpurm_event *event = calloc(1, sizeof(*event));
	event->code = starpurm_event_exit;
	event->workerid = 0;
	_enqueue_event(event);
}

static void callback_worker_going_to_sleep(unsigned workerid)
{

	struct s_starpurm_event *event = calloc(1, sizeof(*event));
	event->code = starpurm_event_worker_going_to_sleep;
	event->workerid = workerid;
	_enqueue_event(event);
}

static void callback_worker_waking_up(unsigned workerid)
{
	struct s_starpurm_event *event = calloc(1, sizeof(*event));
	event->code = starpurm_event_worker_waking_up;
	event->workerid = workerid;
	_enqueue_event(event);
}

void starpurm_enqueue_event_cpu_unit_available(int unit_id)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	assert(unit_id >= 0);
	/*
	 * unit_id may exceed the number of CPU units actually used by StarPU,
	 * if some CPU cores are not used.
	 *
	 * //assert(unit_id < rm->nunits_by_type[starpurm_unit_cpu]);
	 */
	unsigned workerid = rm->units[unit_id].workerid; struct
		s_starpurm_event *event = calloc(1, sizeof(*event));
	event->code = starpurm_event_unit_available; event->workerid =
		workerid; _enqueue_event(event); }

static void *event_thread_func(void *_arg)
{
	(void)_arg;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	int need_refresh = 0;

	pthread_mutex_lock(&rm->event_list_mutex);
	while (rm->event_processing_enabled == 0)
	{
		pthread_cond_wait(&rm->event_processing_cond, &rm->event_list_mutex);
	}
	pthread_mutex_unlock(&rm->event_list_mutex);
	hwloc_cpuset_t owned_cpuset = hwloc_bitmap_dup(rm->global_cpuset);
	hwloc_cpuset_t to_reclaim_cpuset = hwloc_bitmap_alloc();
	hwloc_cpuset_t to_lend_cpuset = hwloc_bitmap_alloc();
	while (1)
	{
		struct s_starpurm_event *event = _dequeue_event();
#ifdef STARPURM_HAVE_DLB
		if ((event == NULL || event->code == starpurm_event_exit) || need_refresh)
#else
		if ((event == NULL || event->code == starpurm_event_exit) && need_refresh)
#endif
		{
			int did_lend_cpuset = 1;
#ifdef STARPURM_HAVE_DLB
			/* notify DLB about changes */
			if (!hwloc_bitmap_iszero(to_reclaim_cpuset))
			{
				starpurm_dlb_notify_starpu_worker_mask_waking_up(to_reclaim_cpuset);
			}
			did_lend_cpuset = 0;
			if (!hwloc_bitmap_iszero(to_lend_cpuset))
			{
				did_lend_cpuset = starpurm_dlb_notify_starpu_worker_mask_going_to_sleep(to_lend_cpuset);
			}
#endif
			/* if DLB is not initialized, ignore lend operations */
			if (did_lend_cpuset)
			{
				hwloc_bitmap_andnot(owned_cpuset, owned_cpuset, to_lend_cpuset);
			}
			hwloc_bitmap_or(owned_cpuset, owned_cpuset, to_reclaim_cpuset);

#if 0
			{
				char *to_lend_str = bitmap_to_str(to_lend_cpuset);
				char *to_reclaim_str = bitmap_to_str(to_reclaim_cpuset);
				free(to_lend_str);
				free(to_reclaim_str);
			}
#endif

			need_refresh = 0;
			hwloc_bitmap_zero(to_lend_cpuset);
			hwloc_bitmap_zero(to_reclaim_cpuset);
		}
		if (event == NULL)
		{
			event = _wait_event();
		}
		if (event->code == starpurm_event_exit)
		{
			free(event);
			break;
		}

		/* TODO: accumulate state change */
		switch (event->code)
		{
			case starpurm_event_worker_going_to_sleep:
				{
					if (event->workerid < rm->nunits)
					{
						int unit_id = rm->worker_unit_ids[event->workerid];
						hwloc_bitmap_or(to_lend_cpuset, to_lend_cpuset, rm->units[unit_id].worker_cpuset);
						hwloc_bitmap_andnot(to_reclaim_cpuset, to_reclaim_cpuset, rm->units[unit_id].worker_cpuset);
					}
				}
				break;
			case starpurm_event_worker_waking_up:
				{
					if (event->workerid < rm->nunits)
					{
						int unit_id = rm->worker_unit_ids[event->workerid];
						hwloc_bitmap_andnot(to_lend_cpuset, to_lend_cpuset, rm->units[unit_id].worker_cpuset);
#ifdef STARPURM_HAVE_DLB
						if (rm->units[unit_id].type == starpurm_unit_cpu && !hwloc_bitmap_intersects(rm->units[unit_id].worker_cpuset, owned_cpuset))
						{
							/* Only reclaim the unit from DLB if StarPU does not own it already. */
							hwloc_bitmap_or(to_reclaim_cpuset, to_reclaim_cpuset, rm->units[unit_id].worker_cpuset);
						}
						else
						{
							pthread_cond_broadcast(&rm->units[unit_id].unit_available_cond);
						}
#else
						hwloc_bitmap_or(to_reclaim_cpuset, to_reclaim_cpuset, rm->units[unit_id].worker_cpuset);
#endif
					}
				}
				break;
#ifdef STARPURM_HAVE_DLB
			case starpurm_event_unit_available:
				{
					if (event->workerid < rm->nunits)
					{
						/* a reclaimed unit is now available from DLB, unlock the corresponding worker waking up */
						int unit_id = rm->worker_unit_ids[event->workerid];
						pthread_cond_broadcast(&rm->units[unit_id].unit_available_cond);
					}
				}
				break;
#endif
			default:
				/* unknown event code */
				assert(0);
				break;
		}
		free(event);
		need_refresh = 1;
	}
	pthread_mutex_lock(&rm->event_list_mutex);
	/* exit event should be last */
	assert(rm->event_list_head == NULL);
	assert(rm->event_list_tail == NULL);
	hwloc_bitmap_free(owned_cpuset);
	hwloc_bitmap_free(to_reclaim_cpuset);
	hwloc_bitmap_free(to_lend_cpuset);
	pthread_mutex_unlock(&rm->event_list_mutex);
	return NULL;
}
#endif /* STARPURM_STARPU_HAVE_WORKER_CALLBACKS */

/* Resource enforcement */
static starpurm_drs_ret_t _starpurm_update_cpuset(hwloc_cpuset_t cpuset)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (hwloc_bitmap_isequal(cpuset, rm->selected_cpuset))
	{
		return starpurm_DRS_SUCCESS;
	}
	pthread_mutex_lock(&rm->temporary_ctxs_mutex);
	if (rm->starpu_in_pause)
	{
		starpu_resume();
		rm->starpu_in_pause = 0;
	}
	int workers_to_remove[_starpurm->nunits];
	unsigned nworkers_to_remove = 0;
	int workers_to_add[_starpurm->nunits];
	unsigned nworkers_to_add = 0;
	int i;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_alloc();
	int new_selected_ncpus = 0;
	for (i=0; i<rm->nunits; i++)
	{
		struct s_starpurm_unit *unit = &rm->units[i];
		hwloc_bitmap_and(temp_cpuset, unit->worker_cpuset, cpuset);
		if (hwloc_bitmap_iszero(temp_cpuset))
		{
			workers_to_remove[nworkers_to_remove] = unit->workerid;
			unit->selected = 0;
			nworkers_to_remove++;
		}
		else
		{
			workers_to_add[nworkers_to_add] = unit->workerid;
			unit->selected = 1;
			nworkers_to_add++;
			if (unit->type == starpurm_unit_cpu)
			{
				new_selected_ncpus++;
			}
		}
	}
	hwloc_bitmap_free(temp_cpuset);
	rm->selected_nworkers = nworkers_to_add;
	rm->selected_ncpus = new_selected_ncpus;
	hwloc_bitmap_free(rm->selected_cpuset);
	rm->selected_cpuset = hwloc_bitmap_dup(cpuset);

	if (nworkers_to_add > 0)
	{
#if defined(STARPURM_HAVE_DLB) && !defined(STARPURM_STARPU_HAVE_WORKER_CALLBACKS)
		{
			/* if StarPU worker callbacks are not enabled, we still
			 * notify DLB about resource usage changes, but we do
			 * not wait for the formal DLB go to use the units */
			hwloc_cpuset_t to_reclaim_cpuset = hwloc_bitmap_alloc();
			for (i=0; i<nworkers_to_add; i++)
			{
				int unit_id = rm->worker_unit_ids[workers_to_add[i]];
				hwloc_bitmap_or(to_reclaim_cpuset, to_reclaim_cpuset, rm->units[unit_id].worker_cpuset);
			}
			starpurm_dlb_notify_starpu_worker_mask_waking_up(to_reclaim_cpuset);
			hwloc_bitmap_free(to_reclaim_cpuset);
		}
#endif
		starpu_sched_ctx_add_workers(workers_to_add, nworkers_to_add, rm->sched_ctx_id);
	}
	if (nworkers_to_remove > 0)
	{
		starpu_sched_ctx_remove_workers(workers_to_remove, nworkers_to_remove, rm->sched_ctx_id);
#if defined(STARPURM_HAVE_DLB) && !defined(STARPURM_STARPU_HAVE_WORKER_CALLBACKS)
		{
			/* if StarPU worker callbacks are not enabled, we still
			 * notify DLB about resource usage changes, but we do
			 * not wait for the workers to become idle */
			hwloc_cpuset_t to_lend_cpuset = hwloc_bitmap_alloc();
			for (i=0; i<nworkers_to_remove; i++)
			{
				int unit_id = rm->worker_unit_ids[workers_to_remove[i]];
				hwloc_bitmap_or(to_lend_cpuset, to_lend_cpuset, rm->units[unit_id].worker_cpuset);
			}
			starpurm_dlb_notify_starpu_worker_mask_going_to_sleep(to_lend_cpuset);
			hwloc_bitmap_free(to_lend_cpuset);
		}
#endif
	}
#ifdef _DEBUG
	starpu_sched_ctx_display_workers(rm->sched_ctx_id, stderr);
#endif /* DEBUG */
	if (rm->selected_nworkers == 0 && rm->avail_temporary_ctxs == rm->max_temporary_ctxs)
	{
		rm->starpu_in_pause = 1;
		starpu_pause();
	}
	pthread_mutex_unlock(&rm->temporary_ctxs_mutex);
	return starpurm_DRS_SUCCESS;
}

static unsigned _starpurm_temporary_context_alloc(hwloc_cpuset_t cpuset)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	assert(_starpurm->max_temporary_ctxs > 0);
	struct s_starpurm *rm = _starpurm;
	pthread_mutex_lock(&rm->temporary_ctxs_mutex);
	while(rm->avail_temporary_ctxs == 0)
	{
		pthread_cond_wait(&rm->temporary_ctxs_cond, &rm->temporary_ctxs_mutex);
	}
	assert(rm->avail_temporary_ctxs > 0);
	rm->avail_temporary_ctxs--;
	if (rm->starpu_in_pause)
	{
		starpu_resume();
		rm->starpu_in_pause = 0;
	}
	pthread_mutex_unlock(&rm->temporary_ctxs_mutex);
	unsigned sched_ctx_id = starpu_sched_ctx_create(NULL, -1, "starpurm_temp", STARPU_SCHED_CTX_POLICY_NAME, "eager", 0);
	assert(sched_ctx_id != STARPU_NMAX_SCHED_CTXS);
	int workers_to_remove[_starpurm->nunits];
	unsigned nworkers_to_remove = 0;
	int workers_to_add[_starpurm->nunits];
	unsigned nworkers_to_add = 0;
	int i;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_alloc();
	for (i=0; i<rm->nunits; i++)
	{
		struct s_starpurm_unit *unit = &rm->units[i];
		hwloc_bitmap_and(temp_cpuset, unit->worker_cpuset, cpuset);
		if (hwloc_bitmap_iszero(temp_cpuset))
		{
			workers_to_remove[nworkers_to_remove] = unit->workerid;
			nworkers_to_remove++;
		}
		else
		{
			workers_to_add[nworkers_to_add] = unit->workerid;
			nworkers_to_add++;
		}
	}
	hwloc_bitmap_free(temp_cpuset);

	if (nworkers_to_add > 0)
		starpu_sched_ctx_add_workers(workers_to_add, nworkers_to_add, sched_ctx_id);
	if (nworkers_to_remove > 0)
		starpu_sched_ctx_remove_workers(workers_to_remove, nworkers_to_remove, sched_ctx_id);
#ifdef _DEBUG
	starpu_sched_ctx_display_workers(sched_ctx_id, stderr);
#endif /* DEBUG */
	return sched_ctx_id;
}

static void _starpurm_temporary_context_free(unsigned ctx)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	assert(_starpurm->max_temporary_ctxs > 0);
	struct s_starpurm *rm = _starpurm;
	starpu_sched_ctx_delete(ctx);
	pthread_mutex_lock(&rm->temporary_ctxs_mutex);
	rm->avail_temporary_ctxs++;
	pthread_cond_signal(&rm->temporary_ctxs_cond);
	if (rm->selected_nworkers == 0 && rm->avail_temporary_ctxs == rm->max_temporary_ctxs)
	{
		rm->starpu_in_pause = 1;
		starpu_pause();
	}
	pthread_mutex_unlock(&rm->temporary_ctxs_mutex);
}

static starpurm_drs_ret_t _starpurm_set_ncpus(unsigned int ncpus)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	int i;
	if (ncpus > rm->nunits_by_type[starpurm_unit_cpu])
	{
		ncpus = rm->nunits_by_type[starpurm_unit_cpu];
	}
	if (ncpus == rm->selected_ncpus)
	{
		return starpurm_DRS_SUCCESS;
	}
	pthread_mutex_lock(&rm->temporary_ctxs_mutex);
	if (rm->starpu_in_pause)
	{
		starpu_resume();
		rm->starpu_in_pause = 0;
	}
	int workers_to_remove[_starpurm->nunits];
	unsigned nworkers_to_remove = 0;
	int workers_to_add[_starpurm->nunits];
	unsigned nworkers_to_add = 0;
	for (i=0; i<rm->nunits; i++)
	{
		struct s_starpurm_unit *unit = &rm->units[i];
		if (unit->type != starpurm_unit_cpu)
			continue;
		if (nworkers_to_add < ncpus)
		{
			workers_to_add[nworkers_to_add] = unit->workerid;
			unit->selected = 1;
			nworkers_to_add++;
			hwloc_bitmap_or(rm->selected_cpuset, rm->selected_cpuset, unit->worker_cpuset);
		}
		else
		{
			workers_to_remove[nworkers_to_remove] = unit->workerid;
			unit->selected = 0;
			hwloc_bitmap_andnot(rm->selected_cpuset, rm->selected_cpuset, unit->worker_cpuset);
			nworkers_to_remove++;
		}
	}

	rm->selected_nworkers = nworkers_to_add;
	rm->selected_ncpus = nworkers_to_add;

	if (nworkers_to_add > 0)
		starpu_sched_ctx_add_workers(workers_to_add, nworkers_to_add, rm->sched_ctx_id);
	if (nworkers_to_remove > 0)
		starpu_sched_ctx_remove_workers(workers_to_remove, nworkers_to_remove, rm->sched_ctx_id);
#if def_DEBUG
	starpu_sched_ctx_display_workers(rm->sched_ctx_id, stderr);
#endif /* DEBUG */
	if (rm->selected_nworkers == 0 && rm->avail_temporary_ctxs == rm->max_temporary_ctxs)
	{
		rm->starpu_in_pause = 1;
		starpu_pause();
	}
	pthread_mutex_unlock(&rm->temporary_ctxs_mutex);
	return starpurm_DRS_SUCCESS;
}

/* Initialize rm state for StarPU */
void starpurm_initialize_with_cpuset(const hwloc_cpuset_t initially_owned_cpuset)
{
	int ret;
	assert(_starpurm == NULL);

	struct s_starpurm *rm = calloc(1, sizeof(*rm));
	pthread_mutex_init(&rm->temporary_ctxs_mutex, NULL);
	pthread_cond_init(&rm->temporary_ctxs_cond, NULL);
	rm->state = state_init;

	/* init hwloc objects */
	hwloc_topology_init(&rm->topology);
	hwloc_topology_load(rm->topology);
	rm->global_cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(rm->global_cpuset);
	
	rm->initially_owned_cpuset_mask = hwloc_bitmap_dup(initially_owned_cpuset);

	rm->all_cpu_workers_cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(rm->all_cpu_workers_cpuset);
	
	rm->all_opencl_device_workers_cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(rm->all_opencl_device_workers_cpuset);
	
	rm->all_cuda_device_workers_cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(rm->all_cuda_device_workers_cpuset);
	
	rm->all_mic_device_workers_cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(rm->all_mic_device_workers_cpuset);

	rm->all_device_workers_cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(rm->all_device_workers_cpuset);

	/* init event list, before StarPU is initialized */
	pthread_mutex_init(&rm->event_list_mutex, NULL);
	pthread_cond_init(&rm->event_list_cond, NULL);
	pthread_cond_init(&rm->event_processing_cond, NULL);
	pthread_mutex_lock(&rm->event_list_mutex);
	rm->event_processing_enabled = 0;
	rm->event_processing_ended = 0;
	rm->event_list_head = NULL;
	rm->event_list_tail = NULL;
	pthread_mutex_unlock(&rm->event_list_mutex);

	/* set _starpurm here since StarPU's callbacks may reference it once starpu_init is called */
	_starpurm = rm;

#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
	/* launch event thread */
	ret = pthread_create(&rm->event_thread, NULL, event_thread_func, rm);
	assert(ret == 0);
#endif

	/* init StarPU */
	struct starpu_conf starpu_conf;
	ret = starpu_conf_init(&starpu_conf);
	assert(ret == 0);

#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
	starpu_conf.callback_worker_going_to_sleep = callback_worker_going_to_sleep;
	starpu_conf.callback_worker_waking_up = callback_worker_waking_up;
#endif

	ret = starpu_init(&starpu_conf);
	assert(ret == 0);

	/* init any worker objects */
	rm->nunits = starpu_worker_get_count_by_type(STARPU_ANY_WORKER);

	/* init device worker objects */
	rm->unit_ntypes = starpurm_unit_ntypes;
	rm->nunits_by_type = calloc(rm->unit_ntypes, sizeof(*rm->nunits_by_type));
	rm->unit_offsets_by_type = calloc(rm->unit_ntypes, sizeof(*rm->unit_offsets_by_type));

	const int cpu_nunits = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	rm->nunits_by_type[starpurm_unit_cpu] = cpu_nunits;

	const int opencl_nunits = starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER);
	rm->nunits_by_type[starpurm_unit_opencl] = opencl_nunits;

	const int cuda_nunits = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
	rm->nunits_by_type[starpurm_unit_cuda] = cuda_nunits;

	const int mic_nunits = starpu_worker_get_count_by_type(STARPU_MIC_WORKER);
	rm->nunits_by_type[starpurm_unit_mic] = mic_nunits;

	const int nunits = cpu_nunits + opencl_nunits + cuda_nunits + mic_nunits;
	rm->nunits = nunits;
	rm->units = calloc(nunits, sizeof(*rm->units));

	int unitid = 0;

	int cpu_workerids[cpu_nunits];
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, cpu_workerids, cpu_nunits);
	rm->unit_offsets_by_type[starpurm_unit_cpu] = unitid;
	unsigned int max_worker_id = 0;
	int i;
	for (i = 0; i < cpu_nunits; i++)
	{
		rm->units[unitid].id = unitid;
		rm->units[unitid].type = starpurm_unit_cpu;
		rm->units[unitid].selected = 1; /* enabled by default */
		rm->units[unitid].workerid = cpu_workerids[i];
		if (max_worker_id < rm->units[unitid].workerid)
		{
			max_worker_id = rm->units[unitid].workerid;
		}
		rm->units[unitid].worker_cpuset = starpu_worker_get_hwloc_cpuset(rm->units[unitid].workerid);
		pthread_cond_init(&rm->units[unitid].unit_available_cond, NULL);
		hwloc_bitmap_or(rm->global_cpuset, rm->global_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_cpu_workers_cpuset, rm->all_cpu_workers_cpuset, rm->units[unitid].worker_cpuset);;
#ifdef STARPURM_VERBOSE
		{
			char * s_unit = NULL;
			hwloc_bitmap_asprintf(&s_unit, rm->units[unitid].worker_cpuset);
			fprintf(stderr, "%s: 'cpu', unitid=%d, cpuset=0x%s, workerid=%d\n", __func__, unitid, s_unit, rm->units[unitid].workerid);
			free(s_unit);
		}
#endif
		unitid++;
	}

	int opencl_workerids[opencl_nunits];
	starpu_worker_get_ids_by_type(STARPU_OPENCL_WORKER, opencl_workerids, opencl_nunits);
	rm->unit_offsets_by_type[starpurm_unit_opencl] = unitid;
	for (i = 0; i < opencl_nunits; i++)
	{
		rm->units[unitid].id = unitid;
		rm->units[unitid].type = starpurm_unit_opencl;
		rm->units[unitid].selected = 1; /* enabled by default */
		rm->units[unitid].workerid = opencl_workerids[i];
		if (max_worker_id < rm->units[unitid].workerid)
		{
			max_worker_id = rm->units[unitid].workerid;
		}
		rm->units[unitid].worker_cpuset = starpu_worker_get_hwloc_cpuset(rm->units[unitid].workerid);
		pthread_cond_init(&rm->units[unitid].unit_available_cond, NULL);
		hwloc_bitmap_or(rm->global_cpuset, rm->global_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_opencl_device_workers_cpuset, rm->all_opencl_device_workers_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_device_workers_cpuset, rm->all_device_workers_cpuset, rm->units[unitid].worker_cpuset);
		unitid++;
	}

	int cuda_workerids[opencl_nunits];
	starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, cuda_workerids, cuda_nunits);
	rm->unit_offsets_by_type[starpurm_unit_cuda] = unitid;
	for (i = 0; i < cuda_nunits; i++)
	{
		rm->units[unitid].id = unitid;
		rm->units[unitid].type = starpurm_unit_cuda;
		rm->units[unitid].selected = 1; /* enabled by default */
		rm->units[unitid].workerid = cuda_workerids[i];
		if (max_worker_id < rm->units[unitid].workerid)
		{
			max_worker_id = rm->units[unitid].workerid;
		}
		rm->units[unitid].worker_cpuset = starpu_worker_get_hwloc_cpuset(rm->units[unitid].workerid);
		pthread_cond_init(&rm->units[unitid].unit_available_cond, NULL);
		hwloc_bitmap_or(rm->global_cpuset, rm->global_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_cuda_device_workers_cpuset, rm->all_cuda_device_workers_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_device_workers_cpuset, rm->all_device_workers_cpuset, rm->units[unitid].worker_cpuset);
		unitid++;
	}

	int mic_workerids[mic_nunits];
	starpu_worker_get_ids_by_type(STARPU_MIC_WORKER, mic_workerids, mic_nunits);
	rm->unit_offsets_by_type[starpurm_unit_mic] = unitid;
	for (i = 0; i < mic_nunits; i++)
	{
		rm->units[unitid].id = unitid;
		rm->units[unitid].type = starpurm_unit_mic;
		rm->units[unitid].selected = 1; /* enabled by default */
		rm->units[unitid].workerid = mic_workerids[i];
		if (max_worker_id < rm->units[unitid].workerid)
		{
			max_worker_id = rm->units[unitid].workerid;
		}
		rm->units[unitid].worker_cpuset = starpu_worker_get_hwloc_cpuset(rm->units[unitid].workerid);
		pthread_cond_init(&rm->units[unitid].unit_available_cond, NULL);
		hwloc_bitmap_or(rm->global_cpuset, rm->global_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_mic_device_workers_cpuset, rm->all_mic_device_workers_cpuset, rm->units[unitid].worker_cpuset);
		hwloc_bitmap_or(rm->all_device_workers_cpuset, rm->all_device_workers_cpuset, rm->units[unitid].worker_cpuset);
		unitid++;
	}

	rm->max_worker_id = max_worker_id;
	{
		int *worker_unit_ids = malloc((max_worker_id+1) * sizeof(*worker_unit_ids));
		for (i = 0; i < max_worker_id+1; i++)
		{
			worker_unit_ids[i] = -1;
		}
		for (i=0; i<rm->nunits; i++)
		{
			worker_unit_ids[rm->units[i].workerid] = i;
		}
		rm->worker_unit_ids = worker_unit_ids;
	}

	/* create StarPU sched_ctx for RM instance */
	{
		int workerids[rm->nunits];
		starpu_worker_get_ids_by_type(STARPU_ANY_WORKER, workerids, rm->nunits);
		/* TODO: make sched_ctx policy configurable */
		rm->sched_ctx_id = starpu_sched_ctx_create(workerids, rm->nunits, "starpurm", STARPU_SCHED_CTX_POLICY_NAME, "eager", 0);
#ifdef _DEBUG
		starpu_sched_ctx_display_workers(rm->sched_ctx_id, stderr);
#endif /* DEBUG */
	}

	starpu_sched_ctx_set_context(&rm->sched_ctx_id);

	/* number selected workers (total) */
	rm->selected_nworkers = rm->nunits;

	/* number of selected CPUs workers */
	rm->selected_ncpus = rm->nunits_by_type[starpurm_unit_cpu];

	/* cpuset of all currently selected workers */
	rm->selected_cpuset = hwloc_bitmap_dup(rm->global_cpuset);

	if (STARPU_NMAX_SCHED_CTXS > 2)
	{
		/* account for main ctx (0) and default rm ctx (1)
		 * TODO: check that no other ctxs are allocated by external codes */
		rm->max_temporary_ctxs = STARPU_NMAX_SCHED_CTXS - 2;
	}
	else
	{
		rm->max_temporary_ctxs = 0;
	}
	rm->avail_temporary_ctxs = rm->max_temporary_ctxs;
	if (rm->selected_nworkers == 0)
	{
		rm->starpu_in_pause = 1;
		starpu_pause();
	}
	else
	{
		rm->starpu_in_pause = 0;
	}

#ifdef STARPURM_HAVE_DLB
	starpurm_dlb_init(rm);
#endif
	pthread_mutex_lock(&rm->event_list_mutex);
	rm->event_processing_enabled = 1;
	pthread_cond_broadcast(&rm->event_processing_cond);
	pthread_mutex_unlock(&rm->event_list_mutex);
	_starpurm = rm;

}

void starpurm_initialize()
{
	hwloc_cpuset_t full_cpuset = hwloc_bitmap_alloc_full();
	starpurm_initialize_with_cpuset(full_cpuset);
	hwloc_bitmap_free(full_cpuset);
}

/* Free rm struct for StarPU */
void starpurm_shutdown(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	
	if (rm->starpu_in_pause)
	{
		starpu_resume();
		rm->starpu_in_pause = 0;
	}

	starpu_sched_ctx_delete(rm->sched_ctx_id);
#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
	_enqueue_exit_event();
#endif
	starpu_shutdown();
#ifdef STARPURM_HAVE_DLB
	starpurm_dlb_exit();
#endif
	hwloc_topology_destroy(rm->topology);
#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
	pthread_join(rm->event_thread, NULL);
#endif
	assert(rm->event_list_head == NULL);
	assert(rm->event_list_tail == NULL);
	pthread_cond_destroy(&rm->event_list_cond);
	pthread_mutex_destroy(&rm->event_list_mutex);

	rm->state = state_uninitialized;

	hwloc_bitmap_free(rm->global_cpuset);
	hwloc_bitmap_free(rm->all_cpu_workers_cpuset);
	hwloc_bitmap_free(rm->all_opencl_device_workers_cpuset);
	hwloc_bitmap_free(rm->all_cuda_device_workers_cpuset);
	hwloc_bitmap_free(rm->all_mic_device_workers_cpuset);
	hwloc_bitmap_free(rm->all_device_workers_cpuset);
	hwloc_bitmap_free(rm->selected_cpuset);
	hwloc_bitmap_free(rm->initially_owned_cpuset_mask);

	int i;
	for (i=0; i<rm->nunits; i++)
	{
		pthread_cond_destroy(&rm->units[i].unit_available_cond);
	}
	free(rm->units);
	rm->units = NULL;

	free(rm->nunits_by_type);
	rm->nunits_by_type = NULL;

	free(rm->unit_offsets_by_type);
	rm->unit_offsets_by_type = NULL;

	free(rm);
	_starpurm = NULL;
}


void starpurm_spawn_kernel_on_cpus(void *data, void(*f)(void *), void *args, hwloc_cpuset_t cpuset)
{
	(void) data;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	unsigned ctx = _starpurm_temporary_context_alloc(cpuset);
	starpu_sched_ctx_set_context(&ctx);
	f(args);
	starpu_sched_ctx_set_context(&rm->sched_ctx_id);
	_starpurm_temporary_context_free(ctx);
}

struct s_starpurm__spawn_args
{
	void(*f)(void *);
	void *args;
	void(*cb_f)(void *);
	void *cb_args;
	hwloc_cpuset_t cpuset;
};

static void *_starpurm_spawn_kernel_thread(void *_spawn_args)
{
	struct s_starpurm__spawn_args *spawn_args = _spawn_args;
	unsigned ctx = _starpurm_temporary_context_alloc(spawn_args->cpuset);
	starpu_sched_ctx_set_context(&ctx);
	spawn_args->f(spawn_args->args);
	struct s_starpurm *rm = _starpurm;
	starpu_sched_ctx_set_context(&rm->sched_ctx_id);
	_starpurm_temporary_context_free(ctx);
	spawn_args->cb_f(spawn_args->cb_args);
	hwloc_bitmap_free(spawn_args->cpuset);
	free(spawn_args);
	return NULL;
}

void starpurm_spawn_kernel_on_cpus_callback(void *data, void(*f)(void *), void *args, hwloc_cpuset_t cpuset, void(*cb_f)(void *), void *cb_args)
{
	(void) data;
	struct s_starpurm__spawn_args *spawn_args = calloc(1, sizeof(*spawn_args));
	spawn_args->f = f;
	spawn_args->args = args;
	spawn_args->cb_f = cb_f;
	spawn_args->cb_args = cb_args;
	spawn_args->cpuset = hwloc_bitmap_dup(cpuset);
	pthread_attr_t attr;
	int ret;
	ret = pthread_attr_init(&attr);
	assert(ret == 0);
	ret = pthread_attr_setdetachstate(&attr, 1);
	assert(ret == 0);
	pthread_t t;
	ret = pthread_create(&t, &attr, _starpurm_spawn_kernel_thread, spawn_args);
	assert(ret == 0);

}

static void *_starpurm_spawn_kernel_in_default_context_thread(void *_spawn_args)
{
	struct s_starpurm__spawn_args *spawn_args = _spawn_args;
	struct s_starpurm *rm = _starpurm;
	starpu_sched_ctx_set_context(&rm->sched_ctx_id);
	spawn_args->f(spawn_args->args);
	spawn_args->cb_f(spawn_args->cb_args);
	free(spawn_args);
	return NULL;
}

void starpurm_spawn_kernel_callback(void *data, void(*f)(void *), void *args, void(*cb_f)(void *), void *cb_args)
{
	(void) data;
	struct s_starpurm__spawn_args *spawn_args = calloc(1, sizeof(*spawn_args));
	spawn_args->f = f;
	spawn_args->args = args;
	spawn_args->cb_f = cb_f;
	spawn_args->cb_args = cb_args;
	pthread_attr_t attr;
	int ret;
	ret = pthread_attr_init(&attr);
	assert(ret == 0);
	ret = pthread_attr_setdetachstate(&attr, 1);
	assert(ret == 0);
	pthread_t t;
	ret = pthread_create(&t, &attr, _starpurm_spawn_kernel_in_default_context_thread, spawn_args);
	assert(ret == 0);

}

hwloc_cpuset_t starpurm_get_unit_cpuset(int unitid)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	assert(unitid >= 0 && unitid < rm->nunits);
	return hwloc_bitmap_dup(rm->units[unitid].worker_cpuset);
}

hwloc_cpuset_t starpurm_get_cpu_worker_cpuset(int unit_rank)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	assert(unit_rank >= 0 && unit_rank < rm->nunits_by_type[starpurm_unit_cpu]);
	return hwloc_bitmap_dup(rm->units[rm->unit_offsets_by_type[starpurm_unit_cpu] + unit_rank].worker_cpuset);
}

/* Dynamic resource sharing */
starpurm_drs_ret_t starpurm_set_drs_enable(starpurm_drs_desc_t *spd)
{
	(void)spd;

	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	rm->dynamic_resource_sharing = 1;
	return starpurm_DRS_SUCCESS;
}

starpurm_drs_ret_t starpurm_set_drs_disable(starpurm_drs_desc_t *spd)
{
	(void)spd;

	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	rm->dynamic_resource_sharing = 0;
	return starpurm_DRS_SUCCESS;
}

int starpurm_drs_enabled_p(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	return rm->dynamic_resource_sharing;
}


starpurm_drs_ret_t starpurm_set_max_parallelism(starpurm_drs_desc_t *spd, int ncpus)
{
	(void)spd;

	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (ncpus > rm->nunits_by_type[starpurm_unit_cpu])
	{
		ncpus = rm->nunits_by_type[starpurm_unit_cpu];
	}
	rm->max_ncpus = ncpus;
	if (rm->selected_ncpus > ncpus)
	{
		return _starpurm_set_ncpus(ncpus);
	}
	return starpurm_DRS_SUCCESS;
}


starpurm_drs_ret_t starpurm_callback_set(starpurm_drs_desc_t *spd, starpurm_drs_cbs_t which, starpurm_drs_cb_t callback)
{
	(void)spd;
	(void)which;
	(void)callback;
	/* unimplemented */
	assert(0);
	return starpurm_DRS_PERM;
}

starpurm_drs_ret_t starpurm_callback_get(starpurm_drs_desc_t *spd, starpurm_drs_cbs_t which, starpurm_drs_cb_t *callback)
{
	(void)spd;
	(void)which;
	(void)callback;
	/* unimplemented */
	assert(0);
	return starpurm_DRS_PERM;
}

starpurm_drs_ret_t starpurm_assign_cpu_to_starpu(starpurm_drs_desc_t *spd, int cpuid)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	starpurm_drs_ret_t ret = 0;
	assert(hwloc_bitmap_isset(rm->global_cpuset, cpuid));
	if (!hwloc_bitmap_isset(rm->selected_cpuset, cpuid))
	{
		hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
		hwloc_bitmap_set(temp_cpuset, cpuid);
		ret = _starpurm_update_cpuset(temp_cpuset);
		hwloc_bitmap_free(temp_cpuset);
	}
	return ret;
}

starpurm_drs_ret_t starpurm_assign_cpus_to_starpu(starpurm_drs_desc_t *spd, int ncpus)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	/* add ncpus more CPUs to the CPUs pool */
	return _starpurm_set_ncpus(rm->selected_ncpus+ncpus);
}

starpurm_drs_ret_t starpurm_assign_cpu_mask_to_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_or(temp_cpuset, temp_cpuset, mask);
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_assign_all_cpus_to_starpu(starpurm_drs_desc_t *spd)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	return starpurm_assign_cpus_to_starpu(spd, rm->nunits_by_type[starpurm_unit_cpu]);
}

starpurm_drs_ret_t starpurm_withdraw_cpu_from_starpu(starpurm_drs_desc_t *spd, int cpuid)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	starpurm_drs_ret_t ret = 0;
	assert(hwloc_bitmap_isset(rm->global_cpuset, cpuid));
	if (hwloc_bitmap_isset(rm->selected_cpuset, cpuid))
	{
		hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
		hwloc_bitmap_clr(temp_cpuset, cpuid);
		ret = _starpurm_update_cpuset(temp_cpuset);
		hwloc_bitmap_free(temp_cpuset);
	}
	return ret;
}

starpurm_drs_ret_t starpurm_withdraw_cpus_from_starpu(starpurm_drs_desc_t *spd, int ncpus)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	/* add ncpus more CPUs to the CPUs pool */
	starpurm_drs_ret_t ret = 0;
	if (ncpus <= rm->nunits_by_type[starpurm_unit_cpu])
	{
		ret = _starpurm_set_ncpus(rm->nunits_by_type[starpurm_unit_cpu]-ncpus);
	}
	else
	{
		ret = _starpurm_set_ncpus(0);
	}
	return ret;
}

starpurm_drs_ret_t starpurm_withdraw_cpu_mask_from_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_andnot(temp_cpuset, temp_cpuset, mask);
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_withdraw_all_cpus_from_starpu(starpurm_drs_desc_t *spd)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	return starpurm_withdraw_cpus_from_starpu(spd, rm->nunits_by_type[starpurm_unit_cpu]);
}

/* --- */

starpurm_drs_ret_t starpurm_lend_cpu(starpurm_drs_desc_t *spd, int cpuid)
{
	return starpurm_assign_cpu_to_starpu(spd, cpuid);
}

starpurm_drs_ret_t starpurm_lend_cpus(starpurm_drs_desc_t *spd, int ncpus)
{
	return starpurm_assign_cpus_to_starpu(spd, ncpus);
}

starpurm_drs_ret_t starpurm_lend_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	return starpurm_assign_cpu_mask_to_starpu(spd, mask);
}

starpurm_drs_ret_t starpurm_lend(starpurm_drs_desc_t *spd)
{
	return starpurm_assign_all_cpus_to_starpu(spd);
}


starpurm_drs_ret_t starpurm_reclaim_cpu(starpurm_drs_desc_t *spd, int cpuid)
{
	return starpurm_withdraw_cpu_from_starpu(spd, cpuid);
}

starpurm_drs_ret_t starpurm_reclaim_cpus(starpurm_drs_desc_t *spd, int ncpus)
{
	return starpurm_withdraw_cpus_from_starpu(spd, ncpus);
}

starpurm_drs_ret_t starpurm_reclaim_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	return starpurm_withdraw_cpu_mask_from_starpu(spd, mask);
}

starpurm_drs_ret_t starpurm_reclaim(starpurm_drs_desc_t *spd)
{
	return starpurm_withdraw_all_cpus_from_starpu(spd);
}

starpurm_drs_ret_t starpurm_acquire(starpurm_drs_desc_t *spd)
{
	return starpurm_withdraw_all_cpus_from_starpu(spd);
}

starpurm_drs_ret_t starpurm_acquire_cpu(starpurm_drs_desc_t *spd, int cpuid)
{
	return starpurm_withdraw_cpu_from_starpu(spd, cpuid);
}

starpurm_drs_ret_t starpurm_acquire_cpus(starpurm_drs_desc_t *spd, int ncpus)
{
	return starpurm_withdraw_cpus_from_starpu(spd, ncpus);
}

starpurm_drs_ret_t starpurm_acquire_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	return starpurm_withdraw_cpu_mask_from_starpu(spd, mask);
}


starpurm_drs_ret_t starpurm_return_all(starpurm_drs_desc_t *spd)
{
	return starpurm_assign_all_cpus_to_starpu(spd);
}

starpurm_drs_ret_t starpurm_return_cpu(starpurm_drs_desc_t *spd, int cpuid)
{
	return starpurm_assign_cpu_to_starpu(spd, cpuid);
}


/* Pause/resume */
starpurm_drs_ret_t starpurm_create_block_condition(starpurm_block_cond_t *cond)
{
	/* unimplemented */
	assert(0);
	return starpurm_DRS_PERM;
}

void starpurm_block_current_task(starpurm_block_cond_t *cond)
{
	/* unimplemented */
	assert(0);
}

void starpurm_signal_block_condition(starpurm_block_cond_t *cond)
{
	/* unimplemented */
	assert(0);
}

 
void starpurm_register_polling_service(const char *service_name, starpurm_polling_t function, void *data)
{
	/* unimplemented */
	assert(0);
}

void starpurm_unregister_polling_service(const char *service_name, starpurm_polling_t function, void *data)
{
	/* unimplemented */
	assert(0);
}

/* devices */
int starpurm_get_device_type_id(const char *type_str)
{
	if (strcmp(type_str, "cpu") == 0)
		return starpurm_unit_cpu;
	if (strcmp(type_str, "opencl") == 0)
		return starpurm_unit_opencl;
	if (strcmp(type_str, "cuda") == 0)
		return starpurm_unit_cuda;
	if (strcmp(type_str, "mic") == 0)
		return starpurm_unit_mic;
	return -1;
}

const char *starpurm_get_device_type_name(int type_id)
{
	if (type_id == starpurm_unit_cpu)
		return "cpu";
	if (type_id == starpurm_unit_opencl)
		return "opencl";
	if (type_id == starpurm_unit_cuda)
		return "cuda";
	if (type_id == starpurm_unit_mic)
		return "mic";
	return NULL;
}

int starpurm_get_nb_devices_by_type(int type_id)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return -1;
	return rm->nunits_by_type[type_id];
}

int starpurm_get_device_id(int type_id, int unit_rank)
{

	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return -1;
	if (unit_rank < 0 || unit_rank >= rm->nunits_by_type[type_id])
		return -1;
	return rm->units[rm->unit_offsets_by_type[type_id] + unit_rank].id;
}

starpurm_drs_ret_t starpurm_assign_device_to_starpu(starpurm_drs_desc_t *spd, int type_id, int unit_rank)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return starpurm_DRS_EINVAL;
	if (unit_rank < 0 || unit_rank >= rm->nunits_by_type[type_id])
		return starpurm_DRS_EINVAL;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_or(temp_cpuset, temp_cpuset, rm->units[rm->unit_offsets_by_type[type_id] + unit_rank].worker_cpuset);
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_assign_devices_to_starpu(starpurm_drs_desc_t *spd, int type_id, int ndevices)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return starpurm_DRS_EINVAL;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	if (ndevices > rm->nunits_by_type[type_id])
	{
		ndevices = rm->nunits_by_type[type_id];
	}
	int i;
	for (i = 0; i < ndevices; i++)
	{
		hwloc_bitmap_or(temp_cpuset, temp_cpuset, rm->units[rm->unit_offsets_by_type[type_id] + i].worker_cpuset);
	}
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_assign_device_mask_to_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_or(temp_cpuset, temp_cpuset, mask);
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_assign_all_devices_to_starpu(starpurm_drs_desc_t *spd, int type_id)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return starpurm_DRS_EINVAL;
	return starpurm_assign_devices_to_starpu(spd, type_id, rm->nunits_by_type[type_id]);
}

starpurm_drs_ret_t starpurm_withdraw_device_from_starpu(starpurm_drs_desc_t *spd, int type_id, int unit_rank)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return starpurm_DRS_EINVAL;
	if (unit_rank < 0 || unit_rank >= rm->nunits_by_type[type_id])
		return starpurm_DRS_EINVAL;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_andnot(temp_cpuset, temp_cpuset, rm->units[rm->unit_offsets_by_type[type_id] + unit_rank].worker_cpuset);
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_withdraw_devices_from_starpu(starpurm_drs_desc_t *spd, int type_id, int ndevices)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return starpurm_DRS_EINVAL;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	if (ndevices > rm->nunits_by_type[type_id])
	{
		ndevices = rm->nunits_by_type[type_id];
	}
	int i;
	for (i = 0; i < ndevices; i++)
	{
		hwloc_bitmap_andnot(temp_cpuset, temp_cpuset, rm->units[rm->unit_offsets_by_type[type_id] + i].worker_cpuset);
	}
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_withdraw_device_mask_from_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	(void)spd;
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	hwloc_cpuset_t temp_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_andnot(temp_cpuset, temp_cpuset, mask);
	starpurm_drs_ret_t ret = _starpurm_update_cpuset(temp_cpuset);
	hwloc_bitmap_free(temp_cpuset);
	return ret;
}

starpurm_drs_ret_t starpurm_withdraw_all_devices_from_starpu(starpurm_drs_desc_t *spd, int type_id)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	if (!rm->dynamic_resource_sharing)
		return starpurm_DRS_DISABLD;
	if (type_id < 0 || type_id >= starpurm_unit_ntypes)
		return starpurm_DRS_EINVAL;
	return starpurm_withdraw_devices_from_starpu(spd, type_id, rm->nunits_by_type[type_id]);
}

/* --- */

starpurm_drs_ret_t starpurm_lend_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank)
{
	return starpurm_assign_device_to_starpu(spd, type_id, unit_rank);
}

starpurm_drs_ret_t starpurm_lend_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices)
{
	return starpurm_assign_devices_to_starpu(spd, type_id, ndevices);
}

starpurm_drs_ret_t starpurm_lend_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	return starpurm_assign_device_mask_to_starpu(spd, mask);
}

starpurm_drs_ret_t starpurm_lend_all_devices(starpurm_drs_desc_t *spd, int type_id)
{
	return starpurm_assign_all_devices_to_starpu(spd, type_id);
}


starpurm_drs_ret_t starpurm_reclaim_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank)
{
	return starpurm_withdraw_device_from_starpu(spd, type_id, unit_rank);
}

starpurm_drs_ret_t starpurm_reclaim_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices)
{
	return starpurm_withdraw_devices_from_starpu(spd, type_id, ndevices);
}

starpurm_drs_ret_t starpurm_reclaim_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	return starpurm_withdraw_device_mask_from_starpu(spd, mask);
}

starpurm_drs_ret_t starpurm_reclaim_all_devices(starpurm_drs_desc_t *spd, int type_id)
{
	return starpurm_withdraw_all_devices_from_starpu(spd, type_id);
}

starpurm_drs_ret_t starpurm_acquire_all_devices(starpurm_drs_desc_t *spd, int type_id)
{
	return starpurm_withdraw_all_devices_from_starpu(spd, type_id);
}

starpurm_drs_ret_t starpurm_acquire_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank)
{
	return starpurm_withdraw_device_from_starpu(spd, type_id, unit_rank);
}

starpurm_drs_ret_t starpurm_acquire_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices)
{
	return starpurm_withdraw_devices_from_starpu(spd, type_id, ndevices);
}

starpurm_drs_ret_t starpurm_acquire_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask)
{
	return starpurm_withdraw_device_mask_from_starpu(spd, mask);
}


starpurm_drs_ret_t starpurm_return_all_devices(starpurm_drs_desc_t *spd, int type_id)
{
	return starpurm_assign_all_devices_to_starpu(spd, type_id);
}

starpurm_drs_ret_t starpurm_return_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank)
{
	return starpurm_assign_device_to_starpu(spd, type_id, unit_rank);
}

/* cpusets */
hwloc_cpuset_t starpurm_get_device_worker_cpuset(int type_id, int unit_rank)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	assert(type_id >= 0 && type_id < starpurm_unit_ntypes);
	assert(unit_rank >= 0 && unit_rank < rm->nunits_by_type[type_id]);
	return hwloc_bitmap_dup(rm->units[rm->unit_offsets_by_type[type_id] + unit_rank].worker_cpuset);
}

hwloc_cpuset_t starpurm_get_global_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->global_cpuset);
}

hwloc_cpuset_t starpurm_get_selected_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->selected_cpuset);
}

hwloc_cpuset_t starpurm_get_all_cpu_workers_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->all_cpu_workers_cpuset);
}

static hwloc_cpuset_t starpurm_get_all_opencl_device_workers_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->all_opencl_device_workers_cpuset);
}

static hwloc_cpuset_t starpurm_get_all_cuda_device_workers_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->all_cuda_device_workers_cpuset);
}

static hwloc_cpuset_t starpurm_get_all_mic_device_workers_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->all_mic_device_workers_cpuset);
}

hwloc_cpuset_t starpurm_get_all_device_workers_cpuset(void)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;

	return hwloc_bitmap_dup(rm->all_device_workers_cpuset);
}

hwloc_cpuset_t starpurm_get_all_device_workers_cpuset_by_type(int typeid)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	assert(typeid != starpurm_unit_cpu);
	if (typeid == starpurm_unit_opencl)
		return starpurm_get_all_opencl_device_workers_cpuset();
	if (typeid == starpurm_unit_cuda)
		return starpurm_get_all_cuda_device_workers_cpuset();
	if (typeid == starpurm_unit_mic)
		return starpurm_get_all_mic_device_workers_cpuset();
	hwloc_cpuset_t empty_bitmap = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(empty_bitmap);
	return empty_bitmap;
}
