/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

/* Heteroprio, which sorts tasks by acceleration factor into buckets, and makes
 * GPUs take accelerated tasks first and CPUs take non-accelerated tasks first */

#include <starpu_sched_component.h>
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>

/* Approximation ratio for acceleration factor bucketing
 * We will put tasks with +-10% similar acceleration into the same bucket. */
#define APPROX 0.10

struct _starpu_heteroprio_data
{
	/* This is an array of priority queues.
	 * The array is sorted by acceleration factor, most accelerated first */
	struct _starpu_prio_deque **bucket;
	float *accel;
	unsigned naccel;

	/* This contains tasks which are not supported on all archs. */
	struct _starpu_prio_deque no_accel;

	/* This protects all queues */
	starpu_pthread_mutex_t mutex;

	struct _starpu_mct_data *mct_data;

	unsigned batch;
};

static int heteroprio_progress_accel(struct starpu_sched_component *component, struct _starpu_heteroprio_data *data, enum starpu_worker_archtype archtype, int front)
{
	struct starpu_task *task = NULL;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int j, ret = 1;
	double acceleration = INFINITY;

	struct _starpu_mct_data * d = data->mct_data;

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	if (front)
		/* Pick up accelerated tasks first */
		for (j = 0; j < (int) data->naccel; j++)
		{
			task = _starpu_prio_deque_pop_task(data->bucket[j]);
			if (task)
				break;
		}
	else
		/* Pick up accelerated tasks last */
		for (j = (int) data->naccel-1; j >= 0; j--)
		{
			if (data->batch && 0)
				task = _starpu_prio_deque_pop_back_task(data->bucket[j]);
			else
				task = _starpu_prio_deque_pop_task(data->bucket[j]);
			if (task)
				break;
		}

	if (task)
	{
		acceleration = data->accel[j];
		//fprintf(stderr, "for %s thus %s, found task %p in bucket %d: %f\n", starpu_worker_get_type_as_string(archtype), front?"front":"back", task, j, acceleration);
	}

	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	if (!task)
		return 1;

	if (data->batch)
		/* In batch mode the fifos below do not use priorities. Do not
		 * leak a priority for the data prefetches either */
		task->priority = INT_MAX;

	/* TODO: we might want to prefer to pick up a task whose data is already on some GPU */

	struct starpu_sched_component * best_component;

	/* Estimated task duration for each child */
	double estimated_lengths[component->nchildren];
	/* Estimated transfer duration for each child */
	double estimated_transfer_length[component->nchildren];
	/* Estimated transfer+task termination for each child */
	double estimated_ends_with_task[component->nchildren];

	/* Minimum transfer+task termination on all children */
	double min_exp_end_with_task;
	/* Maximum transfer+task termination on all children */
	double max_exp_end_with_task;

	unsigned suitable_components[component->nchildren];
	unsigned nsuitable_components;

	nsuitable_components = starpu_mct_compute_execution_times(component, task,
			estimated_lengths,
			estimated_transfer_length,
			suitable_components);

	if (data->batch && 0)
	{
		/* In batch mode, we may want to insist on filling workers with tasks
		 * by ignoring when other workers would finish this. */

		unsigned i;
		for (i = 0; i < component->nchildren; i++)
		{
			int idworker;
			for(idworker = starpu_bitmap_first(component->children[i]->workers);
				idworker != -1;
				idworker = starpu_bitmap_next(component->children[i]->workers, idworker))
			{
				if (starpu_worker_get_type(idworker) == archtype)
					break;
			}

			if (idworker == -1)
			{
				/* Not the targetted arch, avoid it */

				/* XXX: INFINITY doesn't seem to be working properly */
				estimated_lengths[i] = 1000000000;
				estimated_transfer_length[i] = 1000000000;
			}
		}
	}

	/* Entering critical section to make sure no two workers
	   make scheduling decisions at the same time */
	STARPU_COMPONENT_MUTEX_LOCK(&d->scheduling_mutex);

	starpu_mct_compute_expected_times(component, task,
			estimated_lengths,
			estimated_transfer_length,
			estimated_ends_with_task,
			&min_exp_end_with_task, &max_exp_end_with_task,
			suitable_components, nsuitable_components);

	/* And now find out which worker suits best for this task,
	 * including data transfer */
	int best_icomponent = starpu_mct_get_best_component(d, task,
			estimated_lengths,
			estimated_transfer_length,
			estimated_ends_with_task,
			min_exp_end_with_task, max_exp_end_with_task,
			suitable_components, nsuitable_components);

	if (best_icomponent == -1)
		goto out;

	best_component = component->children[best_icomponent];

	int idworker;
	for(idworker = starpu_bitmap_first(best_component->workers);
		idworker != -1;
		idworker = starpu_bitmap_next(best_component->workers, idworker))
	{
		if (starpu_worker_get_type(idworker) == archtype)
			break;
	}

	if (idworker == -1)
		goto out;

	/* Ok, we do have a worker there of that type, try to push it there. */
	STARPU_ASSERT(!starpu_sched_component_is_worker(best_component));
	starpu_sched_task_break(task);
	ret = starpu_sched_component_push_task(component,best_component,task);

	/* I can now exit the critical section: Pushing the task above ensures that its execution
	   time will be taken into account for subsequent scheduling decisions */
	if (!ret)
	{
		STARPU_COMPONENT_MUTEX_UNLOCK(&d->scheduling_mutex);
		//fprintf(stderr, "pushed %p to %d\n", task, best_icomponent);
		/* Great! */
		return 0;
	}

out:
	STARPU_COMPONENT_MUTEX_UNLOCK(&d->scheduling_mutex);
	/* No such kind of worker there, or it refused our task, abort */

	//fprintf(stderr, "could not push %p to %d actually\n", task, best_icomponent);
	/* Could not push to child actually, push that one back */
	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	for (j = 0; j < (int) data->naccel; j++)
	{
		if (acceleration == data->accel[j])
		{
			_starpu_prio_deque_push_front_task(data->bucket[j], task);
			break;
		}
	}
	STARPU_ASSERT(j != (int) data->naccel);
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	//fprintf(stderr, "finished pushing to %d\n", archtype);

	return 1;
}

static int heteroprio_progress_noaccel(struct starpu_sched_component *component, struct _starpu_heteroprio_data *data, struct starpu_task *task)
{
	struct _starpu_mct_data * d = data->mct_data;
	int ret;

	struct starpu_sched_component * best_component;

	/* Estimated task duration for each child */
	double estimated_lengths[component->nchildren];
	/* Estimated transfer duration for each child */
	double estimated_transfer_length[component->nchildren];
	/* Estimated transfer+task termination for each child */
	double estimated_ends_with_task[component->nchildren];

	/* Minimum transfer+task termination on all children */
	double min_exp_end_with_task;
	/* Maximum transfer+task termination on all children */
	double max_exp_end_with_task;

	unsigned suitable_components[component->nchildren];
	unsigned nsuitable_components;

	nsuitable_components = starpu_mct_compute_execution_times(component, task,
			estimated_lengths,
			estimated_transfer_length,
			suitable_components);

	/* If no suitable components were found, it means that the perfmodel of
	 * the task had been purged since it has been pushed on the mct component.
	 * We should send a push_fail message to its parent so that it will
	 * be able to reschedule the task properly. */
	if(nsuitable_components == 0)
		return 1;

	/* Entering critical section to make sure no two workers
	   make scheduling decisions at the same time */
	STARPU_COMPONENT_MUTEX_LOCK(&d->scheduling_mutex);

	starpu_mct_compute_expected_times(component, task,
			estimated_lengths,
			estimated_transfer_length,
			estimated_ends_with_task,
			&min_exp_end_with_task, &max_exp_end_with_task,
			suitable_components, nsuitable_components);

	/* And now find out which worker suits best for this task,
	 * including data transfer */
	int best_icomponent = starpu_mct_get_best_component(d, task,
			estimated_lengths,
			estimated_transfer_length,
			estimated_ends_with_task,
			min_exp_end_with_task, max_exp_end_with_task,
			suitable_components, nsuitable_components);

	/* If no best component is found, it means that the perfmodel of
	 * the task had been purged since it has been pushed on the mct component.
	 * We should send a push_fail message to its parent so that it will
	 * be able to reschedule the task properly. */
	if(best_icomponent == -1)
	{
		STARPU_COMPONENT_MUTEX_UNLOCK(&d->scheduling_mutex);
		return 1;
	}

	best_component = component->children[best_icomponent];

	STARPU_ASSERT(!starpu_sched_component_is_worker(best_component));
	ret = starpu_sched_component_push_task(component,best_component,task);
	STARPU_COMPONENT_MUTEX_UNLOCK(&d->scheduling_mutex);

	return ret;
}

static int heteroprio_progress_one(struct starpu_sched_component *component)
{
	struct _starpu_heteroprio_data * data = component->data;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	struct starpu_task *task;

	struct _starpu_prio_deque * no_accel = &data->no_accel;
	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	task = _starpu_prio_deque_pop_task(no_accel);
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	if (task)
	{
		if (heteroprio_progress_noaccel(component, data, task))
		{
			/* Could not push to child actually, push that one back */
			STARPU_COMPONENT_MUTEX_LOCK(mutex);
			_starpu_prio_deque_push_front_task(no_accel, task);
			STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
		}
	}

	/* Note: this hardcodes acceleration order */
	if (!heteroprio_progress_accel(component, data, STARPU_CUDA_WORKER, 1))
		return 0;
	if (!heteroprio_progress_accel(component, data, STARPU_OPENCL_WORKER, 1))
		return 0;
	if (!heteroprio_progress_accel(component, data, STARPU_MIC_WORKER, 1))
		return 0;
	if (!heteroprio_progress_accel(component, data, STARPU_MPI_MS_WORKER, 0))
		return 0;
	if (!heteroprio_progress_accel(component, data, STARPU_CPU_WORKER, 0))
		return 0;

	return 1;
}

/* Try to push some tasks below */
static void heteroprio_progress(struct starpu_sched_component *component)
{
	STARPU_ASSERT(component && starpu_sched_component_is_heteroprio(component));
	while (!heteroprio_progress_one(component))
		;
}

static int heteroprio_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component && task && starpu_sched_component_is_heteroprio(component));
	struct _starpu_heteroprio_data * data = component->data;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	unsigned nimpl;

	double min_expected = INFINITY, max_expected = -INFINITY;
	double acceleration;

	if (data->batch && 0)
		/* Batch mode, we may want to ignore priorities completely */
		task->priority = INT_MAX;

	/* Compute acceleration between best-performing arch and least-performing arch */
	int workerid;
	for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	{
		unsigned impl_mask;
		if (!starpu_worker_can_execute_task_impl(workerid, task, &impl_mask))
			break;

		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerid, task->sched_ctx);
		double min_arch = INFINITY;
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!(impl_mask & (1U << nimpl)))
				continue;
			double expected = starpu_task_expected_length(task, perf_arch, nimpl);
			if (isnan(expected) || expected == 0.)
			{
				min_arch = expected;
				break;
			}
			if (expected < min_arch)
				min_arch = expected;
		}
		if (isnan(min_arch) || min_arch == 0.)
			/* No known execution time, can't do anything here */
			break;

		STARPU_ASSERT(min_arch != INFINITY);
		if (min_arch < min_expected)
			min_expected = min_arch;
		if (min_arch > max_expected)
			max_expected = min_arch;
	}

	if (workerid == -1)
	{
		/* All archs can run it */
		STARPU_ASSERT(!isnan(min_expected));
		STARPU_ASSERT(!isnan(max_expected));
		STARPU_ASSERT(min_expected != INFINITY);
		STARPU_ASSERT(max_expected != -INFINITY);
		acceleration = max_expected / min_expected;
		STARPU_ASSERT(!isnan(acceleration));

		//fprintf(stderr,"%s: acceleration %f\n", starpu_task_get_name(task), acceleration);

		STARPU_COMPONENT_MUTEX_LOCK(mutex);
		unsigned i, j;
		/* Try to find a bucket with similar acceleration */
		for (i = 0; i < data->naccel; i++)
		{
			if (acceleration >= data->accel[i] * (1 - APPROX) &&
			    acceleration <= data->accel[i] * (1 + APPROX))
				break;
		}

		if (i == data->naccel)
		{
			/* Didn't find it, add one */
			data->naccel++;

			float *newaccel = malloc(data->naccel * sizeof(*newaccel));
			struct _starpu_prio_deque **newbuckets = malloc(data->naccel * sizeof(*newbuckets));
			struct _starpu_prio_deque *newbucket = malloc(sizeof(*newbucket));
			_starpu_prio_deque_init(newbucket);
			int inserted = 0;

			for (j = 0; j < data->naccel-1; j++)
			{
				if (!inserted && acceleration > data->accel[j])
				{
					/* Insert the new bucket here */
					i = j;
					newbuckets[j] = newbucket;
					newaccel[j] = acceleration;
					inserted = 1;
				}
				newbuckets[j+inserted] = data->bucket[j];
				newaccel[j+inserted] = data->accel[j];
			}
			if (!inserted)
			{
				/* Insert it last */
				newbuckets[data->naccel-1] = newbucket;
				newaccel[data->naccel-1] = acceleration;
			}
			free(data->bucket);
			free(data->accel);
			data->bucket = newbuckets;
			data->accel = newaccel;
		}
#if 0
		fprintf(stderr,"buckets:");
		for (j = 0; j < data->naccel; j++)
		{
			fprintf(stderr, " %f", data->accel[j]);
		}
		fprintf(stderr,"\ninserting %p %f to %d\n", task, acceleration, i);
#endif
		_starpu_prio_deque_push_back_task(data->bucket[i],task);
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
	}
	else
	{
		/* Not all archs can run it, will resort to HEFT strategy */
		acceleration = INFINITY;
		//fprintf(stderr,"%s: some archs can't do it\n", starpu_task_get_name(task));
		struct _starpu_prio_deque * no_accel = &data->no_accel;
		STARPU_COMPONENT_MUTEX_LOCK(mutex);
		_starpu_prio_deque_push_back_task(no_accel,task);
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
	}

	heteroprio_progress(component);

	return 0;
}

static int heteroprio_can_push(struct starpu_sched_component *component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	heteroprio_progress(component);
	int ret = 0;
	unsigned j;
	for(j=0; j < component->nparents; j++)
	{
		if(component->parents[j] == NULL)
			continue;
		else
		{
			ret = component->parents[j]->can_push(component->parents[j], component);
			if(ret)
				break;
		}
	}
	return ret;
}

static void heteroprio_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(starpu_sched_component_is_heteroprio(component));
	struct _starpu_heteroprio_data * d = component->data;
	struct _starpu_mct_data * mct_d = d->mct_data;
	unsigned i;
	for (i = 0; i < d->naccel; i++)
	{
		_starpu_prio_deque_destroy(d->bucket[i]);
		free(d->bucket[i]);
	}
	free(d->bucket);
	free(d->accel);
	_starpu_prio_deque_destroy(&d->no_accel);
	STARPU_PTHREAD_MUTEX_DESTROY(&d->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&mct_d->scheduling_mutex);
	free(mct_d);
	free(d);
}

int starpu_sched_component_is_heteroprio(struct starpu_sched_component * component)
{
	return component->push_task == heteroprio_push_task;
}

struct starpu_sched_component * starpu_sched_component_heteroprio_create(struct starpu_sched_tree *tree, struct starpu_sched_component_heteroprio_data * params)
{
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "heteroprio");
	struct _starpu_mct_data *mct_data = starpu_mct_init_parameters(params ? params->mct : NULL);
	struct _starpu_heteroprio_data *data;
	_STARPU_MALLOC(data, sizeof(*data));

	data->bucket = NULL;
	data->accel = NULL;
	data->naccel = 0;
	_starpu_prio_deque_init(&data->no_accel);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	data->mct_data = mct_data;
	STARPU_PTHREAD_MUTEX_INIT(&mct_data->scheduling_mutex,NULL);
	if (params)
		data->batch = params->batch;
	else
		data->batch = 1;
	component->data = data;

	component->push_task = heteroprio_push_task;
	component->can_push = heteroprio_can_push;
	component->deinit_data = heteroprio_component_deinit_data;

	return component;
}
