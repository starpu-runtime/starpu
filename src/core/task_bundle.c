/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
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
#include <starpu_task_bundle.h>
#include <starpu_scheduler.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/list.h>

/* Initialize a task bundle */
void starpu_task_bundle_init(struct starpu_task_bundle *bundle)
{
	STARPU_ASSERT(bundle);

	PTHREAD_MUTEX_INIT(&bundle->mutex, NULL);
	bundle->closed = 0;

	/* Start with an empty list */
	bundle->previous_workerid = -1;
	bundle->list = NULL;

	/* By default, bundle are destroyed */
	bundle->destroy = 1;

}

/* Deinitialize a bundle. In case the destroy flag is set, the bundle structure
 * is freed too. */
void starpu_task_bundle_deinit(struct starpu_task_bundle *bundle)
{
	/* Remove all entries from the bundle (which is likely to be empty) */
	while (bundle->list)
	{
		struct starpu_task_bundle_entry *entry = bundle->list;
		bundle->list = bundle->list->next;
		free(entry);
	}

	PTHREAD_MUTEX_DESTROY(&bundle->mutex);

	if (bundle->destroy)
		free(bundle);
}

/* Insert a task into a bundle. */
int starpu_task_bundle_insert(struct starpu_task_bundle *bundle, struct starpu_task *task)
{
	PTHREAD_MUTEX_LOCK(&bundle->mutex);

	if (bundle->closed)
	{
		/* The bundle is closed, we cannot add tasks anymore */
		PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		return -EPERM;
	}

	if (task->status != STARPU_TASK_INVALID)
	{
		/* the task has already been submitted, it's too late to put it
		 * into a bundle now. */
		PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		return -EINVAL;
	}

	/* Insert a task at the end of the bundle */
	struct starpu_task_bundle_entry *entry;
	entry = (struct starpu_task_bundle_entry *) malloc(sizeof(struct starpu_task_bundle_entry));
	STARPU_ASSERT(entry);
	entry->task = task;
	entry->next = NULL;

	if (!bundle->list)
	{
		bundle->list = entry;
	}
	else {
		struct starpu_task_bundle_entry *item;
		item = bundle->list;
		while (item->next)
			item = item->next;

		item->next = entry;
	}

	task->bundle = bundle;

	PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
	return 0;
}

/* Remove a task from a bundle. This method must be called with bundle->mutex
 * hold. This function returns 0 if the task was found, -ENOENT if the element
 * was not found, 1 if the element is found and if the list was deinitialized
 * because it became empty. */
int starpu_task_bundle_remove(struct starpu_task_bundle *bundle, struct starpu_task *task)
{
	struct starpu_task_bundle_entry *item;

	item = bundle->list;

	if (!item)
		return -ENOENT;

	STARPU_ASSERT(task->bundle == bundle);
	task->bundle = NULL;

	if (item->task == task)
	{
		/* Remove the first element */
		bundle->list = item->next;
		free(item);

		/* If the list is now empty, deinitialize the bundle */
		if (bundle->closed && bundle->list == NULL)
		{
			PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
			starpu_task_bundle_deinit(bundle);
			return 1;
		}

		return 0;
	}

	while (item->next)
	{
		struct starpu_task_bundle_entry *next;
		next = item->next;

		if (next->task == task)
		{
			/* Remove the next element */
			item->next = next->next;
			free(next);
			return 0;
		}

		item = next;
	}

	/* We could not find the task in the bundle */
	return -ENOENT;
}

/* Close a bundle. No task can be added to a closed bundle. A closed bundle
 * automatically gets deinitialized when it becomes empty. */
void starpu_task_bundle_close(struct starpu_task_bundle *bundle)
{
	PTHREAD_MUTEX_LOCK(&bundle->mutex);

	/* If the bundle is already empty, we deinitialize it now. */
	if (bundle->list == NULL)
	{
		PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		starpu_task_bundle_deinit(bundle);
		return;
	}

	/* Mark the bundle as closed */
	bundle->closed = 1;

	PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

}

/* Return the expected duration of the entire task bundle in µs */
double starpu_task_bundle_expected_length(struct starpu_task_bundle *bundle,  enum starpu_perf_archtype arch, unsigned nimpl)
{
	double expected_length = 0.0;

	/* We expect the length of the bundle the be the sum of the different tasks length. */
	PTHREAD_MUTEX_LOCK(&bundle->mutex);

	struct starpu_task_bundle_entry *entry;
	entry = bundle->list;

	while (entry) {
		double task_length = starpu_task_expected_length(entry->task, arch, nimpl);

		/* In case the task is not calibrated, we consider the task
		 * ends immediately. */
		if (task_length > 0.0)
			expected_length += task_length;

		entry = entry->next;
	}
	
	PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	return expected_length;
}

/* Return the expected power consumption of the entire task bundle in J */
double starpu_task_bundle_expected_power(struct starpu_task_bundle *bundle,  enum starpu_perf_archtype arch, unsigned nimpl)
{
	double expected_power = 0.0;

	/* We expect total consumption of the bundle the be the sum of the different tasks consumption. */
	PTHREAD_MUTEX_LOCK(&bundle->mutex);

	struct starpu_task_bundle_entry *entry;
	entry = bundle->list;

	while (entry) {
		double task_power = starpu_task_expected_power(entry->task, arch, nimpl);

		/* In case the task is not calibrated, we consider the task
		 * ends immediately. */
		if (task_power > 0.0)
			expected_power += task_power;

		entry = entry->next;
	}
	
	PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	return expected_power;
}

struct handle_list {
	starpu_data_handle handle;
	starpu_access_mode mode;
	struct handle_list *next;
};

static void insertion_handle_sorted(struct handle_list **listp, starpu_data_handle handle, starpu_access_mode mode)
{
	STARPU_ASSERT(listp);

	struct handle_list *list = *listp;

	if (!list || list->handle > handle)
	{
		/* We insert the first element of the list */
		struct handle_list *link = (struct handle_list *) malloc(sizeof(struct handle_list));
		STARPU_ASSERT(link);
		link->handle = handle;
		link->mode = mode;
		link->next = list;
		*listp = link;
		return;
	}

	/* Look for the element or a place to insert it. */
	struct handle_list *prev = list;

	while (list && (handle > list->handle))
	{
		prev = list;
		list = list->next;
	}

	/* The element should be in prev or not in the list */

	if (prev->handle == handle)
	{
		/* The handle is already in the list */
		prev->mode |= mode;
	}
	else {
		/* The handle was not in the list, we insert it after prev */
		struct handle_list *link = (struct handle_list *) malloc(sizeof(struct handle_list));
		STARPU_ASSERT(link);
		link->handle = handle;
		link->mode = mode;
		link->next = prev->next;
		prev->next = link;
	}
}

/* Return the time (in µs) expected to transfer all data used within the bundle */
double starpu_task_bundle_expected_data_transfer_time(struct starpu_task_bundle *bundle, unsigned memory_node)
{
	PTHREAD_MUTEX_LOCK(&bundle->mutex);

	struct handle_list *handles = NULL;

	/* We list all the handle that are accessed within the bundle. */

	/* For each task in the bundle */
	struct starpu_task_bundle_entry *entry = bundle->list;
	while (entry) {
		struct starpu_task *task = entry->task;

		if (task->cl)
		{
			unsigned b;
			for (b = 0; b < task->cl->nbuffers; b++)
			{
				starpu_data_handle handle = task->buffers[b].handle;
				starpu_access_mode mode = task->buffers[b].mode;

				if (!(mode & STARPU_R))
					continue;

				/* Insert the handle in the sorted list in case
				 * it's not already in that list. */
				insertion_handle_sorted(&handles, handle, mode);
			}
		}

		entry = entry->next;
	}

	/* Compute the sum of data transfer time, and destroy the list */

	double total_exp = 0.0;

	while (handles)
	{
		struct handle_list *current = handles;
		handles = handles->next;

		double exp;
		exp = starpu_data_expected_transfer_time(current->handle, memory_node, current->mode);

		total_exp += exp;

		free(current);
	}

	PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	return total_exp;
}
