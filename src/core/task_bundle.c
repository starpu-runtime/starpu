/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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
#include <core/task_bundle.h>
#include <starpu_scheduler.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/list.h>
#include <common/thread.h>

/* Initialize a task bundle */
void starpu_task_bundle_create(starpu_task_bundle_t *bundle)
{
	_STARPU_MALLOC(*bundle, sizeof(struct _starpu_task_bundle));

	STARPU_PTHREAD_MUTEX_INIT(&(*bundle)->mutex, NULL);
	/* Of course at the beginning a bundle is open,
	 * user can insert and remove tasks from it */
	(*bundle)->closed = 0;

	/* Start with an empty list */
	(*bundle)->list = NULL;

}

int starpu_task_bundle_insert(starpu_task_bundle_t bundle, struct starpu_task *task)
{
	STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

	if (bundle->closed)
	{
		/* The bundle is closed, we cannot add task anymore */
		STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		return -EPERM;
	}

	if (task->status != STARPU_TASK_INIT)
	{
		/* The task has already been submitted, it's too late to put it
		 * into a bundle now. */
		STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		return -EINVAL;
	}

	/* Insert a task at the end of the bundle */
	struct _starpu_task_bundle_entry *entry;
	_STARPU_MALLOC(entry, sizeof(struct _starpu_task_bundle_entry));
	entry->task = task;
	entry->next = NULL;

	if (!bundle->list)
	{
		bundle->list = entry;
	}
	else
	{
		struct _starpu_task_bundle_entry *item;
		item = bundle->list;
		while (item->next)
			item = item->next;

		item->next = entry;
	}

	/* Mark the task as belonging the bundle */
	task->bundle = bundle;

	STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
	return 0;
}

int starpu_task_bundle_remove(starpu_task_bundle_t bundle, struct starpu_task *task)
{
	struct _starpu_task_bundle_entry *item;

	STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

	item = bundle->list;

	/* List is empty, there is no way the task
	 * belong to it */
	if (!item)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		return -ENOENT;
	}

	STARPU_ASSERT_MSG(task->bundle == bundle, "Task %p was not in bundle %p, but in bundle %p", task, bundle, task->bundle);
	task->bundle = NULL;

	if (item->task == task)
	{
		/* Remove the first element */
		bundle->list = item->next;
		free(item);

		/* If the list is now empty, deinitialize the bundle */
		if (bundle->closed && bundle->list == NULL)
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
			_starpu_task_bundle_destroy(bundle);
			return 0;
		}

		STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		return 0;
	}

	/* Go through the list until we find the right task,
	 * then we delete it */
	while (item->next)
	{
		struct _starpu_task_bundle_entry *next;
		next = item->next;

		if (next->task == task)
		{
			/* Remove the next element */
			item->next = next->next;
			STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
			free(next);
			return 0;
		}

		item = next;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	/* We could not find the task in the bundle */
	return -ENOENT;
}

/* Close a bundle. No task can be added to a closed bundle. Tasks can still be
 * removed from a closed bundle. A closed bundle automatically gets
 * deinitialized when it becomes empty. A closed bundle cannot be reopened. */
void starpu_task_bundle_close(starpu_task_bundle_t bundle)
{
	STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

	/* If the bundle is already empty, we deinitialize it now as the
	 * user closed it and thus don't intend to insert new tasks in it. */
	if (bundle->list == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
		_starpu_task_bundle_destroy(bundle);
		return;
	}

	/* Mark the bundle as closed */
	bundle->closed = 1;

	STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

}

void _starpu_task_bundle_destroy(starpu_task_bundle_t bundle)
{
	/* Remove all entries from the bundle (which is likely to be empty) */
	while (bundle->list)
	{
		struct _starpu_task_bundle_entry *entry = bundle->list;
		bundle->list = bundle->list->next;
		free(entry);
	}

	STARPU_PTHREAD_MUTEX_DESTROY(&bundle->mutex);

	free(bundle);
}

void _insertion_handle_sorted(struct _starpu_handle_list **listp, starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	STARPU_ASSERT(listp);

	struct _starpu_handle_list *list = *listp;

	/* If the list is empty or the handle's address the smallest among the
	 * list, we insert it as first element */
	if (!list || list->handle > handle)
	{
		struct _starpu_handle_list *link;
		_STARPU_MALLOC(link, sizeof(struct _starpu_handle_list));
		link->handle = handle;
		link->mode = mode;
		link->next = list;
		*listp = link;
		return;
	}

	struct _starpu_handle_list *prev = list;

	/* Look for the same handle if already present in the list.
	 * Else place it right before the smallest following handle */
	while (list && (handle >= list->handle))
	{
		prev = list;
		list = list->next;
	}

	if (prev->handle == handle)
	{
		/* The handle is already in the list, the merge both the access modes */
		prev->mode = (enum starpu_data_access_mode) ((int) prev->mode | (int) mode);
	}
	else
	{
		/* The handle was not in the list, we insert it after 'prev', thus right before
		 * 'list' which is the smallest following handle */
		struct _starpu_handle_list *link;
		_STARPU_MALLOC(link, sizeof(struct _starpu_handle_list));
		link->handle = handle;
		link->mode = mode;
		link->next = prev->next;
		prev->next = link;
	}
}
