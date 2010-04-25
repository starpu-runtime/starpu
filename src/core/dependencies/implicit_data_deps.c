/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <starpu.h>
#include <common/config.h>
#include <datawizard/datawizard.h>

static void _starpu_detect_implicit_data_deps_with_handle(struct starpu_task *task, starpu_data_handle handle, starpu_access_mode mode)
{
	PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);

	if (handle->sequential_consistency)
	{
		starpu_access_mode previous_mode = handle->last_submitted_mode;
	
		if (mode != STARPU_R)
		{
			if (previous_mode != STARPU_R)
			{
				/* (Read) Write */
				/* This task depends on the previous writer */
				if (handle->last_submitted_writer)
				{
					struct starpu_task *task_array[1] = {handle->last_submitted_writer};
					starpu_task_declare_deps_array(task, 1, task_array);
				}
	
				handle->last_submitted_writer = task;
			}
			else {
				/* The task submitted previously were in read-only
				 * mode: this task must depend on all those read-only
				 * tasks and we get rid of the list of readers */
			
				/* Count the readers */
				unsigned nreaders = 0;
				struct starpu_task_list *l;
				l = handle->last_submitted_readers;
				while (l)
				{
					nreaders++;
					l = l->next;
				}
	
				struct starpu_task *task_array[nreaders];
				unsigned i = 0;
				l = handle->last_submitted_readers;
				while (l)
				{
					STARPU_ASSERT(l->task);
					task_array[i++] = l->task;

					struct starpu_task_list *prev = l;
					l = l->next;
					free(prev);
				}
	
				handle->last_submitted_readers = NULL;
				handle->last_submitted_writer = task;
	
				starpu_task_declare_deps_array(task, nreaders, task_array);
			}
	
		}
		else {
			/* Add a reader */
			STARPU_ASSERT(task);
	
			/* Add this task to the list of readers */
			struct starpu_task_list *link = malloc(sizeof(struct starpu_task_list));
			link->task = task;
			link->next = handle->last_submitted_readers;
			handle->last_submitted_readers = link;


			/* This task depends on the previous writer if any */
			if (handle->last_submitted_writer)
			{
				struct starpu_task *task_array[1] = {handle->last_submitted_writer};
				starpu_task_declare_deps_array(task, 1, task_array);
			}
		}
	
		handle->last_submitted_mode = mode;
	}

	PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
}

/* Create the implicit dependencies for a newly submitted task */
void _starpu_detect_implicit_data_deps(struct starpu_task *task)
{
	if (!task->cl)
		return;

	unsigned nbuffers = task->cl->nbuffers;

	unsigned buffer;
	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle handle = task->buffers[buffer].handle;
		starpu_access_mode mode = task->buffers[buffer].mode;

		_starpu_detect_implicit_data_deps_with_handle(task, handle, mode);
	}
}
