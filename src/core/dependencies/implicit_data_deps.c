/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <core/task.h>
#include <datawizard/datawizard.h>

#if 0
# define _STARPU_DEP_DEBUG(fmt, args ...) fprintf(stderr, fmt, ##args);
#else
# define _STARPU_DEP_DEBUG(fmt, args ...)
#endif

/* This function adds the implicit task dependencies introduced by data
 * sequential consistency. Two tasks are provided: pre_sync and post_sync which
 * respectively indicates which task is going to depend on the previous deps
 * and on which task future deps should wait. In the case of a dependency
 * introduced by a task submission, both tasks are just the submitted task, but
 * in the case of user interactions with the DSM, these may be different tasks.
 * */
/* NB : handle->sequential_consistency_mutex must be hold by the caller */
void _starpu_detect_implicit_data_deps_with_handle(struct starpu_task *pre_sync_task, struct starpu_task *post_sync_task,
						starpu_data_handle handle, starpu_access_mode mode)
{
	STARPU_ASSERT(!(mode & STARPU_SCRATCH));

	if (handle->sequential_consistency)
	{
#ifdef STARPU_USE_FXT
		/* In case we are generating the DAG, we add an implicit
		 * dependency between the pre and the post sync tasks in case
		 * they are not the same. */
		if (pre_sync_task != post_sync_task)
		{
			starpu_job_t pre_sync_job = _starpu_get_job_associated_to_task(pre_sync_task);
			starpu_job_t post_sync_job = _starpu_get_job_associated_to_task(post_sync_task);
			STARPU_TRACE_GHOST_TASK_DEPS(pre_sync_job->job_id, post_sync_job->job_id);
		}
#endif

		starpu_access_mode previous_mode = handle->last_submitted_mode;
	
		if (mode & STARPU_W)
		{
			_STARPU_DEP_DEBUG("W %p\n", handle);
			if (previous_mode & STARPU_W)
			{
				_STARPU_DEP_DEBUG("WAW %p\n", handle);
				/* (Read) Write */
				/* This task depends on the previous writer */
				if (handle->last_submitted_writer)
				{
					struct starpu_task *task_array[1] = {handle->last_submitted_writer};
					starpu_task_declare_deps_array(pre_sync_task, 1, task_array);
				}

#ifdef STARPU_USE_FXT
				/* If there is a ghost writer instead, we
				 * should declare a ghost dependency here, and
				 * invalidate the ghost value. */
				if (handle->last_submitted_ghost_writer_id_is_valid)
				{
					starpu_job_t post_sync_job = _starpu_get_job_associated_to_task(post_sync_task);
					STARPU_TRACE_GHOST_TASK_DEPS(handle->last_submitted_ghost_writer_id, post_sync_job->job_id);
					handle->last_submitted_ghost_writer_id_is_valid = 0;
				}
#endif
	
				handle->last_submitted_writer = post_sync_task;
			}
			else {
				/* The task submitted previously were in read-only
				 * mode: this task must depend on all those read-only
				 * tasks and we get rid of the list of readers */
			
				_STARPU_DEP_DEBUG("WAR %p\n", handle);
				/* Count the readers */
				unsigned nreaders = 0;
				struct starpu_task_wrapper_list *l;
				l = handle->last_submitted_readers;
				while (l)
				{
					nreaders++;
					l = l->next;
				}
				_STARPU_DEP_DEBUG("%d readers\n", nreaders);

				struct starpu_task *task_array[nreaders];

				unsigned i = 0;
				l = handle->last_submitted_readers;
				while (l)
				{
					STARPU_ASSERT(l->task);
					task_array[i++] = l->task;

					struct starpu_task_wrapper_list *prev = l;
					l = l->next;
					free(prev);
				}
#ifdef STARPU_USE_FXT
				/* Declare all dependencies with ghost readers */
				starpu_job_t post_sync_job = _starpu_get_job_associated_to_task(post_sync_task);

				struct starpu_jobid_list *ghost_readers_id = handle->last_submitted_ghost_readers_id;
				while (ghost_readers_id)
				{
					unsigned long id = ghost_readers_id->id;
					STARPU_TRACE_GHOST_TASK_DEPS(id, post_sync_job->job_id);

					struct starpu_jobid_list *prev = ghost_readers_id;
					ghost_readers_id = ghost_readers_id->next;
					free(prev);
				}
				handle->last_submitted_ghost_readers_id = NULL;
#endif

				handle->last_submitted_readers = NULL;
				handle->last_submitted_writer = post_sync_task;
	
				starpu_task_declare_deps_array(pre_sync_task, nreaders, task_array);
			}
	
		}
		else {
			_STARPU_DEP_DEBUG("R %p\n", handle);
			/* Add a reader */
			STARPU_ASSERT(pre_sync_task);
			STARPU_ASSERT(post_sync_task);
	
			/* Add this task to the list of readers */
			struct starpu_task_wrapper_list *link = malloc(sizeof(struct starpu_task_wrapper_list));
			link->task = post_sync_task;
			link->next = handle->last_submitted_readers;
			handle->last_submitted_readers = link;

			/* This task depends on the previous writer if any */
			if (handle->last_submitted_writer)
			{
				_STARPU_DEP_DEBUG("RAW %p\n", handle);
				struct starpu_task *task_array[1] = {handle->last_submitted_writer};
				starpu_task_declare_deps_array(pre_sync_task, 1, task_array);
			}

#ifdef STARPU_USE_FXT
			/* There was perhaps no last submitted writer but a
			 * ghost one, we should report that here, and keep the
			 * ghost writer valid */
			if (handle->last_submitted_ghost_writer_id_is_valid)
			{
				starpu_job_t post_sync_job = _starpu_get_job_associated_to_task(post_sync_task);
				STARPU_TRACE_GHOST_TASK_DEPS(handle->last_submitted_ghost_writer_id, post_sync_job->job_id);
			}
#endif
		}
	
		handle->last_submitted_mode = mode;
	}
}

/* Create the implicit dependencies for a newly submitted task */
void _starpu_detect_implicit_data_deps(struct starpu_task *task)
{
	STARPU_ASSERT(task->cl);

	unsigned nbuffers = task->cl->nbuffers;

	unsigned buffer;
	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle handle = task->buffers[buffer].handle;
		starpu_access_mode mode = task->buffers[buffer].mode;

		/* Scratch memory does not introduce any deps */
		if (mode & STARPU_SCRATCH)
			continue;

		PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
		_starpu_detect_implicit_data_deps_with_handle(task, task, handle, mode);
		PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
	}
}

/* This function is called when a task has been executed so that we don't
 * create dependencies to task that do not exist anymore. */
/* NB: We maintain a list of "ghost deps" in case FXT is enabled. Ghost
 * dependencies are the dependencies that are implicitely enforced by StarPU
 * even if they do not imply a real dependency. For instance in the following
 * sequence, f(Ar) g(Ar) h(Aw), we expect to have h depend on both f and g, but
 * if h is submitted after the termination of f or g, StarPU will not create a
 * dependency as this is not needed anymore. */
void _starpu_release_data_enforce_sequential_consistency(struct starpu_task *task, starpu_data_handle handle)
{
	PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);

	if (handle->sequential_consistency)
	{

		/* If this is the last writer, there is no point in adding
		 * extra deps to that tasks that does not exists anymore */
		if (task == handle->last_submitted_writer)
		{
			handle->last_submitted_writer = NULL;
			
#ifdef STARPU_USE_FXT
			/* Save the previous writer as the ghost last writer */
			handle->last_submitted_ghost_writer_id_is_valid = 1;
			starpu_job_t ghost_job = _starpu_get_job_associated_to_task(task);
			handle->last_submitted_ghost_writer_id = ghost_job->job_id;
#endif
			
		}
		
		/* XXX can a task be both the last writer associated to a data
		 * and be in its list of readers ? If not, we should not go
		 * through the entire list once we have detected it was the
		 * last writer. */

		/* Same if this is one of the readers: we go through the list
		 * of readers and remove the task if it is found. */
		struct starpu_task_wrapper_list *l;
		l = handle->last_submitted_readers;
		struct starpu_task_wrapper_list *prev = NULL;
		while (l)
		{
			struct starpu_task_wrapper_list *next = l->next;

			if (l->task == task)
			{
				/* If we found the task in the reader list */
				free(l);

#ifdef STARPU_USE_FXT
				/* Save the job id of the reader task in the ghost reader linked list list */
				starpu_job_t ghost_reader_job = _starpu_get_job_associated_to_task(task);
				struct starpu_jobid_list *link = malloc(sizeof(struct starpu_jobid_list));
				STARPU_ASSERT(link);
				link->next = handle->last_submitted_ghost_readers_id;
				link->id = ghost_reader_job->job_id; 
				handle->last_submitted_ghost_readers_id = link;
#endif

				if (prev)
				{
					prev->next = next;
				}
				else {
					/* This is the first element of the list */
					handle->last_submitted_readers = next;
				}

				/* XXX can we really find the same task again
				 * once we have found it ? Otherwise, we should
				 * avoid going through the entire list and stop
				 * as soon as we find the task. TODO: check how
				 * duplicate dependencies are treated. */
			}
			else {
				prev = l;
			}

			l = next;
		}
	}

	PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
}

void _starpu_add_post_sync_tasks(struct starpu_task *post_sync_task, starpu_data_handle handle)
{
	PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);

	if (handle->sequential_consistency)
	{
		handle->post_sync_tasks_cnt++;

		struct starpu_task_wrapper_list *link = malloc(sizeof(struct starpu_task_wrapper_list));
		link->task = post_sync_task;
		link->next = handle->post_sync_tasks;
		handle->post_sync_tasks = link;		
	}

	PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
}

void _starpu_unlock_post_sync_tasks(starpu_data_handle handle)
{
	struct starpu_task_wrapper_list *post_sync_tasks = NULL;
	unsigned do_submit_tasks = 0;

	PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);

	if (handle->sequential_consistency)
	{
		STARPU_ASSERT(handle->post_sync_tasks_cnt > 0);

		if (--handle->post_sync_tasks_cnt == 0)
		{
			/* unlock all tasks : we need not hold the lock while unlocking all these tasks */
			do_submit_tasks = 1;
			post_sync_tasks = handle->post_sync_tasks;
			handle->post_sync_tasks = NULL;
		}

	}

	PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);

	if (do_submit_tasks)
	{
		struct starpu_task_wrapper_list *link = post_sync_tasks;

		while (link) {
			/* There is no need to depend on that task now, since it was already unlocked */
			_starpu_release_data_enforce_sequential_consistency(link->task, handle);

			int ret = starpu_task_submit(link->task);
			STARPU_ASSERT(!ret);
			link = link->next;
		}
	}
}

/* If sequential consistency mode is enabled, this function blocks until the
 * handle is available in the requested access mode. */
int _starpu_data_wait_until_available(starpu_data_handle handle, starpu_access_mode mode)
{
	/* If sequential consistency is enabled, wait until data is available */
	PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
	int sequential_consistency = handle->sequential_consistency;
	if (sequential_consistency)
	{
		struct starpu_task *sync_task;
		sync_task = starpu_task_create();
		sync_task->detach = 0;
		sync_task->destroy = 1;

		/* It is not really a RW access, but we want to make sure that
		 * all previous accesses are done */
		_starpu_detect_implicit_data_deps_with_handle(sync_task, sync_task, handle, mode);
		PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);

		/* TODO detect if this is superflous */
		int ret = starpu_task_submit(sync_task);
		STARPU_ASSERT(!ret);
		starpu_task_wait(sync_task);
	}
	else {
		PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
	}

	return 0;
}
