/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <core/dependencies/data-concurrency.h>
#include <datawizard/coherency.h>
#include <core/policies/sched_policy.h>
#include <common/starpu-spinlock.h>

static unsigned _submit_job_enforce_data_deps(job_t j, unsigned start_buffer_index);

static unsigned unlock_one_requester(data_requester_t r)
{
	job_t j = r->j;
	unsigned nbuffers = j->task->cl->nbuffers;
	unsigned buffer_index = r->buffer_index;

	if (buffer_index + 1 < nbuffers)
	{
		/* not all buffers are protected yet */
		return _submit_job_enforce_data_deps(j, buffer_index + 1);
	}
	else
		return 0;
}

/* the header lock must be taken by the caller */
static unsigned may_unlock_data_req_list_head(data_state *data)
{
	/* if there is no one to unlock ... */
	if (data_requester_list_empty(data->req_list))
		return 0;

	/* if there is no reference to the data anymore, we can use it */
	if (data->refcnt == 0)
	{
		STARPU_ASSERT(!data->per_node[0].request);
		STARPU_ASSERT(!data->per_node[1].request);
		return 1;
	}

	if (data->current_mode == STARPU_W)
		return 0;

	/* data->current_mode == STARPU_R, so we can process more readers */
	data_requester_t r = data_requester_list_front(data->req_list);
	
	return (r->mode == STARPU_R);
}


unsigned attempt_to_submit_data_request_from_apps(data_state *data, starpu_access_mode mode,
						void (*callback)(void *), void *argcb)
{
	unsigned ret;

	starpu_spin_lock(&data->header_lock);

	if (data->refcnt == 0)
	{
		/* there is nobody currently about to manipulate the data */
		data->refcnt++;
		data->current_mode = mode;

		/* success */
		ret = 0;
	}
	else
	{
		/* there is already someone that may access the data */
		if ( (mode == STARPU_R) && (data->current_mode == STARPU_R))
		{
			data->refcnt++;

			/* success : there is a new reader */
			ret = 0;
		}
		else
		{
			/* there cannot be multiple writers or a new writer
			 * while the data is in read mode */
			
			/* enqueue the request */
			data_requester_t r = data_requester_new();
				r->mode = mode;
				r->is_requested_by_codelet = 0;
				r->ready_data_callback = callback;
				r->argcb = argcb;

			data_requester_list_push_back(data->req_list, r);

			/* failed */
			ret = 1;
		}
	}

	starpu_spin_unlock(&data->header_lock);
	return ret;
}

static unsigned attempt_to_submit_data_request_from_job(job_t j, unsigned buffer_index)
{
	unsigned ret;

	data_state *data = j->task->buffers[buffer_index].handle;
	starpu_access_mode mode = j->task->buffers[buffer_index].mode;

	while (starpu_spin_trylock(&data->header_lock))
		datawizard_progress(get_local_memory_node(), 0);

	if (data->refcnt == 0)
	{
		/* there is nobody currently about to manipulate the data */
		data->refcnt++;
		data->current_mode = (mode==STARPU_R)?STARPU_R:STARPU_W;

		/* success */
		ret = 0;
	}
	else
	{
		/* there is already someone that may access the data */
		if ( (mode == STARPU_R) && (data->current_mode == STARPU_R))
		{
			data->refcnt++;

			/* success : there is a new reader */
			ret = 0;
		}
		else
		{
			/* there cannot be multiple writers or a new writer
			 * while the data is in read mode */
			
			/* enqueue the request */
			data_requester_t r = data_requester_new();
				r->mode = mode;
				r->is_requested_by_codelet = 1;
				r->j = j;
				r->buffer_index = buffer_index;

			data_requester_list_push_back(data->req_list, r);

			/* failed */
			ret = 1;
		}
	}

	starpu_spin_unlock(&data->header_lock);
	return ret;
}

static unsigned _submit_job_enforce_data_deps(job_t j, unsigned start_buffer_index)
{
	unsigned buf;

	/* TODO compute an ordered list of the data */

	unsigned nbuffers = j->task->cl->nbuffers;
	for (buf = start_buffer_index; buf < nbuffers; buf++)
	{
		if (attempt_to_submit_data_request_from_job(j, buf))
			return 1;
	}

	return 0;
}

/* When a new task is submitted, we make sure that there cannot be codelets
   with concurrent data-access at the same time in the scheduling engine (eg.
   there can be 2 tasks reading a piece of data, but there cannot be one
   reading and another writing) */
unsigned submit_job_enforce_data_deps(job_t j)
{
	if ((j->task->cl == NULL) || (j->task->cl->nbuffers == 0))
		return 0;

	return _submit_job_enforce_data_deps(j, 0);
}


void notify_data_dependencies(data_state *data)
{
	starpu_spin_lock(&data->header_lock);

	data->refcnt--;

	while (may_unlock_data_req_list_head(data))
	{
		/* unlock the head of the requester list */
		data_requester_t r = data_requester_list_pop_front(data->req_list);

		data->refcnt++;
	
		starpu_spin_unlock(&data->header_lock);

		if (r->is_requested_by_codelet)
		{
			if (!unlock_one_requester(r))
				push_task(r->j);
		}
		else
		{
			STARPU_ASSERT(r->ready_data_callback);

			/* execute the callback associated with the data requester */
			r->ready_data_callback(r->argcb);
		}

		data_requester_delete(r);
		
		starpu_spin_lock(&data->header_lock);
	}
	
	starpu_spin_unlock(&data->header_lock);

}
