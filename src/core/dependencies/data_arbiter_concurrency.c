/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/dependencies/data_concurrency.h>
#include <datawizard/coherency.h>
#include <core/sched_policy.h>
#include <common/starpu_spinlock.h>
#include <datawizard/sort_data_handles.h>
#include <datawizard/memalloc.h>
#include <datawizard/memory_nodes.h>

/* TODO factorize with data_concurrency.c and btw support redux */

//#define LOCK_OR_DELEGATE

/*
 * This implements a solution for the dining philosophers problem (see
 * data_concurrency.c for the rationale) based on a centralized arbiter.  This
 * allows to get a more parallel solution than the Dijkstra solution, by
 * avoiding strictly serialized executions, and instead opportunistically find
 * which tasks can take data.
 *
 * These are the algorithms implemented below:
 *
 *
 * at termination of task T:
 *
 * - for each handle h of T:
 *   - mutex_lock(&arbiter)
 *   - release reference on h
 *   - call _starpu_notify_arbitered_dependencies which does the following
 *   - for each task Tc waiting for h:
 *     - for each data Tc_h it is waiting for:
 *       - if Tc_h is busy, goto fail
 *     // Ok, now really take them
 *     - For each data Tc_h it is waiting:
 *       - lock(Tc_h)
 *       - take reference on h (it should be still available since we hold the arbiter)
 *       - unlock(Tc_h)
 *     // Ok, we managed to find somebody, we're finished!
 *     _starpu_push_task(Tc);
 *     break;
 *     fail:
 *       - unrecord T as waiting on h
 *       - record T as waiting on Tc_h
 *       // No luck, let's try another task
 *       continue;
 *   // Release the arbiter mutex a bit from time to time
 *   - mutex_unlock(&arbiter)
 *
 *
 * at submission of task T (_starpu_submit_job_enforce_arbitered_deps):
 *
 * - mutex_lock(&arbiter)
 * - for each handle h of T:
 *   - lock(h)
 *   - try to take a reference on h, goto fail on failure
 *   - unlock(h)
 * // Success!
 * - mutex_unlock(&arbiter);
 * - return 0;
 *
 * fail:
 * // couldn't take everything, record task T and abort
 * - record T as waiting on h
 * // drop spurious references
 * - for each handle h of T already taken:
 *   - lock(h)
 *   - release reference on h
 *   - unlock(h)
 * - mutex_unlock(&arbiter)
 * - return 1;
 *
 * at acquire (_starpu_attempt_to_submit_arbitered_data_request):
 * - mutex_lock(&arbiter)
 * - try to take a reference on h
 *   - on failure, record as waiting on h
 * - mutex_unlock(&arbiter);
 * - return 0 if succeeded, 1 if failed;
 */

static int _starpu_arbiter_filter_modes(int mode)
{
	/* Do not care about some flags */
	mode &= ~STARPU_COMMUTE;
	mode &= ~STARPU_SSEND;
	mode &= ~STARPU_LOCALITY;
	if (mode == STARPU_RW)
		mode = STARPU_W;
	return mode;
}

struct starpu_arbiter
{
#ifdef LOCK_OR_DELEGATE
/* The list of task to perform */
	struct LockOrDelegateListNode* dlTaskListHead;

/* To protect the list of tasks */
	struct _starpu_spinlock dlListLock;
/* Whether somebody is working on the list */
	int working;
#else /* LOCK_OR_DELEGATE */
	starpu_pthread_mutex_t mutex;
#endif /* LOCK_OR_DELEGATE */
};

#ifdef LOCK_OR_DELEGATE

/* In case of congestion, we don't want to needlessly wait for the arbiter lock
 * while we can just delegate the work to the worker already managing some
 * dependencies.
 *
 * So we push work on the dlTastListHead queue and only one worker will process
 * the list.
 */

/* A LockOrDelegate task list */
struct LockOrDelegateListNode
{
	void (*func)(void*);
	void* data;
	struct LockOrDelegateListNode* next;
};

/* Post a task to perfom if possible, otherwise put it in the list
 * If we can perfom this task, we may also perfom all the tasks in the list
 * This function return 1 if the task (and maybe some others) has been done
 * by the calling thread and 0 otherwise (if the task has just been put in the list)
 */
static int _starpu_LockOrDelegatePostOrPerform(starpu_arbiter_t arbiter, void (*func)(void*), void* data)
{
	struct LockOrDelegateListNode *newNode, *iter, *next;
	int did = 0;

	_STARPU_MALLOC(newNode, sizeof(*newNode));
	newNode->data = data;
	newNode->func = func;

	_starpu_spin_lock(&arbiter->dlListLock);
	if (arbiter->working)
	{
		/* Somebody working on it, insert the node */
		newNode->next = arbiter->dlTaskListHead;
		arbiter->dlTaskListHead = newNode;
	}
	else
	{
		/* Nobody working on the list, we'll work */
		arbiter->working = 1;

		/* work on what was pushed so far first */
		iter = arbiter->dlTaskListHead;
		arbiter->dlTaskListHead = NULL;
		_starpu_spin_unlock(&arbiter->dlListLock);
		while (iter != NULL)
		{
			(*iter->func)(iter->data);
			next = iter->next;
			free(iter);
			iter = next;
		}

		/* And then do our job */
		(*func)(data);
		free(newNode);
		did = 1;

		_starpu_spin_lock(&arbiter->dlListLock);
		/* And finish working on anything that could have been pushed
		 * in the meanwhile */
		while (arbiter->dlTaskListHead != 0)
		{
			iter = arbiter->dlTaskListHead;
			arbiter->dlTaskListHead = arbiter->dlTaskListHead->next;
			_starpu_spin_unlock(&arbiter->dlListLock);

			(*iter->func)(iter->data);
			free(iter);
			_starpu_spin_lock(&arbiter->dlListLock);
		}

		arbiter->working = 0;
	}

	_starpu_spin_unlock(&arbiter->dlListLock);
	return did;
}

#endif

/* Try to submit just one data request, in case the request can be processed
 * immediatly, return 0, if there is still a dependency that is not compatible
 * with the current mode, the request is put in the per-handle list of
 * "requesters", and this function returns 1. */
#ifdef LOCK_OR_DELEGATE
struct starpu_submit_arbitered_args
{
	unsigned request_from_codelet;
	starpu_data_handle_t handle;
	enum starpu_data_access_mode mode;
	void (*callback)(void *);
	void *argcb;
	struct _starpu_job *j;
	unsigned buffer_index;
};
static unsigned ___starpu_attempt_to_submit_arbitered_data_request(unsigned request_from_codelet,
						       starpu_data_handle_t handle, enum starpu_data_access_mode mode,
						       void (*callback)(void *), void *argcb,
						       struct _starpu_job *j, unsigned buffer_index);
static void __starpu_attempt_to_submit_arbitered_data_request(void *inData)
{
	struct starpu_submit_arbitered_args* args = inData;
	unsigned request_from_codelet = args->request_from_codelet;
	starpu_data_handle_t handle = args->handle;
	enum starpu_data_access_mode mode = args->mode;
	void (*callback)(void*) = args->callback;
	void *argcb = args->argcb;
	struct _starpu_job *j = args->j;
	unsigned buffer_index = args->buffer_index;
	free(args);
	if (!___starpu_attempt_to_submit_arbitered_data_request(request_from_codelet, handle, mode, callback, argcb, j, buffer_index))
		/* Success, but we have no way to report it to original caller,
		 * so call callback ourself */
		callback(argcb);
}

unsigned _starpu_attempt_to_submit_arbitered_data_request(unsigned request_from_codelet,
						       starpu_data_handle_t handle, enum starpu_data_access_mode mode,
						       void (*callback)(void *), void *argcb,
						       struct _starpu_job *j, unsigned buffer_index)
{
	struct starpu_submit_arbitered_args* args;
	_STARPU_MALLOC(args, sizeof(*args));
	args->request_from_codelet = request_from_codelet;
	args->handle = handle;
	args->mode = mode;
	args->callback = callback;
	args->argcb = argcb;
	args->j = j;
	args->buffer_index = buffer_index;
	/* The function will delete args */
	_starpu_LockOrDelegatePostOrPerform(handle->arbiter, &__starpu_attempt_to_submit_arbitered_data_request, args);
	return 1;
}

unsigned ___starpu_attempt_to_submit_arbitered_data_request(unsigned request_from_codelet,
						       starpu_data_handle_t handle, enum starpu_data_access_mode mode,
						       void (*callback)(void *), void *argcb,
						       struct _starpu_job *j, unsigned buffer_index)
{
	STARPU_ASSERT(handle->arbiter);
#else // LOCK_OR_DELEGATE
unsigned _starpu_attempt_to_submit_arbitered_data_request(unsigned request_from_codelet,
						       starpu_data_handle_t handle, enum starpu_data_access_mode mode,
						       void (*callback)(void *), void *argcb,
						       struct _starpu_job *j, unsigned buffer_index)
{
	starpu_arbiter_t arbiter = handle->arbiter;
	STARPU_PTHREAD_MUTEX_LOCK(&arbiter->mutex);
#endif // LOCK_OR_DELEGATE

	mode = _starpu_arbiter_filter_modes(mode);

	STARPU_ASSERT_MSG(!(mode & STARPU_REDUX), "REDUX with arbiter is not implemented\n");

	/* Take the lock protecting the header. We try to do some progression
	 * in case this is called from a worker, otherwise we just wait for the
	 * lock to be available. */
	if (request_from_codelet)
	{
		int cpt = 0;
		while (cpt < STARPU_SPIN_MAXTRY && _starpu_spin_trylock(&handle->header_lock))
		{
			cpt++;
			_starpu_datawizard_progress(0);
		}
		if (cpt == STARPU_SPIN_MAXTRY)
			_starpu_spin_lock(&handle->header_lock);
	}
	else
	{
		_starpu_spin_lock(&handle->header_lock);
	}

	/* If there is currently nobody accessing the piece of data, or it's
	 * not another writter and if this is the same type of access as the
	 * current one, we can proceed. */
	unsigned put_in_list = 1;

	if ((handle->refcnt == 0) || (!(mode == STARPU_W) && (handle->current_mode == mode)))
	{
		/* TODO: Detect whether this is the end of a reduction phase etc. like in data_concurrency.c */
		if (0)
		{
		}
		else
		{
			put_in_list = 0;
		}
	}

	if (put_in_list)
	{
		/* there cannot be multiple writers or a new writer
		 * while the data is in read mode */

		handle->busy_count++;
		/* enqueue the request */
		struct _starpu_data_requester *r = _starpu_data_requester_new();
		r->mode = mode;
		r->is_requested_by_codelet = request_from_codelet;
		r->j = j;
		r->buffer_index = buffer_index;
		r->prio = j ? j->task->priority : 0;
		r->ready_data_callback = callback;
		r->argcb = argcb;

		_starpu_data_requester_prio_list_push_back(&handle->arbitered_req_list, r);

		/* failed */
		put_in_list = 1;
	}
	else
	{
		handle->refcnt++;
		handle->busy_count++;

		/* Do not write to handle->current_mode if it is already
		 * R. This avoids a spurious warning from helgrind when
		 * the following happens:
		 * acquire(R) in thread A
		 * acquire(R) in thread B
		 * release_data_on_node() in thread A
		 * helgrind would shout that the latter reads current_mode
		 * unsafely.
		 *
		 * This actually basically explains helgrind that it is a
		 * shared R acquisition.
		 */
		if (mode != STARPU_R || handle->current_mode != mode)
			handle->current_mode = mode;

		/* success */
		put_in_list = 0;
	}

	_starpu_spin_unlock(&handle->header_lock);
#ifndef LOCK_OR_DELEGATE
	STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif // LOCK_OR_DELEGATE
	return put_in_list;

}



#ifdef LOCK_OR_DELEGATE
/* These are the arguments passed to _submit_job_enforce_arbitered_deps */
struct starpu_enforce_arbitered_args
{
	struct _starpu_job *j;
	unsigned buf;
	unsigned nbuffers;
};

static void ___starpu_submit_job_enforce_arbitered_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers);
static void __starpu_submit_job_enforce_arbitered_deps(void* inData)
{
	struct starpu_enforce_arbitered_args* args = inData;
	struct _starpu_job *j = args->j;
	unsigned buf		  = args->buf;
	unsigned nbuffers	 = args->nbuffers;
	/* we are in charge of freeing the args */
	free(args);
	___starpu_submit_job_enforce_arbitered_deps(j, buf, nbuffers);
}

void _starpu_submit_job_enforce_arbitered_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
	struct starpu_enforce_arbitered_args* args;
	_STARPU_MALLOC(args, sizeof(*args));
	starpu_data_handle_t handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, buf);
	args->j = j;
	args->buf = buf;
	args->nbuffers = nbuffers;
	/* The function will delete args */
	_starpu_LockOrDelegatePostOrPerform(handle->arbiter, &__starpu_submit_job_enforce_arbitered_deps, args);
}

static void ___starpu_submit_job_enforce_arbitered_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
	starpu_arbiter_t arbiter = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, buf)->arbiter;
#else // LOCK_OR_DELEGATE
void _starpu_submit_job_enforce_arbitered_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
	struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);
	starpu_arbiter_t arbiter = descrs[buf].handle->arbiter;
	STARPU_PTHREAD_MUTEX_LOCK(&arbiter->mutex);
#endif
	STARPU_ASSERT(arbiter);

	const unsigned start_buf_arbiter = buf;
	unsigned idx_buf_arbiter;
	unsigned all_arbiter_available = 1;

	starpu_data_handle_t handle;
	enum starpu_data_access_mode mode;

	for (idx_buf_arbiter = start_buf_arbiter; idx_buf_arbiter < nbuffers; idx_buf_arbiter++)
	{
		handle = descrs[idx_buf_arbiter].handle;
		mode = descrs[idx_buf_arbiter].mode & ~STARPU_COMMUTE;

		mode = _starpu_arbiter_filter_modes(mode);

		STARPU_ASSERT_MSG(!(mode & STARPU_REDUX), "REDUX with arbiter is not implemented\n");

		if (idx_buf_arbiter && (descrs[idx_buf_arbiter-1].handle == handle))
			/* We have already requested this data, skip it. This
			 * depends on ordering putting writes before reads, see
			 * _starpu_compar_handles.  */
			continue;

		if (handle->arbiter != arbiter)
		{
			/* another arbiter */
			break;
		}

		/* Try to take handle */
		_starpu_spin_lock(&handle->header_lock);
		if ((handle->refcnt == 0) || (!(mode == STARPU_W) && (handle->current_mode == mode)))
		{
			/* Got it */
			handle->refcnt++;
			handle->busy_count++;
			if (mode != STARPU_R || handle->current_mode != mode)
				handle->current_mode = mode;
			_starpu_spin_unlock(&handle->header_lock);
		}
		else
		{
			/* a handle does not have a refcnt == 0, stop */
			_starpu_spin_unlock(&handle->header_lock);
			all_arbiter_available = 0;
			break;
		}
	}
	if (all_arbiter_available == 0)
	{
		/* Oups, record ourself as waiting for this data */

		struct _starpu_data_requester *r = _starpu_data_requester_new();
		r->mode = mode;
		r->is_requested_by_codelet = 1;
		r->j = j;
		r->buffer_index = start_buf_arbiter;
		r->prio = j->task->priority;
		r->ready_data_callback = NULL;
		r->argcb = NULL;

		/* store node in list */
		_starpu_data_requester_prio_list_push_front(&handle->arbitered_req_list, r);

		_starpu_spin_lock(&handle->header_lock);
		handle->busy_count++;
		_starpu_spin_unlock(&handle->header_lock);

		/* and cancel all taken */
		unsigned idx_buf_cancel;
		for (idx_buf_cancel = start_buf_arbiter; idx_buf_cancel < idx_buf_arbiter ; idx_buf_cancel++)
		{
			starpu_data_handle_t cancel_handle = descrs[idx_buf_cancel].handle;

			if (idx_buf_cancel && (descrs[idx_buf_cancel-1].handle == cancel_handle))
				continue;
			if (cancel_handle->arbiter != arbiter)
				/* Will have to process another arbiter, will do that later */
				break;

			_starpu_spin_lock(&cancel_handle->header_lock);
			/* reset the counter because finally we do not take the data */
			STARPU_ASSERT(cancel_handle->refcnt >= 1);
			cancel_handle->refcnt--;
			STARPU_ASSERT(cancel_handle->busy_count > 0);
			cancel_handle->busy_count--;
			if (!_starpu_data_check_not_busy(cancel_handle))
				_starpu_spin_unlock(&cancel_handle->header_lock);
		}

#ifndef LOCK_OR_DELEGATE
		STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif
		return;
	}
#ifndef LOCK_OR_DELEGATE
	STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif

	// all_arbiter_available is true
	if (idx_buf_arbiter < nbuffers)
		/* Other arbitered data, process them */
		_starpu_submit_job_enforce_arbitered_deps(j, idx_buf_arbiter, nbuffers);
	else
		/* Finished with all data, can eventually push! */
		_starpu_push_task(j);
}

#ifdef LOCK_OR_DELEGATE
void ___starpu_notify_arbitered_dependencies(starpu_data_handle_t handle);
void __starpu_notify_arbitered_dependencies(void* inData)
{
	starpu_data_handle_t handle = inData;
	___starpu_notify_arbitered_dependencies(handle);
}
void _starpu_notify_arbitered_dependencies(starpu_data_handle_t handle)
{
	_starpu_LockOrDelegatePostOrPerform(handle->arbiter, &__starpu_notify_arbitered_dependencies, handle);
}
void ___starpu_notify_arbitered_dependencies(starpu_data_handle_t handle)
#else // LOCK_OR_DELEGATE
void _starpu_notify_arbitered_dependencies(starpu_data_handle_t handle)
#endif
{
	starpu_arbiter_t arbiter = handle->arbiter;
#ifndef LOCK_OR_DELEGATE
	STARPU_PTHREAD_MUTEX_LOCK(&arbiter->mutex);
#endif

	/* Since the request has been posted the handle may have been proceed and released */
	if (_starpu_data_requester_prio_list_empty(&handle->arbitered_req_list))
	{
		/* No waiter, just remove our reference */
		_starpu_spin_lock(&handle->header_lock);
		STARPU_ASSERT(handle->refcnt > 0);
		handle->refcnt--;
		STARPU_ASSERT(handle->busy_count > 0);
		handle->busy_count--;
#ifndef LOCK_OR_DELEGATE
		STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif
		if (_starpu_data_check_not_busy(handle))
			/* Handle was even destroyed, don't unlock it.  */
			return;
		_starpu_spin_unlock(&handle->header_lock);
		return;
	}

	/* There is a waiter, remove our reference */
	_starpu_spin_lock(&handle->header_lock);
	STARPU_ASSERT(handle->refcnt > 0);
	handle->refcnt--;
	STARPU_ASSERT(handle->busy_count > 0);
	handle->busy_count--;
	/* There should be at least one busy_count reference for the waiter
	 * (thus we don't risk to see the handle disappear below) */
	STARPU_ASSERT(handle->busy_count > 0);
	_starpu_spin_unlock(&handle->header_lock);

	/* Note: we may be putting back our own requests, so avoid looping by
	 * extracting the list */
	struct _starpu_data_requester_prio_list l = handle->arbitered_req_list;
	_starpu_data_requester_prio_list_init(&handle->arbitered_req_list);

	while (!_starpu_data_requester_prio_list_empty(&l))
	{
		struct _starpu_data_requester *r = _starpu_data_requester_prio_list_pop_front_highest(&l);

		if (!r->is_requested_by_codelet)
		{
			/* data_acquire_cb, process it */
			enum starpu_data_access_mode r_mode = r->mode;
			int put_in_list = 1;

			r_mode = _starpu_arbiter_filter_modes(r_mode);

			_starpu_spin_lock(&handle->header_lock);
			handle->busy_count++;
			if ((handle->refcnt == 0) || (!(r_mode == STARPU_W) && (handle->current_mode == r_mode)))
			{
				handle->refcnt++;
				handle->current_mode = r_mode;
				put_in_list = 0;
			}
			_starpu_spin_unlock(&handle->header_lock);

			if (put_in_list)
				_starpu_data_requester_prio_list_push_front(&l, r);

			/* Put back remaining requests */
			_starpu_data_requester_prio_list_push_prio_list_back(&handle->arbitered_req_list, &l);
#ifndef LOCK_OR_DELEGATE
			STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif
			if (!put_in_list)
			{
				r->ready_data_callback(r->argcb);
				_starpu_data_requester_delete(r);
			}

			_starpu_spin_lock(&handle->header_lock);
			STARPU_ASSERT(handle->busy_count > 0);
			handle->busy_count--;
			if (!_starpu_data_check_not_busy(handle))
				_starpu_spin_unlock(&handle->header_lock);
			return;
		}

		/* A task waiting for a set of data, try to acquire them */

		struct _starpu_job* j = r->j;
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);

		unsigned idx_buf_arbiter;
		unsigned all_arbiter_available = 1;
		starpu_data_handle_t handle_arbiter;
		enum starpu_data_access_mode mode;

		unsigned start_buf_arbiter = r->buffer_index;
		struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);

		for (idx_buf_arbiter = start_buf_arbiter; idx_buf_arbiter < nbuffers; idx_buf_arbiter++)
		{
			handle_arbiter = descrs[idx_buf_arbiter].handle;
			if (idx_buf_arbiter && (descrs[idx_buf_arbiter-1].handle == handle_arbiter))
				/* We have already requested this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles.  */
				continue;
			if (handle_arbiter->arbiter != arbiter)
				/* Will have to process another arbiter, will do that later */
				break;

			mode = descrs[idx_buf_arbiter].mode;
			mode = _starpu_arbiter_filter_modes(mode);

			/* we post all arbiter  */
			_starpu_spin_lock(&handle_arbiter->header_lock);
			if (!((handle_arbiter->refcnt == 0) || (!(mode == STARPU_W) && (handle_arbiter->current_mode == mode))))
			{
				/* handle is not available, record ourself */
				_starpu_spin_unlock(&handle_arbiter->header_lock);
				all_arbiter_available = 0;
				break;
			}
			/* mark the handle as taken */
			handle_arbiter->refcnt++;
			handle_arbiter->busy_count++;
			handle_arbiter->current_mode = mode;
			_starpu_spin_unlock(&handle_arbiter->header_lock);
		}

		if (all_arbiter_available)
		{
			/* Success! Drop request */
			_starpu_data_requester_delete(r);

			_starpu_spin_lock(&handle->header_lock);
			STARPU_ASSERT(handle->busy_count > 0);
			handle->busy_count--;
			if (!_starpu_data_check_not_busy(handle))
				_starpu_spin_unlock(&handle->header_lock);

			/* Put back remaining requests */
			_starpu_data_requester_prio_list_push_prio_list_back(&handle->arbitered_req_list, &l);
#ifndef LOCK_OR_DELEGATE
			STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif

			if (idx_buf_arbiter < nbuffers)
				/* Other arbitered data, process them */
				_starpu_submit_job_enforce_arbitered_deps(j, idx_buf_arbiter, nbuffers);
			else
				/* Finished with all data, can eventually push! */
				_starpu_push_task(j);

			return;
		}
		else
		{
			/* all handles are not available - record that task on the first unavailable handle */

			/* store node in list */
			r->mode = mode;
			_starpu_data_requester_prio_list_push_front(&handle_arbiter->arbitered_req_list, r);

			/* Move check_busy reference too */
			_starpu_spin_lock(&handle->header_lock);
			STARPU_ASSERT(handle->busy_count > 0);
			handle->busy_count--;
			if (!_starpu_data_check_not_busy(handle))
				_starpu_spin_unlock(&handle->header_lock);

			_starpu_spin_lock(&handle_arbiter->header_lock);
			handle_arbiter->busy_count++;
			_starpu_spin_unlock(&handle_arbiter->header_lock);

			/* and revert the mark */
			unsigned idx_buf_cancel;
			for (idx_buf_cancel = start_buf_arbiter; idx_buf_cancel < idx_buf_arbiter ; idx_buf_cancel++)
			{
				starpu_data_handle_t cancel_handle = descrs[idx_buf_cancel].handle;
				if (idx_buf_cancel && (descrs[idx_buf_cancel-1].handle == cancel_handle))
					continue;
				if (cancel_handle->arbiter != arbiter)
					break;
				_starpu_spin_lock(&cancel_handle->header_lock);
				STARPU_ASSERT(cancel_handle->refcnt >= 1);
				cancel_handle->refcnt--;
				STARPU_ASSERT(cancel_handle->busy_count > 0);
				cancel_handle->busy_count--;
				if (!_starpu_data_check_not_busy(cancel_handle))
					_starpu_spin_unlock(&cancel_handle->header_lock);
			}
		}
	}
	/* no task has been pushed */
#ifndef LOCK_OR_DELEGATE
	STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
#endif
	return;
}

starpu_arbiter_t starpu_arbiter_create(void)
{
	starpu_arbiter_t res;
	_STARPU_MALLOC(res, sizeof(*res));

#ifdef LOCK_OR_DELEGATE
	res->dlTaskListHead = NULL;
	_starpu_spin_init(&res->dlListLock);
	res->working = 0;
#else /* LOCK_OR_DELEGATE */
	STARPU_PTHREAD_MUTEX_INIT(&res->mutex, NULL);
#endif /* LOCK_OR_DELEGATE */

	return res;
}

void starpu_data_assign_arbiter(starpu_data_handle_t handle, starpu_arbiter_t arbiter)
{
	if (handle->arbiter && handle->arbiter == _starpu_global_arbiter)
		/* Just for testing purpose */
		return;
	STARPU_ASSERT_MSG(!handle->arbiter, "handle can only be assigned one arbiter");
	STARPU_ASSERT_MSG(!handle->refcnt, "arbiter can be assigned to handle only right after initialization");
	STARPU_ASSERT_MSG(!handle->busy_count, "arbiter can be assigned to handle only right after initialization");
	handle->arbiter = arbiter;
}

void starpu_arbiter_destroy(starpu_arbiter_t arbiter)
{
#ifdef LOCK_OR_DELEGATE
	_starpu_spin_lock(&arbiter->dlListLock);
	STARPU_ASSERT(!arbiter->dlTaskListHead);
	STARPU_ASSERT(!arbiter->working);
	_starpu_spin_unlock(&arbiter->dlListLock);
	_starpu_spin_destroy(&arbiter->dlListLock);
#else /* LOCK_OR_DELEGATE */
	STARPU_PTHREAD_MUTEX_LOCK(&arbiter->mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&arbiter->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&arbiter->mutex);
#endif /* LOCK_OR_DELEGATE */
	free(arbiter);
}
