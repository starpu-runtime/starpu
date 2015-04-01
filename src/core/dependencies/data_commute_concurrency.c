/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Université de Bordeaux
 * Copyright (C) 2015  Inria
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

#define LOCK_OR_DELEGATE

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
 *   - for each task Tc waiting for h:
 *     - for each data Tc_h it is waiting:
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
 *       // No luck, let's try another task
 *       continue;
 *   // Release the arbiter mutex a bit from time to time
 *   - mutex_unlock(&arbiter)
 *
 *
 * at submission of task T:
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
 * // couldn't take everything, abort and record task T
 * // drop spurious references
 * - for each handle h of T already taken:
 *   - lock(h)
 *   - release reference on h
 *   - unlock(h)
 * // record T on the list of requests for h
 * - for each handle h of T:
 *   - record T as waiting on h
 * - mutex_unlock(&arbiter)
 * - return 1;
 */

/* Here are the LockOrDelegate functions
 * There are two version depending on the support of the compare and exchange
 * support from the compiler
 */

#ifdef LOCK_OR_DELEGATE

/* A LockOrDelegate task list */
struct LockOrDelegateListNode
{
	void (*func)(void*);
	void* data;
	struct LockOrDelegateListNode* next;
};

/* If the compiler support C11 and the usage of atomic functions */
#if (201112L <= __STDC_VERSION__) && !(defined(__STDC_NO_ATOMICS__))

#include <stdatomic.h>

/* To know the number of task to perform and attributes the tickets */
static atomic_int dlAtomicCounter;
/* The list of task to perform */
static _Atomic struct LockOrDelegateListNode* dlListHead;

/* Post a task to perform if possible, otherwise put it in the list
 * If we can perform this task, we may also perform all the tasks in the list
 * This function return 1 if the task (and maybe some others) has been done
 * by the calling thread and 0 otherwise (if the task has just been put in the list)
 */
static int _starpu_LockOrDelegatePostOrPerform(void (*func)(void*), void* data)
{
	/* Get our ticket */
	int insertionPosition = atomic_load(&dlAtomicCounter);
	while (!atomic_compare_exchange_weak(&dlAtomicCounter, &insertionPosition, insertionPosition+1))
		;

	/* If we obtain 0 we are responsible of computing all the tasks */
	if(insertionPosition == 0)
	{
		/* start by our current task */
		(*func)(data);

		/* Compute task of other and manage ticket */
		while(1)
		{
			STARPU_ASSERT(atomic_load(&dlAtomicCounter) > 0);

			/* Dec ticket and see if something else has to be done */
			int removedPosition = atomic_load(&dlAtomicCounter);
			while(!atomic_compare_exchange_weak(&dlAtomicCounter, &removedPosition,removedPosition-1))
				;
			if(removedPosition-1 == 0)
			{
				break;
			}

			/* Get the next task */
			struct LockOrDelegateListNode* removedNode = (struct LockOrDelegateListNode*)atomic_load(&dlListHead);
			// Maybe it has not been pushed yet (listHead.load() == nullptr)
			while((removedNode = (struct LockOrDelegateListNode*)atomic_load(&dlListHead)) == NULL || !atomic_compare_exchange_weak(&dlListHead, &removedNode,removedNode->next))
				;
			STARPU_ASSERT(removedNode);
			/* call the task */
			(*removedNode->func)(removedNode->data);
			// Delete node
			free(removedNode);
		}

		return 1;
	}

	struct LockOrDelegateListNode* newNode = (struct LockOrDelegateListNode*)malloc(sizeof(struct LockOrDelegateListNode));
	STARPU_ASSERT(newNode);
	newNode->data = data;
	newNode->func = func;
	newNode->next = (struct LockOrDelegateListNode*)atomic_load(&dlListHead);
	while(!atomic_compare_exchange_weak(&dlListHead, &newNode->next, newNode))
		;

	return 0;
}

#else
/* We cannot rely on the C11 atomics */
#warning Lock based version of Lock or Delegate

/* The list of task to perform */
static struct LockOrDelegateListNode* dlTaskListHead = NULL;

/* To protect the list of tasks */
static starpu_pthread_mutex_t dlListLock = STARPU_PTHREAD_MUTEX_INITIALIZER;
/* To know who is responsible to compute all the tasks */
static starpu_pthread_mutex_t dlWorkLock = STARPU_PTHREAD_MUTEX_INITIALIZER;

/* Post a task to perfom if possible, otherwise put it in the list
 * If we can perfom this task, we may also perfom all the tasks in the list
 * This function return 1 if the task (and maybe some others) has been done
 * by the calling thread and 0 otherwise (if the task has just been put in the list)
 */
static int _starpu_LockOrDelegatePostOrPerform(void (*func)(void*), void* data)
{
	/* We could avoid to allocate if we will be responsible but for simplicity
	 * we always push the task in the list */
	struct LockOrDelegateListNode* newNode = (struct LockOrDelegateListNode*)malloc(sizeof(struct LockOrDelegateListNode));
	STARPU_ASSERT(newNode);
	newNode->data = data;
	newNode->func = func;
	int ret;

	/* insert the node */
	STARPU_PTHREAD_MUTEX_LOCK(&dlListLock);
	newNode->next = dlTaskListHead;
	dlTaskListHead = newNode;
	STARPU_PTHREAD_MUTEX_UNLOCK(&dlListLock);

	/* See if we can compute all the tasks */
	if((ret = STARPU_PTHREAD_MUTEX_TRYLOCK(&dlWorkLock)) == 0)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&dlListLock);
		while(dlTaskListHead != 0)
		{
			struct LockOrDelegateListNode* iter = dlTaskListHead;
			dlTaskListHead = dlTaskListHead->next;
			STARPU_PTHREAD_MUTEX_UNLOCK(&dlListLock);

			(*iter->func)(iter->data);
			free(iter);
			STARPU_PTHREAD_MUTEX_LOCK(&dlListLock);
		}

		/* First unlock the list! this is important */
		STARPU_PTHREAD_MUTEX_UNLOCK(&dlWorkLock);
		STARPU_PTHREAD_MUTEX_UNLOCK(&dlListLock);

		return 1;
	}
	STARPU_ASSERT(ret == EBUSY);
	return 0;
}

#endif

#else // LOCK_OR_DELEGATE

starpu_pthread_mutex_t commute_global_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

#endif

/* This function find a node that contains the parameter j as job and remove it from the list
 * the function return 0 if a node was found and deleted, 1 otherwise
 */
static unsigned remove_job_from_requester_list(struct _starpu_data_requester_list* req_list, struct _starpu_job * j)
{
	struct _starpu_data_requester * iter = _starpu_data_requester_list_begin(req_list);//_head;
	while(iter != _starpu_data_requester_list_end(req_list) && iter->j != j)
	{
		iter = _starpu_data_requester_list_next(iter); // iter = iter->_next;
	}
	if(iter)
	{
		_starpu_data_requester_list_erase(req_list, iter);
		return 0;
	}
	return 1;
}

#ifdef LOCK_OR_DELEGATE
/* These are the arguments passed to _submit_job_enforce_commute_deps */
struct starpu_enforce_commute_args
{
	struct _starpu_job *j;
	unsigned buf;
	unsigned nbuffers;
};

static void ___starpu_submit_job_enforce_commute_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers);
static void __starpu_submit_job_enforce_commute_deps(void* inData)
{
	struct starpu_enforce_commute_args* args = (struct starpu_enforce_commute_args*)inData;
	struct _starpu_job *j = args->j;
	unsigned buf		  = args->buf;
	unsigned nbuffers	 = args->nbuffers;
	/* we are in charge of freeing the args */
	free(args);
	args = NULL;
	inData = NULL;
	___starpu_submit_job_enforce_commute_deps(j, buf, nbuffers);
}

void _starpu_submit_job_enforce_commute_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
	struct starpu_enforce_commute_args* args = (struct starpu_enforce_commute_args*)malloc(sizeof(struct starpu_enforce_commute_args));
	args->j = j;
	args->buf = buf;
	args->nbuffers = nbuffers;
	/* The function will delete args */
	_starpu_LockOrDelegatePostOrPerform(&__starpu_submit_job_enforce_commute_deps, args);
}

static void ___starpu_submit_job_enforce_commute_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
#else // LOCK_OR_DELEGATE
void _starpu_submit_job_enforce_commute_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
	STARPU_PTHREAD_MUTEX_LOCK(&commute_global_mutex);
#endif

	const unsigned nb_non_commute_buff = buf;
	unsigned idx_buf_commute;
	unsigned all_commutes_available = 1;

	for (idx_buf_commute = nb_non_commute_buff; idx_buf_commute < nbuffers; idx_buf_commute++)
	{
		starpu_data_handle_t handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute);
		enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_commute);

		if (idx_buf_commute && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute-1)==handle))
			/* We have already requested this data, skip it. This
			 * depends on ordering putting writes before reads, see
			 * _starpu_compar_handles.  */
			continue;
		/* we post all commute  */
		STARPU_ASSERT(mode & STARPU_COMMUTE);

		_starpu_spin_lock(&handle->header_lock);
		if(handle->refcnt == 0)
		{
			handle->refcnt += 1;
			handle->busy_count += 1;
			handle->current_mode = mode;
			_starpu_spin_unlock(&handle->header_lock);
		}
		else
		{
			/* stop if an handle do not have a refcnt == 0 */
			_starpu_spin_unlock(&handle->header_lock);
			all_commutes_available = 0;
			break;
		}
	}
	if(all_commutes_available == 0)
	{
		/* Oups cancel all taken and put req in commute list */
		unsigned idx_buf_cancel;
		for (idx_buf_cancel = nb_non_commute_buff; idx_buf_cancel < idx_buf_commute ; idx_buf_cancel++)
		{
			starpu_data_handle_t cancel_handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel);

			if (idx_buf_cancel && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel-1)==cancel_handle))
				continue;

			_starpu_spin_lock(&cancel_handle->header_lock);
			/* reset the counter because finally we do not take the data */
			STARPU_ASSERT(cancel_handle->refcnt == 1);
			cancel_handle->refcnt -= 1;
			_starpu_spin_unlock(&cancel_handle->header_lock);
		}

		for (idx_buf_cancel = nb_non_commute_buff; idx_buf_cancel < nbuffers ; idx_buf_cancel++)
		{
			starpu_data_handle_t cancel_handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel);
			enum starpu_data_access_mode cancel_mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_cancel);

			STARPU_ASSERT(cancel_mode & STARPU_COMMUTE);

			struct _starpu_data_requester *r = _starpu_data_requester_new();
			r->mode = cancel_mode;
			r->is_requested_by_codelet = 1;
			r->j = j;
			r->buffer_index = idx_buf_cancel;
			r->ready_data_callback = NULL;
			r->argcb = NULL;

			_starpu_spin_lock(&cancel_handle->header_lock);
			/* create list if needed */
			if(cancel_handle->commute_req_list == NULL)
				cancel_handle->commute_req_list = _starpu_data_requester_list_new();
			/* store node in list */
			_starpu_data_requester_list_push_front(cancel_handle->commute_req_list, r);
			/* inc the busy count if it has not been changed in the previous loop */
			if(idx_buf_commute <= idx_buf_cancel)
				cancel_handle->busy_count += 1;
			_starpu_spin_unlock(&cancel_handle->header_lock);
		}

#ifndef LOCK_OR_DELEGATE
		STARPU_PTHREAD_MUTEX_UNLOCK(&commute_global_mutex);
#endif
		return 1;
	}

	// all_commutes_available is true
	_starpu_push_task(j);
#ifndef LOCK_OR_DELEGATE
	STARPU_PTHREAD_MUTEX_UNLOCK(&commute_global_mutex);
#endif
	return 0;
}

#ifdef LOCK_OR_DELEGATE
void ___starpu_notify_commute_dependencies(starpu_data_handle_t handle);
void __starpu_notify_commute_dependencies(void* inData)
{
	starpu_data_handle_t handle = (starpu_data_handle_t)inData;
	___starpu_notify_commute_dependencies(handle);
}
void _starpu_notify_commute_dependencies(starpu_data_handle_t handle)
{
	_starpu_LockOrDelegatePostOrPerform(&__starpu_notify_commute_dependencies, handle);
}
void ___starpu_notify_commute_dependencies(starpu_data_handle_t handle)
{
#else // LOCK_OR_DELEGATE
void _starpu_notify_commute_dependencies(starpu_data_handle_t handle)
{
	STARPU_PTHREAD_MUTEX_LOCK(&commute_global_mutex);
#endif
	/* Since the request has been posted the handle may have been proceed and released */
	if(handle->commute_req_list == NULL)
	{
#ifndef LOCK_OR_DELEGATE
		STARPU_PTHREAD_MUTEX_UNLOCK(&commute_global_mutex);
#endif
		return 1;
	}
	/* no one has the right to work on commute_req_list without a lock on commute_global_mutex
	   so we do not need to lock the handle for safety */
	struct _starpu_data_requester *r;
	r = _starpu_data_requester_list_begin(handle->commute_req_list); //_head;
	while(r)
	{
		struct _starpu_job* j = r->j;
		STARPU_ASSERT(r->mode & STARPU_COMMUTE);
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
		unsigned nb_non_commute_buff;
		/* find the position of commute buffers */
		for (nb_non_commute_buff = 0; nb_non_commute_buff < nbuffers; nb_non_commute_buff++)
		{
			starpu_data_handle_t handle_commute = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, nb_non_commute_buff);
			if (nb_non_commute_buff && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, nb_non_commute_buff-1) == handle_commute))
				/* We have already requested this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles.  */
				continue;
			enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, nb_non_commute_buff);
			if(mode & STARPU_COMMUTE)
			{
				break;
			}
		}

		unsigned idx_buf_commute;
		unsigned all_commutes_available = 1;

		for (idx_buf_commute = nb_non_commute_buff; idx_buf_commute < nbuffers; idx_buf_commute++)
		{
			starpu_data_handle_t handle_commute = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute);
			if (idx_buf_commute && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute-1)==handle_commute))
				/* We have already requested this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles.  */
				continue;
			/* we post all commute  */
			enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_commute);
			STARPU_ASSERT(mode & STARPU_COMMUTE);

			_starpu_spin_lock(&handle_commute->header_lock);
			if(handle_commute->refcnt != 0)
			{
				/* handle is not available */
				_starpu_spin_unlock(&handle_commute->header_lock);
				all_commutes_available = 0;
				break;
			}
			/* mark the handle as taken */
			handle_commute->refcnt += 1;
			handle_commute->current_mode = mode;
			_starpu_spin_unlock(&handle_commute->header_lock);
		}

		if(all_commutes_available)
		{
			for (idx_buf_commute = nb_non_commute_buff; idx_buf_commute < nbuffers; idx_buf_commute++)
			{
				starpu_data_handle_t handle_commute = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute);
				if (idx_buf_commute && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute-1)==handle_commute))
					continue;
				/* we post all commute  */
				enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_commute);
				STARPU_ASSERT(mode & STARPU_COMMUTE);

				_starpu_spin_lock(&handle_commute->header_lock);
				STARPU_ASSERT(handle_commute->refcnt == 1);
				STARPU_ASSERT( handle_commute->busy_count >= 1);
				STARPU_ASSERT( handle_commute->current_mode == mode);
				const unsigned correctly_deleted = remove_job_from_requester_list(handle_commute->commute_req_list, j);
				STARPU_ASSERT(correctly_deleted == 0);
				if(_starpu_data_requester_list_empty(handle_commute->commute_req_list)) // If size == 0
				{
					_starpu_data_requester_list_delete(handle_commute->commute_req_list);
					handle_commute->commute_req_list = NULL;
				}
				_starpu_spin_unlock(&handle_commute->header_lock);
			}
			/* delete list node */
			_starpu_data_requester_delete(r);

			/* push the task */
			_starpu_push_task(j);

			/* release global mutex */
#ifndef LOCK_OR_DELEGATE
			STARPU_PTHREAD_MUTEX_UNLOCK(&commute_global_mutex);
#endif
			/* We need to lock when returning 0 */
			return 0;
		}
		else
		{
			unsigned idx_buf_cancel;
			/* all handles are not available - revert the mark */
			for (idx_buf_cancel = nb_non_commute_buff; idx_buf_cancel < idx_buf_commute ; idx_buf_cancel++)
			{
				starpu_data_handle_t cancel_handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel);
				_starpu_spin_lock(&cancel_handle->header_lock);
				STARPU_ASSERT(cancel_handle->refcnt == 1);
				cancel_handle->refcnt -= 1;
				_starpu_spin_unlock(&cancel_handle->header_lock);
			}
		}

		r = r->_next;
	}
	/* no task has been pushed */
#ifndef LOCK_OR_DELEGATE
	STARPU_PTHREAD_MUTEX_UNLOCK(&commute_global_mutex);
#endif
	return 1;
}
