/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

/*
 * We have a kind of dining philosophers problem: various tasks are accessing
 * various data concurrently in different modes: STARPU_R, STARPU_RW, STARPU_W,
 * STARPU_SCRATCH and STARPU_REDUX. STARPU_RW is managed as a STARPU_W access.
 * We have the following constraints:
 *
 * - A single STARPU_W access is allowed at a time.
 * - Concurrent STARPU_R accesses are allowed.
 * - Concurrent STARPU_SCRATCH accesses are allowed.
 * - Concurrent STARPU_REDUX accesses are allowed.
 *
 * What we do here is implementing the Dijkstra solutions: handles are sorted
 * by pointer value order, and tasks call
 * _starpu_attempt_to_submit_data_request for each requested data in that order
 * (see _starpu_sort_task_handles call in _starpu_submit_job_enforce_data_deps).
 *
 * _starpu_attempt_to_submit_data_request will either:
 * - obtain access to the data, and thus the task can proceed with acquiring
 *   other data (see _submit_job_enforce_data_deps)
 * - queue a request on the data handle
 *
 * When a task finishes, it calls _starpu_notify_data_dependencies for each
 * data, to free its acquisitions. This will look whether the first queued
 * request can be fulfilled, and in such case make the task try to acquire its
 * next data.
 *
 * The same mechanism is used for application data aquisition
 * (starpu_data_acquire).
 */

// WIP_COMMUTE Begin

//#define NO_LOCK_OR_DELEGATE

/* Here are the high level algorithms which have been discussed in order
 * to manage the commutes.
    Pour chaque handle h en commute:
        mutex_lock(&arbiter)
        relâcher h
        Pour chaque tâche Tc en attente sur le handle:
            // Juste tester si on peut prendre:
            Pour chaque donnée Tc_h qu’il attend:
                Si Tc_h est occupé, goto fail
            // Vraiment prendre
            Pour chaque donnée Tc_h qu’il attend:
                lock(Tc_h)
                prendre(h) (il devrait être encore disponible si tout le reste utilise bien le mutex arbiter)
                lock(Tc_h)
            // on a trouvé quelqu’un, on a fini!
            _starpu_push_task(Tc);
            break;
            fail:
                // Pas de bol, on essaie une autre tâche
                continue;
        // relâcher un peu le mutex arbiter de temps en temps
        mutex_unlock(&arbiter)

    mutex_lock(&arbiter)
    Pour chaque handle h en commute:
        lock(h)
        essayer de prendre h, si échec goto fail;
        unlock(h)
        mutex_unlock(&arbiter)
        return 0

        fail:
            // s’enregistrer sur la liste des requêtes de h
            Pour chaque handle déjà pris:
                lock(handle)
                relâcher handle
                unlock(handle)
     mutex_unlock(&arbiter)
 */

/* Here are the LockOrDelegate functions
 * There are two version depending on the support of the compare and exchange
 * support from the compiler
 */

#include <assert.h>
#include <stdlib.h>

#ifndef NO_LOCK_OR_DELEGATE

/* A LockOrDelegate task list */
struct LockOrDelegateListNode
{
	int (*func)(void*);
	void* data;
	struct LockOrDelegateListNode* next;
};

/* If the compiler support C11 and the usage of atomic functions */
#if (201112L <= __STDC_VERSION__) && !(defined(__STDC_NO_ATOMICS__))

#include <stdatomic.h>

/* To know the number of task to perform and attributes the tickets */
atomic_int dlAtomicCounter;
/* The list of task to perform */
_Atomic struct LockOrDelegateListNode* dlListHead;

/* Post a task to perfom if possible, otherwise put it in the list
 * If we can perfom this task, we may also perfom all the tasks in the list
 * This function return 1 if the task (and maybe some others) has been done
 * by the calling thread and 0 otherwise (if the task has just been put in the list)
 */
int LockOrDelegatePostOrPerform(int (*func)(void*), void* data)
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
			assert(atomic_load(&dlAtomicCounter) > 0);

			/* Dec ticket and see if something else has to be done */
			int removedPosition = atomic_load(&dlAtomicCounter);
			while(!atomic_compare_exchange_weak(&dlAtomicCounter, &removedPosition,removedPosition-1));
			if(removedPosition-1 == 0)
		{
				break;
			}

			/* Get the next task */
			struct LockOrDelegateListNode* removedNode = (struct LockOrDelegateListNode*)atomic_load(&dlListHead);
			// Maybe it has not been pushed yet (listHead.load() == nullptr)
			while((removedNode = (struct LockOrDelegateListNode*)atomic_load(&dlListHead)) == NULL || !atomic_compare_exchange_weak(&dlListHead, &removedNode,removedNode->next))
			;
			assert(removedNode);
			/* call the task */
			(*removedNode->func)(removedNode->data);
			// Delete node
			free(removedNode);
		}

		return 1;
	}

	struct LockOrDelegateListNode* newNode = (struct LockOrDelegateListNode*)malloc(sizeof(struct LockOrDelegateListNode));
	assert(newNode);
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

#include <pthread.h>
#include <errno.h>

/* The list of task to perform */
struct LockOrDelegateListNode* dlTaskListHead = NULL;

/* To protect the list of tasks */
pthread_mutex_t dlListLock = PTHREAD_MUTEX_INITIALIZER;
/* To know who is responsible to compute all the tasks */
pthread_mutex_t dlWorkLock = PTHREAD_MUTEX_INITIALIZER;

/* Post a task to perfom if possible, otherwise put it in the list
 * If we can perfom this task, we may also perfom all the tasks in the list
 * This function return 1 if the task (and maybe some others) has been done
 * by the calling thread and 0 otherwise (if the task has just been put in the list)
 */
int LockOrDelegatePostOrPerform(int (*func)(void*), void* data)
{
	/* We could avoid to allocate if we will be responsible but for simplicity
	 * we always push the task in the list */
	struct LockOrDelegateListNode* newNode = (struct LockOrDelegateListNode*)malloc(sizeof(struct LockOrDelegateListNode));
	assert(newNode);
	newNode->data = data;
	newNode->func = func;

	/* insert the node */
	int ret = pthread_mutex_lock(&dlListLock);
	assert(ret == 0);
	newNode->next = dlTaskListHead;
	dlTaskListHead = newNode;
	ret = pthread_mutex_unlock(&dlListLock);
	assert(ret == 0);

	/* See if we can compute all the tasks */
	if((ret = pthread_mutex_trylock(&dlWorkLock)) == 0)
	{
		ret = pthread_mutex_lock(&dlListLock);
		assert(ret == 0);
		while(dlTaskListHead != 0)
		{
			struct LockOrDelegateListNode* iter = dlTaskListHead;
			dlTaskListHead = dlTaskListHead->next;
			ret = pthread_mutex_unlock(&dlListLock);
			assert(ret == 0);

			(*iter->func)(iter->data);
			free(iter);
			ret = pthread_mutex_lock(&dlListLock);
			assert(ret == 0);
		}

		/* First unlock the list! this is important */
		ret = pthread_mutex_unlock(&dlWorkLock);
		assert(ret == 0);
		ret = pthread_mutex_unlock(&dlListLock);
		assert(ret == 0);

		return 1;
	}
	assert(ret == EBUSY);
	return 0;
}

#endif

#else // NO_LOCK_OR_DELEGATE

pthread_mutex_t commute_global_mutex = PTHREAD_MUTEX_INITIALIZER;

#endif

/* This function find a node that contains the parameter j as job and remove it from the list
 * the function return 0 if a node was found and deleted, 1 otherwise
 */
unsigned remove_job_from_requester_list(struct _starpu_data_requester_list* req_list, struct _starpu_job * j)
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

#ifndef NO_LOCK_OR_DELEGATE

/* These are the arguments passed to _submit_job_enforce_commute_deps */
struct EnforceCommuteArgs
{
	struct _starpu_job *j;
	unsigned buf;
	unsigned nbuffers;
};

int _submit_job_enforce_commute_deps(void* inData)
{
	struct EnforceCommuteArgs* args = (struct EnforceCommuteArgs*)inData;
	struct _starpu_job *j = args->j;
	unsigned buf		  = args->buf;
	unsigned nbuffers	 = args->nbuffers;
	/* we are in charge of freeing the args */
	free(args);
	args = NULL;
	inData = NULL;
#else // NO_LOCK_OR_DELEGATE
int _submit_job_enforce_commute_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers)
{
	int ret = pthread_mutex_lock(&commute_global_mutex);
	assert(ret == 0);
#endif

	const unsigned nb_non_commute_buff = buf;
	unsigned idx_buf_commute;
	unsigned all_commutes_available = 1;

	for (idx_buf_commute = nb_non_commute_buff; idx_buf_commute < nbuffers; idx_buf_commute++)
	{
		if (idx_buf_commute && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute-1)==_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute)))
			continue;
		/* we post all commute  */
		starpu_data_handle_t handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute);
		enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_commute);
		assert(mode & STARPU_COMMUTE);

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
			if (idx_buf_cancel && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel-1)==_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel)))
				continue;

			starpu_data_handle_t cancel_handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel);
			_starpu_spin_lock(&cancel_handle->header_lock);
			/* reset the counter because finally we do not take the data */
			assert(cancel_handle->refcnt == 1);
			cancel_handle->refcnt -= 1;
			_starpu_spin_unlock(&cancel_handle->header_lock);
		}

		for (idx_buf_cancel = nb_non_commute_buff; idx_buf_cancel < nbuffers ; idx_buf_cancel++)
		{
			starpu_data_handle_t cancel_handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_cancel);
			enum starpu_data_access_mode cancel_mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_cancel);

			assert(cancel_mode & STARPU_COMMUTE);

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

#ifdef NO_LOCK_OR_DELEGATE
		ret = pthread_mutex_unlock(&commute_global_mutex);
		assert(ret == 0);
#endif
		return 1;
	}

	// all_commutes_available is true
	_starpu_push_task(j);
#ifdef NO_LOCK_OR_DELEGATE
	ret = pthread_mutex_unlock(&commute_global_mutex);
	assert(ret == 0);
#endif
	return 0;
}

#ifndef NO_LOCK_OR_DELEGATE
int _starpu_notify_commute_dependencies(void* inData)
{
	starpu_data_handle_t handle = (starpu_data_handle_t)inData;
#else // NO_LOCK_OR_DELEGATE
int _starpu_notify_commute_dependencies(starpu_data_handle_t handle)
{
	int ret = pthread_mutex_lock(&commute_global_mutex);
	assert(ret == 0);
#endif
	/* Since the request has been posted the handle may have been proceed and released */
	if(handle->commute_req_list == NULL)
	{
#ifdef NO_LOCK_OR_DELEGATE
		ret = pthread_mutex_unlock(&commute_global_mutex);
		assert(ret == 0);
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
		assert(r->mode & STARPU_COMMUTE);
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
		unsigned nb_non_commute_buff;
		/* find the position of commute buffers */
		for (nb_non_commute_buff = 0; nb_non_commute_buff < nbuffers; nb_non_commute_buff++)
		{
			if (nb_non_commute_buff && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, nb_non_commute_buff-1) == _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, nb_non_commute_buff)))
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
			if (idx_buf_commute && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute-1)==_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute)))
				continue;
			/* we post all commute  */
			starpu_data_handle_t handle_commute = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute);
			enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_commute);
			assert(mode & STARPU_COMMUTE);

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
				if (idx_buf_commute && (_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute-1)==_STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute)))
					continue;
				/* we post all commute  */
				starpu_data_handle_t handle_commute = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, idx_buf_commute);
				enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, idx_buf_commute);
				assert(mode & STARPU_COMMUTE);

				_starpu_spin_lock(&handle_commute->header_lock);
				assert(handle_commute->refcnt == 1);
				assert( handle_commute->busy_count >= 1);
				assert( handle_commute->current_mode == mode);
				const unsigned correctly_deleted = remove_job_from_requester_list(handle_commute->commute_req_list, j);
				assert(correctly_deleted == 0);
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
#ifdef NO_LOCK_OR_DELEGATE
			ret = pthread_mutex_unlock(&commute_global_mutex);
			assert(ret == 0);
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
				assert(cancel_handle->refcnt == 1);
				cancel_handle->refcnt -= 1;
				_starpu_spin_unlock(&cancel_handle->header_lock);
			}
		}

		r = r->_next;
	}
	/* no task has been pushed */
#ifdef NO_LOCK_OR_DELEGATE
	ret = pthread_mutex_unlock(&commute_global_mutex);
	assert(ret == 0);
#endif
	return 1;
}

// WIP_COMMUTE End

/*
 * Check to see whether the first queued request can proceed, and return it in
 * such case.
 */
/* the header lock must be taken by the caller */
static struct _starpu_data_requester *may_unlock_data_req_list_head(starpu_data_handle_t handle)
{
	struct _starpu_data_requester_list *req_list;

	if (handle->reduction_refcnt > 0)
	{
		req_list = handle->reduction_req_list;
	}
	else
	{
		if (_starpu_data_requester_list_empty(handle->reduction_req_list))
			req_list = handle->req_list;
		else
			req_list = handle->reduction_req_list;
	}

	/* if there is no one to unlock ... */
	if (_starpu_data_requester_list_empty(req_list))
		return NULL;

	/* if there is no reference to the data anymore, we can use it */
	if (handle->refcnt == 0)
		return _starpu_data_requester_list_pop_front(req_list);

	/* Already writing to it, do not let another write access through */
	// WIP_COMMUTE Was
	// if (handle->current_mode == STARPU_W)
	//	return NULL;
	// WIP_COMMUTE Begin
	if (handle->current_mode & STARPU_W)
		return NULL;
	// WIP_COMMUTE End

	/* data->current_mode == STARPU_R, so we can process more readers */
	struct _starpu_data_requester *r = _starpu_data_requester_list_front(req_list);

	enum starpu_data_access_mode r_mode = r->mode;
	// WIP_COMMUTE Was
	// if (r_mode == STARPU_RW)
	//	r_mode = STARPU_W;
	// WIP_COMMUTE Begin
	if (r_mode & STARPU_RW)
		r_mode &= ~STARPU_R;
	// WIP_COMMUTE End

	/* If this is a STARPU_R, STARPU_SCRATCH or STARPU_REDUX type of
	 * access, we only proceed if the current mode is the same as the
	 * requested mode. */
	if (r_mode == handle->current_mode)
		return _starpu_data_requester_list_pop_front(req_list);
	else
		return NULL;
}

/* Try to submit a data request, in case the request can be processed
 * immediatly, return 0, if there is still a dependency that is not compatible
 * with the current mode, the request is put in the per-handle list of
 * "requesters", and this function returns 1. */
static unsigned _starpu_attempt_to_submit_data_request(unsigned request_from_codelet,
						       starpu_data_handle_t handle, enum starpu_data_access_mode current_mode,
						       void (*callback)(void *), void *argcb,
						       struct _starpu_job *j, unsigned buffer_index)
{
	// WIP_COMMUTE Begin
	enum starpu_data_access_mode mode = (current_mode & ~STARPU_COMMUTE);
	// WIP_COMMUTE End

	if (mode == STARPU_RW)
		mode = STARPU_W;

	/* Take the lock protecting the header. We try to do some progression
	 * in case this is called from a worker, otherwise we just wait for the
	 * lock to be available. */
	if (request_from_codelet)
	{
		int cpt = 0;
		while (cpt < STARPU_SPIN_MAXTRY && _starpu_spin_trylock(&handle->header_lock))
		{
			cpt++;
			_starpu_datawizard_progress(_starpu_memory_node_get_local_key(), 0);
		}
		if (cpt == STARPU_SPIN_MAXTRY)
			_starpu_spin_lock(&handle->header_lock);
	}
	else
	{
		_starpu_spin_lock(&handle->header_lock);
	}

	/* If we have a request that is not used for the reduction, and that a
	 * reduction is pending, we put it at the end of normal list, and we
	 * use the reduction_req_list instead */
	unsigned pending_reduction = (handle->reduction_refcnt > 0);
	unsigned frozen = 0;

	/* If we are currently performing a reduction, we freeze any request
	 * that is not explicitely a reduction task. */
	unsigned is_a_reduction_task = (request_from_codelet && j->reduction_task);

	if (pending_reduction && !is_a_reduction_task)
		frozen = 1;

	/* If there is currently nobody accessing the piece of data, or it's
	 * not another writter and if this is the same type of access as the
	 * current one, we can proceed. */
	unsigned put_in_list = 1;

	// WIP_COMMUTE Was
	//enum starpu_data_access_mode previous_mode = handle->current_mode;
	// WIP_COMMUTE Begin
	enum starpu_data_access_mode previous_mode = (handle->current_mode & ~STARPU_COMMUTE);
	// WIP_COMMUTE End

	if (!frozen && ((handle->refcnt == 0) || (!(mode == STARPU_W) && (handle->current_mode == mode))))
	{
		/* Detect whether this is the end of a reduction phase */
			/* We don't want to start multiple reductions of the
			 * same handle at the same time ! */

		if ((handle->reduction_refcnt == 0) && (previous_mode == STARPU_REDUX) && (mode != STARPU_REDUX))
		{
			_starpu_data_end_reduction_mode(handle);

			/* Since we need to perform a mode change, we freeze
			 * the request if needed. */
			put_in_list = (handle->reduction_refcnt > 0);
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
		r->mode = current_mode;
		r->is_requested_by_codelet = request_from_codelet;
		r->j = j;
		r->buffer_index = buffer_index;
		r->ready_data_callback = callback;
		r->argcb = argcb;

		/* We put the requester in a specific list if this is a reduction task */
		struct _starpu_data_requester_list *req_list =
			is_a_reduction_task?handle->reduction_req_list:handle->req_list;

		_starpu_data_requester_list_push_back(req_list, r);

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
			handle->current_mode = current_mode;

		if ((mode == STARPU_REDUX) && (previous_mode != STARPU_REDUX))
			_starpu_data_start_reduction_mode(handle);

		/* success */
		put_in_list = 0;
	}

	_starpu_spin_unlock(&handle->header_lock);
	return put_in_list;

}

unsigned _starpu_attempt_to_submit_data_request_from_apps(starpu_data_handle_t handle, enum starpu_data_access_mode mode,
							  void (*callback)(void *), void *argcb)
{
	return _starpu_attempt_to_submit_data_request(0, handle, mode, callback, argcb, NULL, 0);
}

static unsigned attempt_to_submit_data_request_from_job(struct _starpu_job *j, unsigned buffer_index)
{
	/* Note that we do not access j->task->handles, but j->ordered_buffers
	 * which is a sorted copy of it. */
	starpu_data_handle_t handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, buffer_index);
	// WIP_COMMUTE Was
	// enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, buffer_index) & ~STARPU_COMMUTE;
	// WIP_COMMUTE Begin
	enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, buffer_index);
	// WIP_COMMUTE End

	return _starpu_attempt_to_submit_data_request(1, handle, mode, NULL, NULL, j, buffer_index);
}

/* Acquire all data of the given job, one by one in handle pointer value order
 */
static unsigned _submit_job_enforce_data_deps(struct _starpu_job *j, unsigned start_buffer_index)
{
	unsigned buf;

	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
	for (buf = start_buffer_index; buf < nbuffers; buf++)
	{
		if (buf)
		{
			starpu_data_handle_t handle_m1 = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, buf-1);
			starpu_data_handle_t handle = _STARPU_JOB_GET_ORDERED_BUFFER_HANDLE(j, buf);
			if (handle_m1 == handle)
				/* We have already requested this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles.  */
				continue;
		}

                j->task->status = STARPU_TASK_BLOCKED_ON_DATA;

		// WIP_COMMUTE Begin
		enum starpu_data_access_mode mode = _STARPU_JOB_GET_ORDERED_BUFFER_MODE(j, buf);
		if(mode & STARPU_COMMUTE)
		{
			/* We arrived on the commute we stop and do not proceed as usual */
			break;
		}
		// WIP_COMMUTE End

                if (attempt_to_submit_data_request_from_job(j, buf))
		{
			return 1;
                }
	}

	// WIP_COMMUTE Begin
	/* We arrive on the commutes */
	if(buf != nbuffers)
	{
#ifndef NO_LOCK_OR_DELEGATE
		struct EnforceCommuteArgs* args = (struct EnforceCommuteArgs*)malloc(sizeof(struct EnforceCommuteArgs));
		args->j = j;
		args->buf = buf;
		args->nbuffers = nbuffers;
		/* The function will delete args */
		LockOrDelegatePostOrPerform(&_submit_job_enforce_commute_deps, args);
#else // NO_LOCK_OR_DELEGATE
		_submit_job_enforce_commute_deps(j, buf, nbuffers);
#endif
		return 1;
	}
	// WIP_COMMUTE End

	return 0;
}

/* Sort the data used by the given job by handle pointer value order, and
 * acquire them in that order */
unsigned _starpu_submit_job_enforce_data_deps(struct _starpu_job *j)
{
	struct starpu_codelet *cl = j->task->cl;

	if ((cl == NULL) || (STARPU_TASK_GET_NBUFFERS(j->task) == 0))
		return 0;

	/* Compute an ordered list of the different pieces of data so that we
	 * grab then according to a total order, thus avoiding a deadlock
	 * condition */
	unsigned i;
	for (i=0 ; i<STARPU_TASK_GET_NBUFFERS(j->task); i++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(j->task, i);
		_STARPU_JOB_SET_ORDERED_BUFFER_HANDLE(j, handle, i);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(j->task, i);
		_STARPU_JOB_SET_ORDERED_BUFFER_MODE(j, mode, i);
		int node = -1;
		if (j->task->cl->specific_nodes)
			node = STARPU_CODELET_GET_NODE(j->task->cl, i);
		_STARPU_JOB_SET_ORDERED_BUFFER_NODE(j, node, i);
	}

	_starpu_sort_task_handles(_STARPU_JOB_GET_ORDERED_BUFFERS(j), STARPU_TASK_GET_NBUFFERS(j->task));

	return _submit_job_enforce_data_deps(j, 0);
}

/* This request got fulfilled, continue with the other requests of the
 * corresponding job */
static unsigned unlock_one_requester(struct _starpu_data_requester *r)
{
	struct _starpu_job *j = r->j;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
	unsigned buffer_index = r->buffer_index;

	if (buffer_index + 1 < nbuffers)
		/* not all buffers are protected yet */
		return _submit_job_enforce_data_deps(j, buffer_index + 1);
	else
		return 0;
}

/* This is called when a task is finished with a piece of data
 * (or on starpu_data_release)
 *
 * The header lock must already be taken by the caller.
 * This may free the handle if it was lazily unregistered (1 is returned in
 * that case). The handle pointer thus becomes invalid for the caller.
 */
int _starpu_notify_data_dependencies(starpu_data_handle_t handle)
{
	_starpu_spin_checklocked(&handle->header_lock);
	/* A data access has finished so we remove a reference. */
	STARPU_ASSERT(handle->refcnt > 0);
	handle->refcnt--;
	STARPU_ASSERT(handle->busy_count > 0);
	handle->busy_count--;
	if (_starpu_data_check_not_busy(handle))
		/* Handle was destroyed, nothing left to do.  */
		return 1;

	/* In case there is a pending reduction, and that this is the last
	 * requester, we may go back to a "normal" coherency model. */
	if (handle->reduction_refcnt > 0)
	{
		//fprintf(stderr, "NOTIFY REDUCTION TASK RED REFCNT %d\n", handle->reduction_refcnt);
		handle->reduction_refcnt--;
		if (handle->reduction_refcnt == 0)
			_starpu_data_end_reduction_mode_terminate(handle);
	}

	struct _starpu_data_requester *r;
	while ((r = may_unlock_data_req_list_head(handle)))
	{
		/* STARPU_RW accesses are treated as STARPU_W */
		enum starpu_data_access_mode r_mode = r->mode;
		if (r_mode == STARPU_RW)
			r_mode = STARPU_W;

		int put_in_list = 1;
		if ((handle->reduction_refcnt == 0) && (handle->current_mode == STARPU_REDUX) && (r_mode != STARPU_REDUX))
		{
			_starpu_data_end_reduction_mode(handle);

			/* Since we need to perform a mode change, we freeze
			 * the request if needed. */
			put_in_list = (handle->reduction_refcnt > 0);
		}
		else
		{
			put_in_list = 0;
		}

		if (put_in_list)
		{
			/* We need to put the request back because we must
			 * perform a reduction before. */
			_starpu_data_requester_list_push_front(handle->req_list, r);
		}
		else
		{
			/* The data is now attributed to that request so we put a
			 * reference on it. */
			handle->refcnt++;
			handle->busy_count++;

			enum starpu_data_access_mode previous_mode = handle->current_mode;
			handle->current_mode = r_mode;

			/* In case we enter in a reduction mode, we invalidate all per
			 * worker replicates. Note that the "per_node" replicates are
			 * kept intact because we'll reduce a valid copy of the
			 * "per-node replicate" with the per-worker replicates .*/
			if ((r_mode == STARPU_REDUX) && (previous_mode != STARPU_REDUX))
				_starpu_data_start_reduction_mode(handle);

			_starpu_spin_unlock(&handle->header_lock);

			if (r->is_requested_by_codelet)
			{
				if (!unlock_one_requester(r))
					_starpu_push_task(r->j);
			}
			else
			{
				STARPU_ASSERT(r->ready_data_callback);

				/* execute the callback associated with the data requester */
				r->ready_data_callback(r->argcb);
			}

			_starpu_data_requester_delete(r);

			_starpu_spin_lock(&handle->header_lock);
			STARPU_ASSERT(handle->busy_count > 0);
			handle->busy_count--;
			if (_starpu_data_check_not_busy(handle))
				return 1;
		}
	}

	// WIP_COMMUTE Begin

	if(handle->refcnt == 0 && handle->commute_req_list != NULL)
	{
		/* We need to delloc current handle because it is currently locked
		 * but we alloc fist the global mutex and than the handles mutex
		 */
		_starpu_spin_unlock(&handle->header_lock);
#ifndef NO_LOCK_OR_DELEGATE
		LockOrDelegatePostOrPerform(&_starpu_notify_commute_dependencies, handle);
#else // NO_LOCK_OR_DELEGATE
		_starpu_notify_commute_dependencies(handle);
#endif
		/* We need to lock when returning 0 */
		return 1;
	}
	// WIP_COMMUTE End

	return 0;
}
