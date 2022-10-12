/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <datawizard/coherency.h>
#include <starpu_mpi_cache.h>

/*
 * One node sends the same data to several nodes. Gather them into a
 * "coop_sends", which then has a global view of all the required sends, and can
 * establish a diffusion tree by telling receiving nodes to retransmit what they
 * received (forwards) to others, and to others that they will receive from the
 * former (redirects).
 */

/* This is called after a request is finished processing, to release the data */
void _starpu_mpi_release_req_data(struct _starpu_mpi_req *req)
{
	if (!req->data_handle)
		return;

	if (_starpu_mpi_req_multilist_queued_coop_sends(req))
	{
		struct _starpu_mpi_coop_sends *coop_sends = req->coop_sends_head;
		assert(coop_sends != NULL);

		struct _starpu_mpi_data *mpi_data = coop_sends->mpi_data;
		int last;
		_starpu_spin_lock(&mpi_data->coop_lock);
		/* Part of a cooperative send, dequeue ourself from others */
		_starpu_mpi_req_multilist_erase_coop_sends(&coop_sends->reqs, req);
		last = _starpu_mpi_req_multilist_empty_coop_sends(&coop_sends->reqs);
		_starpu_spin_unlock(&mpi_data->coop_lock);
		if (last)
		{
			/* We were last, release data */
			free(coop_sends->reqs_array);
			free(coop_sends);
			starpu_data_release_on_node(req->data_handle, req->node);
		}
	}
	else
	{
		/* Trivial request */
		starpu_data_release_on_node(req->data_handle, req->node);
	}
}

/* The data was acquired in terms of dependencies, we can now look the
 * current state of the handle and decide which node we prefer for the data
 * fetch */
static void _starpu_mpi_coop_send_acquired_callback(void *arg, int *nodep, enum starpu_data_access_mode mode)
{
	struct _starpu_mpi_coop_sends *coop_sends = arg;
	int node = *nodep;

	if (node < 0)
		node = _starpu_mpi_choose_node(coop_sends->data_handle, mode);

	/* Record the node in the first req */
	_starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs)->node = node;

	*nodep = node;
}

/* Comparison function for getting qsort to put requests with high priority first */
static int _starpu_mpi_reqs_prio_compare(const void *a, const void *b)
{
	const struct _starpu_mpi_req * const *ra = a;
	const struct _starpu_mpi_req * const *rb = b;
	if ((*rb)->prio < (*ra)->prio)
		return -1;
	else if ((*rb)->prio == (*ra)->prio)
		return 0;
	else
		return 1;
}

/* Sort the requests by priority and build a diffusion tree. Actually does something only once per coop_sends bag. */
static void _starpu_mpi_coop_sends_optimize(struct _starpu_mpi_coop_sends *coop_sends)
{
	STARPU_ASSERT(coop_sends->n > 1);

	_starpu_spin_lock(&coop_sends->lock);
	if (!coop_sends->reqs_array)
	{
		unsigned n = coop_sends->n, i;
		struct _starpu_mpi_req *cur;
		struct _starpu_mpi_req **reqs;

		_STARPU_MPI_DEBUG(0, "handling cooperative sends %p for %u neighbours\n", coop_sends, n);

		/* Store them in an array */
		_STARPU_CALLOC(reqs, n, sizeof(*reqs));
		for (cur  = _starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs), i = 0;
		     cur != _starpu_mpi_req_multilist_end_coop_sends(&coop_sends->reqs);
		     cur  = _starpu_mpi_req_multilist_next_coop_sends(cur), i++)
			reqs[i] = cur;
		coop_sends->reqs_array = reqs;

		/* Sort them */
		qsort(reqs, n, sizeof(*reqs), _starpu_mpi_reqs_prio_compare);

#if 0
		/* And build the diffusion tree */
		_starpu_mpi_coop_sends_build_tree(coop_sends);
#endif
	}
	_starpu_spin_unlock(&coop_sends->lock);
}

/* This is called on completion of acquisition of data for a cooperative send */
static void _starpu_mpi_coop_sends_data_ready(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_coop_sends *coop_sends = arg;
	struct _starpu_mpi_data *mpi_data = coop_sends->mpi_data;
	struct _starpu_mpi_req *cur;
	unsigned node;

	/* Take the cooperative send bag out from more submissions */
	if (mpi_data->coop_sends == coop_sends)
	{
		_starpu_spin_lock(&mpi_data->coop_lock);
		if (mpi_data->coop_sends == coop_sends)
			mpi_data->coop_sends = NULL;
		_starpu_spin_unlock(&mpi_data->coop_lock);
	}

	/* Copy over the memory node number */
	cur = _starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs);
	node = cur->node;

	for (;
	     cur != _starpu_mpi_req_multilist_end_coop_sends(&coop_sends->reqs);
	     cur  = _starpu_mpi_req_multilist_next_coop_sends(cur))
	{
		cur->node = node;
		cur->pre_sync_jobid = coop_sends->pre_sync_jobid; // for tracing purposes
	}

	if (coop_sends->n == 1)
	{
		/* Trivial case, just submit it */
		_starpu_mpi_submit_ready_request(_starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs));
	}
	else
	{
		/* Build diffusion tree */
		_starpu_mpi_coop_sends_optimize(coop_sends);

		/* And submit them */
		if (STARPU_TEST_AND_SET(&coop_sends->redirects_sent, 1) == 0)
		{
			mpi_data->nb_future_sends = 0;
			_starpu_mpi_submit_coop_sends(coop_sends, 1, 1);
		}
		else
			_starpu_mpi_submit_coop_sends(coop_sends, 0, 1);
	}
	_STARPU_MPI_LOG_OUT();
}

/* This is called when we want to stop including new members in a cooperative send,
 * either because we know there won't be any other members due to the algorithm
 * or because the value has changed.  */
static void _starpu_mpi_coop_send_flush(struct _starpu_mpi_coop_sends *coop_sends)
{
	if (!coop_sends || coop_sends->n == 1)
		return;

	/* Build diffusion tree */
	_starpu_mpi_coop_sends_optimize(coop_sends);

	/* And submit them */
	if (STARPU_TEST_AND_SET(&coop_sends->redirects_sent, 1) == 0)
		_starpu_mpi_submit_coop_sends(coop_sends, 1, 0);
}

/* This is called when a write to the data was just submitted, which means we
 * can't make future sends cooperate with past sends since it's not the same value
 */
void _starpu_mpi_data_flush(starpu_data_handle_t data_handle)
{
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;
	struct _starpu_mpi_coop_sends *coop_sends;
	if (!mpi_data)
		return;

	_starpu_spin_lock(&mpi_data->coop_lock);
	coop_sends = mpi_data->coop_sends;
	if (coop_sends)
		mpi_data->coop_sends = NULL;
	_starpu_spin_unlock(&mpi_data->coop_lock);
	if (coop_sends)
	{
		_STARPU_MPI_DEBUG(0, "%p: data written to, flush cooperative sends %p\n", data_handle, coop_sends);
		_starpu_mpi_coop_send_flush(coop_sends);
	}
}

/* Test whether a request is compatible with a cooperative send */
static int _starpu_mpi_coop_send_compatible(struct _starpu_mpi_req *req, struct _starpu_mpi_coop_sends *coop_sends)
{
	if (!_starpu_cache_enabled)
	{
		/* If MPI cache isn't enabled, duplicates can appear in the list
		 * of recipients.
		 * Presence of duplicates can lead to deadlocks, so if adding
		 * this req request to the coop_sends will introduce
		 * duplicates, we consider this req as incompatible.
		 *
		 * This a requirement coming from the NewMadeleine
		 * implementation. If one day, there is a MPI implementation,
		 * this constraint might move to the NewMadeleine backend.
		 *
		 * See mpi/tests/coop_cache.c for a test case.
		 */
		int inserting_dest = req->node_tag.node.rank;
		struct _starpu_mpi_req* cur = NULL;
		for (cur = _starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs);
		cur != _starpu_mpi_req_multilist_end_coop_sends(&coop_sends->reqs);
		cur  = _starpu_mpi_req_multilist_next_coop_sends(cur))
		{
			if (cur->node_tag.node.rank == inserting_dest)
			{
				return 0;
			}
		}
	}

	struct _starpu_mpi_req *prevreq = _starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs);
	return /* we can cope with tag being different */
	          prevreq->node_tag.node.comm == req->node_tag.node.comm
	       && prevreq->sequential_consistency == req->sequential_consistency;
}


void _starpu_mpi_coop_send(starpu_data_handle_t data_handle, struct _starpu_mpi_req *req, enum starpu_data_access_mode mode, int sequential_consistency)
{
	struct _starpu_mpi_data *mpi_data = _starpu_mpi_data_get(data_handle);
	struct _starpu_mpi_coop_sends *coop_sends = NULL, *tofree = NULL;
	int done = 0, queue, first = 1;

	/* Try to add ourself to something existing, otherwise create one.  */
	while (!done)
	{
		_starpu_spin_lock(&mpi_data->coop_lock);
		if (mpi_data->coop_sends)
		{
			/* Already something, check we are coherent with it */
			queue = _starpu_mpi_coop_send_compatible(req, mpi_data->coop_sends);
			if (queue)
			{
				/* Yes, queue ourself there */
				if (coop_sends)
				{
					/* Remove ourself from what we created for ourself first */

					/* Note 2022-09-21: according to code coverage(see
					 * https://files.inria.fr/starpu/testing/master/coverage/mpi/src/starpu_mpi_coop_sends.c.gcov.html),
					 * this block is dead code. */
					_starpu_mpi_req_multilist_erase_coop_sends(&coop_sends->reqs, req);
					tofree = coop_sends;
				}
				coop_sends = mpi_data->coop_sends;
				_STARPU_MPI_DEBUG(0, "%p: add to cooperative sends %p, dest %d\n", data_handle, coop_sends, req->node_tag.node.rank);

				/* Get the pre_sync_jobid of the first send request, to build a coherent DAG in the traces: */
				struct _starpu_mpi_req *firstreq;
				firstreq = _starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs);
				req->pre_sync_jobid = firstreq->pre_sync_jobid;

				_starpu_mpi_req_multilist_push_back_coop_sends(&coop_sends->reqs, req);
				coop_sends->n++;
				req->coop_sends_head = coop_sends;
				first = 0;
				done = 1;
			}
			else
			{
				/* Nope, incompatible, send it as a regular point-to-point communication
				 *
				 * TODO: this could be improved by having several coop_sends "bags" available
				 * simultaneously, which will trigger different broadcasts. */
				_starpu_spin_unlock(&mpi_data->coop_lock);

				_starpu_mpi_isend_irecv_common(req, mode, sequential_consistency);
				return;
			}
		}
		else if (coop_sends)
		{
			/* Nobody else and we have allocated one, we're first! */
			_STARPU_MPI_DEBUG(0, "%p: new cooperative sends %p for tag %"PRIi64", dest %d\n", data_handle, coop_sends, req->node_tag.data_tag, req->node_tag.node.rank);
			mpi_data->coop_sends = coop_sends;
			first = 1;
			done = 1;
		}
		_starpu_spin_unlock(&mpi_data->coop_lock);

		if (!done && !coop_sends)
		{
			/* Didn't find something to join, create one out of critical section */
			_STARPU_MPI_CALLOC(coop_sends, 1, sizeof(*coop_sends));
			coop_sends->data_handle = data_handle;
			coop_sends->redirects_sent = 0;
			coop_sends->n = 1;
			_starpu_mpi_req_multilist_head_init_coop_sends(&coop_sends->reqs);
			_starpu_mpi_req_multilist_push_back_coop_sends(&coop_sends->reqs, req);
			_starpu_spin_init(&coop_sends->lock);
			req->coop_sends_head = coop_sends;
			coop_sends->mpi_data = mpi_data;
		}
		/* We at worse do two iteration */
		STARPU_ASSERT(done || coop_sends);
	}

	STARPU_ASSERT(coop_sends);

	/* In case we created one for nothing after all */
	free(tofree);

	if ((mpi_data->nb_future_sends != 0 && mpi_data->nb_future_sends == coop_sends->n) || (mpi_data->nb_future_sends == 0 && first))
		/* We were first, we are responsible for acquiring the data for everybody */
		starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(req->data_handle, -1, mode, _starpu_mpi_coop_send_acquired_callback, _starpu_mpi_coop_sends_data_ready, coop_sends, sequential_consistency, 0, &coop_sends->pre_sync_jobid, NULL, req->prio);
	else
		req->pre_sync_jobid = coop_sends->pre_sync_jobid;
}

void starpu_mpi_coop_sends_data_handle_nb_sends(starpu_data_handle_t data_handle, int nb_sends)
{
	struct _starpu_mpi_data *mpi_data = _starpu_mpi_data_get(data_handle);

	/* Has no effect is coops are disabled: this attribute is used only in
	 * _starpu_mpi_coop_send() that is called only if coops are enabled */
	mpi_data->nb_future_sends = nb_sends;
}

void starpu_mpi_coop_sends_set_use(int use_coop_sends)
{
	if (starpu_mpi_world_size() <= 2)
	{
		_STARPU_DISP("Not enough MPI processes to use coop_sends\n");
		return;
	}

	_starpu_mpi_use_coop_sends = use_coop_sends;
}

int starpu_mpi_coop_sends_get_use(void)
{
	return _starpu_mpi_use_coop_sends;
}
