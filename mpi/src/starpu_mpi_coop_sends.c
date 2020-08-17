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

#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <datawizard/coherency.h>

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
			starpu_data_release_on_node(req->data_handle, STARPU_MAIN_RAM);
		}
	}
	else
	{
		/* Trivial request */
		starpu_data_release_on_node(req->data_handle, STARPU_MAIN_RAM);
	}
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
	if (coop_sends->n == 1)
		/* Trivial case, don't optimize */
		return;

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

		/* And build the diffusion tree */
		_starpu_mpi_coop_sends_build_tree(coop_sends);
	}
	_starpu_spin_unlock(&coop_sends->lock);
}

/* This is called on completion of acquisition of data for a cooperative send */
static void _starpu_mpi_coop_sends_data_ready(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_coop_sends *coop_sends = arg;
	struct _starpu_mpi_data *mpi_data = coop_sends->mpi_data;

	/* Take the cooperative send bag out from more submissions */
	if (mpi_data->coop_sends == coop_sends)
	{
		_starpu_spin_lock(&mpi_data->coop_lock);
		if (mpi_data->coop_sends == coop_sends)
			mpi_data->coop_sends = NULL;
		_starpu_spin_unlock(&mpi_data->coop_lock);
	}

	/* Build diffusion tree */
	_starpu_mpi_coop_sends_optimize(coop_sends);

	if (coop_sends->n == 1)
	{
		/* Trivial case, just submit it */
		_starpu_mpi_submit_ready_request(_starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs));
	}
	else
	{
		/* And submit them */
		if (STARPU_TEST_AND_SET(&coop_sends->redirects_sent, 1) == 0)
			_starpu_mpi_submit_coop_sends(coop_sends, 1, 1);
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
	if (!coop_sends)
		return;

	/* Build diffusion tree */
	_starpu_mpi_coop_sends_optimize(coop_sends);

	if (coop_sends->n == 1)
		/* Trivial case, we will just send the data */
		return;

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
	struct _starpu_mpi_req *prevreq;

	prevreq = _starpu_mpi_req_multilist_begin_coop_sends(&coop_sends->reqs);
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
					_starpu_mpi_req_multilist_erase_coop_sends(&coop_sends->reqs, req);
					tofree = coop_sends;
				}
				coop_sends = mpi_data->coop_sends;
				_STARPU_MPI_DEBUG(0, "%p: add to cooperative sends %p, dest %d\n", data_handle, coop_sends, req->node_tag.node.rank);
				_starpu_mpi_req_multilist_push_back_coop_sends(&coop_sends->reqs, req);
				coop_sends->n++;
				req->coop_sends_head = coop_sends;
				first = 0;
				done = 1;
			}
			else
			{
				/* Nope, incompatible, put ours instead */
				_STARPU_MPI_DEBUG(0, "%p: new cooperative sends %p, dest %d\n", data_handle, coop_sends, req->node_tag.node.rank);
				mpi_data->coop_sends = coop_sends;
				first = 1;
				_starpu_spin_unlock(&mpi_data->coop_lock);
				/* and flush it */
				_starpu_mpi_coop_send_flush(coop_sends);
				break;
			}
		}
		else if (coop_sends)
		{
			/* Nobody else and we have allocated one, we're first! */
			_STARPU_MPI_DEBUG(0, "%p: new cooperative sends %p, dest %d\n", data_handle, coop_sends, req->node_tag.node.rank);
			mpi_data->coop_sends = coop_sends;
			first = 1;
			done = 1;
		}
		_starpu_spin_unlock(&mpi_data->coop_lock);

		if (!done && !coop_sends)
		{
			/* Didn't find something to join, create one out of critical section */
			_STARPU_MPI_CALLOC(coop_sends, 1, sizeof(*coop_sends));
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

	/* In case we created one for nothing after all */
	free(tofree);

	if (first)
	{
		/* We were first, we are responsible for acquiring the data for everybody */
		starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(req->data_handle, STARPU_MAIN_RAM, mode, _starpu_mpi_coop_sends_data_ready, coop_sends, sequential_consistency, 0, &req->pre_sync_jobid, NULL);
	}
}

