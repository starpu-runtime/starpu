/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2017  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017  CNRS
 * Copyright (C) 2016  Inria
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

#include <stdlib.h>
#include <limits.h>
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <starpu_profiling.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_cache.h>
#include <mpi/starpu_mpi_sync_data.h>
#include <mpi/starpu_mpi_early_data.h>
#include <mpi/starpu_mpi_early_request.h>
#include <starpu_mpi_select_node.h>
#include <mpi/starpu_mpi_tag.h>
#include <mpi/starpu_mpi_comm.h>
#include <starpu_mpi_init.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <core/simgrid.h>
#include <core/task.h>
#include <core/topology.h>
#include <core/workers.h>

static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int data_tag, MPI_Comm comm,
							unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg,
							int sequential_consistency)
{
	return _starpu_mpi_isend_irecv_common(data_handle, dest, data_tag, comm, detached, sync, prio, callback, arg, SEND_REQ, _starpu_mpi_isend_size_func,
#ifdef STARPU_MPI_PEDANTIC_ISEND
					      STARPU_RW,
#else
					      STARPU_R,
#endif
					      sequential_consistency, 0, 0);
}

int starpu_mpi_isend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, int prio, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_isend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(dest, data_tag, 0);
	req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 0, 0, prio, NULL, NULL, 1);
	_STARPU_MPI_TRACE_ISEND_COMPLETE_END(dest, data_tag, 0);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, MPI_Comm comm)
{
	return starpu_mpi_isend_prio(data_handle, public_req, dest, data_tag, 0, comm);
}

int starpu_mpi_isend_detached_prio(starpu_data_handle_t data_handle, int dest, int data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 0, prio, callback, arg, 1);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_isend_detached_prio(data_handle, dest, data_tag, 0, comm, callback, arg);
}

int starpu_mpi_send_prio(starpu_data_handle_t data_handle, int dest, int data_tag, int prio, MPI_Comm comm)
{
	starpu_mpi_req req;
	MPI_Status status;

	_STARPU_MPI_LOG_IN();
	memset(&status, 0, sizeof(MPI_Status));

	starpu_mpi_isend_prio(data_handle, &req, dest, data_tag, prio, comm);
	starpu_mpi_wait(&req, &status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm)
{
	return starpu_mpi_send_prio(data_handle, dest, data_tag, 0, comm);
}

int starpu_mpi_issend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, int prio, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_issend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 0, 1, prio, NULL, NULL, 1);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, MPI_Comm comm)
{
	return starpu_mpi_issend_prio(data_handle, public_req, dest, data_tag, 0, comm);
}

int starpu_mpi_issend_detached_prio(starpu_data_handle_t data_handle, int dest, int data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 1, prio, callback, arg, 1);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_issend_detached_prio(data_handle, dest, data_tag, 0, comm, callback, arg);
}

struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count)
{
	return _starpu_mpi_isend_irecv_common(data_handle, source, data_tag, comm, detached, sync, 0, callback, arg, RECV_REQ, _starpu_mpi_irecv_data_func, STARPU_W, sequential_consistency, is_internal_req, count);
}

int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, int data_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_irecv needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(source, data_tag);
	req = _starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 0, 0, NULL, NULL, 1, 0, 0);
	_STARPU_MPI_TRACE_IRECV_COMPLETE_END(source, data_tag);
	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_irecv_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, 1, 0, 0);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, sequential_consistency, 0, 0);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, MPI_Status *status)
{
	starpu_mpi_req req;
	_STARPU_MPI_LOG_IN();

	starpu_mpi_irecv(data_handle, &req, source, data_tag, comm);
	starpu_mpi_wait(&req, status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	int ret;
	struct _starpu_mpi_req *req = *public_req;
	struct _starpu_mpi_req *waiting_req;

	_STARPU_MPI_LOG_IN();

	/* We cannot try to complete a MPI request that was not actually posted
	 * to MPI yet. */
	STARPU_PTHREAD_MUTEX_LOCK(&(req->req_mutex));
	while (!(req->submitted))
		STARPU_PTHREAD_COND_WAIT(&(req->req_cond), &(req->req_mutex));
	STARPU_PTHREAD_MUTEX_UNLOCK(&(req->req_mutex));

	/* Initialize the request structure */
	 _starpu_mpi_request_init(&waiting_req);
	waiting_req->prio = INT_MAX;
	waiting_req->status = status;
	waiting_req->other_request = req;
	waiting_req->func = _starpu_mpi_wait_func;
	waiting_req->request_type = WAIT_REQ;

	_starpu_mpi_submit_ready_request_inc(waiting_req);

	/* We wait for the MPI request to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	while (!req->completed)
		STARPU_PTHREAD_COND_WAIT(&req->req_cond, &req->req_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	ret = req->ret;

	/* The internal request structure was automatically allocated */
	*public_req = NULL;
	if (req->internal_req)
	{
		_starpu_mpi_request_destroy(req->internal_req);
	}
	_starpu_mpi_request_destroy(req);
	_starpu_mpi_request_destroy(waiting_req);

	_STARPU_MPI_LOG_OUT();
	return ret;
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	int ret = 0;

	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_test needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req = *public_req;

	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Test cannot be called on a detached request");

	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	unsigned submitted = req->submitted;
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	if (submitted)
	{
		struct _starpu_mpi_req *testing_req;

		/* Initialize the request structure */
		_starpu_mpi_request_init(&testing_req);
		testing_req->prio = INT_MAX;
		testing_req->flag = flag;
		testing_req->status = status;
		testing_req->other_request = req;
		testing_req->func = _starpu_mpi_test_func;
		testing_req->completed = 0;
		testing_req->request_type = TEST_REQ;

		_starpu_mpi_submit_ready_request_inc(testing_req);

		/* We wait for the test request to finish */
		STARPU_PTHREAD_MUTEX_LOCK(&(testing_req->req_mutex));
		while (!(testing_req->completed))
			STARPU_PTHREAD_COND_WAIT(&(testing_req->req_cond), &(testing_req->req_mutex));
		STARPU_PTHREAD_MUTEX_UNLOCK(&(testing_req->req_mutex));

		ret = testing_req->ret;

		if (*(testing_req->flag))
		{
			/* The request was completed so we free the internal
			 * request structure which was automatically allocated
			 * */
			*public_req = NULL;
			if (req->internal_req)
			{
				_starpu_mpi_request_destroy(req->internal_req);
			}
			_starpu_mpi_request_destroy(req);
		}

		_starpu_mpi_request_destroy(testing_req);
	}
	else
	{
		*flag = 0;
	}

	_STARPU_MPI_LOG_OUT();
	return ret;
}

int starpu_mpi_barrier(MPI_Comm comm)
{
	_starpu_mpi_barrier(comm);
	return 0;
}

void _starpu_mpi_data_clear(starpu_data_handle_t data_handle)
{
	_starpu_mpi_tag_data_release(data_handle);
	_starpu_mpi_cache_data_clear(data_handle);
	free(data_handle->mpi_data);
}

void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, int tag, int rank, MPI_Comm comm)
{
	struct _starpu_mpi_data *mpi_data;
	if (data_handle->mpi_data)
	{
		mpi_data = data_handle->mpi_data;
	}
	else
	{
		_STARPU_CALLOC(mpi_data, 1, sizeof(struct _starpu_mpi_data));
		mpi_data->magic = 42;
		mpi_data->node_tag.data_tag = -1;
		mpi_data->node_tag.rank = -1;
		mpi_data->node_tag.comm = MPI_COMM_WORLD;
		data_handle->mpi_data = mpi_data;
		_starpu_mpi_cache_data_init(data_handle);
		_starpu_mpi_tag_data_register(data_handle, tag);
		_starpu_data_set_unregister_hook(data_handle, _starpu_mpi_data_clear);
	}

	if (tag != -1)
	{
		mpi_data->node_tag.data_tag = tag;
	}
	if (rank != -1)
	{
		_STARPU_MPI_TRACE_DATA_SET_RANK(data_handle, rank);
		mpi_data->node_tag.rank = rank;
		mpi_data->node_tag.comm = comm;
		_starpu_mpi_comm_register(comm);
	}
}

void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm)
{
	starpu_mpi_data_register_comm(handle, -1, rank, comm);
}

void starpu_mpi_data_set_tag(starpu_data_handle_t handle, int tag)
{
	starpu_mpi_data_register_comm(handle, tag, -1, MPI_COMM_WORLD);
}

int starpu_mpi_data_get_rank(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.rank;
}

int starpu_mpi_data_get_tag(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.data_tag;
}

void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register() or starpu_mpi_data_register()\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return;

	tag = starpu_mpi_data_get_tag(data_handle);
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register() or starpu_mpi_data_register()\n");
	}

	if (me == node)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = _starpu_mpi_cache_received_data_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_irecv_detached(data_handle, rank, tag, comm, callback, arg);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = _starpu_mpi_cache_sent_data_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_isend_detached(data_handle, node, tag, comm, NULL, NULL);
		}
	}
}

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return;

	tag = starpu_mpi_data_get_tag(data_handle);
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
	}

	if (me == node)
	{
		MPI_Status status;
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = _starpu_mpi_cache_received_data_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_recv(data_handle, rank, tag, comm, &status);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = _starpu_mpi_cache_sent_data_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_send(data_handle, node, tag, comm);
		}
	}
}

void starpu_mpi_get_data_on_all_nodes_detached(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int size, i;
	starpu_mpi_comm_size(comm, &size);
#ifdef STARPU_DEVEL
#warning TODO: use binary communication tree to optimize broadcast
#endif
	for (i = 0; i < size; i++)
		starpu_mpi_get_data_on_node_detached(comm, data_handle, i, NULL, NULL);
}

void starpu_mpi_data_migrate(MPI_Comm comm, starpu_data_handle_t data, int new_rank)
{
	int old_rank = starpu_mpi_data_get_rank(data);
	if (new_rank == old_rank)
		/* Already there */
		return;

	/* First submit data migration if it's not already on destination */
	starpu_mpi_get_data_on_node_detached(comm, data, new_rank, NULL, NULL);

	/* And note new owner */
	starpu_mpi_data_set_rank_comm(data, new_rank, comm);

	/* Flush cache in all other nodes */
	/* TODO: Ideally we'd transmit the knowledge of who owns it */
	starpu_mpi_cache_flush(comm, data);
	return;
}

int starpu_mpi_wait_for_all(MPI_Comm comm)
{
	int mpi = 1;
	int task = 1;
	while (task || mpi)
	{
		task = _starpu_task_wait_for_all_and_return_nb_waited_tasks();
		mpi = _starpu_mpi_barrier(comm);
	}
	return 0;
}
