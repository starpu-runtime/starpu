/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Federal University of Rio Grande do Sul (UFRGS)
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
#include <starpu_mpi_select_node.h>
#include <starpu_mpi_init.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <core/simgrid.h>
#include <core/task.h>
#include <core/topology.h>
#include <core/workers.h>

static void _starpu_mpi_isend_irecv_common(struct _starpu_mpi_req *req, enum starpu_data_access_mode mode, int sequential_consistency)
{
	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, _starpu_mpi_submit_ready_request(req) is called and
	 * the request is actually submitted */
	starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(req->data_handle, STARPU_MAIN_RAM, mode, _starpu_mpi_submit_ready_request, (void *)req, sequential_consistency, 1, &req->pre_sync_jobid, &req->post_sync_jobid);
}

static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg, int sequential_consistency)
{
	if (_starpu_mpi_fake_world_size != -1)
	{
		/* Don't actually do the communication */
		return NULL;
	}

#ifdef STARPU_MPI_PEDANTIC_ISEND
	enum starpu_data_access_mode mode = STARPU_RW;
#else
	enum starpu_data_access_mode mode = STARPU_R;
#endif

	struct _starpu_mpi_req *req = _starpu_mpi_request_fill(data_handle, dest, data_tag, comm, detached, sync, prio, callback, arg, SEND_REQ, _starpu_mpi_isend_size_func, sequential_consistency, 0, 0);
	_starpu_mpi_req_willpost(req);

	if (_starpu_mpi_use_coop_sends && detached == 1 && sync == 0 && callback == NULL)
	{
		/* It's a send & forget send, we can perhaps optimize its distribution over several nodes */
		_starpu_mpi_coop_send(data_handle, req, mode, sequential_consistency);
		return req;
	}

	/* Post normally */
	_starpu_mpi_isend_irecv_common(req, mode, sequential_consistency);
	return req;
}

int starpu_mpi_isend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm)
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

int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm)
{
	return starpu_mpi_isend_prio(data_handle, public_req, dest, data_tag, 0, comm);
}

int starpu_mpi_isend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 0, prio, callback, arg, 1);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_isend_detached_prio(data_handle, dest, data_tag, 0, comm, callback, arg);
}

int starpu_mpi_send_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm)
{
	starpu_mpi_req req;
	MPI_Status status;

	_STARPU_MPI_LOG_IN();
	starpu_mpi_isend_prio(data_handle, &req, dest, data_tag, prio, comm);

	memset(&status, 0, sizeof(MPI_Status));
	starpu_mpi_wait(&req, &status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm)
{
	return starpu_mpi_send_prio(data_handle, dest, data_tag, 0, comm);
}

int starpu_mpi_issend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm)
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

int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm)
{
	return starpu_mpi_issend_prio(data_handle, public_req, dest, data_tag, 0, comm);
}

int starpu_mpi_issend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 1, prio, callback, arg, 1);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_issend_detached_prio(data_handle, dest, data_tag, 0, comm, callback, arg);
}

struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count)
{
	if (_starpu_mpi_fake_world_size != -1)
	{
		/* Don't actually do the communication */
		return NULL;
	}

	struct _starpu_mpi_req *req = _starpu_mpi_request_fill(data_handle, source, data_tag, comm, detached, sync, 0, callback, arg, RECV_REQ, _starpu_mpi_irecv_size_func, sequential_consistency, is_internal_req, count);
	_starpu_mpi_req_willpost(req);
	_starpu_mpi_isend_irecv_common(req, STARPU_W, sequential_consistency);
	return req;
}

int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm)
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

int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, 1, 0, 0);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, sequential_consistency, 0, 0);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, MPI_Status *status)
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
	return _starpu_mpi_wait(public_req, status);
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	return _starpu_mpi_test(public_req, flag, status);
}

int starpu_mpi_barrier(MPI_Comm comm)
{
	return _starpu_mpi_barrier(comm);
}

void _starpu_mpi_data_clear(starpu_data_handle_t data_handle)
{
	_mpi_backend._starpu_mpi_backend_data_clear(data_handle);
	_starpu_mpi_cache_data_clear(data_handle);
	free(data_handle->mpi_data);
	data_handle->mpi_data = NULL;
}

struct _starpu_mpi_data *_starpu_mpi_data_get(starpu_data_handle_t data_handle)
{
	struct _starpu_mpi_data *mpi_data = data_handle->mpi_data;
	if (mpi_data)
	{
		STARPU_ASSERT(mpi_data->magic == 42);
	}
	else
	{
		_STARPU_CALLOC(mpi_data, 1, sizeof(struct _starpu_mpi_data));
		mpi_data->magic = 42;
		mpi_data->node_tag.data_tag = -1;
		mpi_data->node_tag.node.rank = -1;
		mpi_data->node_tag.node.comm = MPI_COMM_WORLD;
		_starpu_spin_init(&mpi_data->coop_lock);
		data_handle->mpi_data = mpi_data;
		_starpu_mpi_cache_data_init(data_handle);
		_starpu_data_set_unregister_hook(data_handle, _starpu_mpi_data_clear);
	}
	return mpi_data;
}

void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag, int rank, MPI_Comm comm)
{
	struct _starpu_mpi_data *mpi_data = _starpu_mpi_data_get(data_handle);

	if (data_tag != -1)
	{
		_mpi_backend._starpu_mpi_backend_data_register(data_handle, data_tag);
		mpi_data->node_tag.data_tag = data_tag;
		_STARPU_MPI_TRACE_DATA_SET_TAG(data_handle, data_tag);
	}
	if (rank != -1)
	{
		_STARPU_MPI_TRACE_DATA_SET_RANK(data_handle, rank);
		mpi_data->node_tag.node.rank = rank;
		mpi_data->node_tag.node.comm = comm;
	}
}

void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm)
{
	starpu_mpi_data_register_comm(handle, -1, rank, comm);
}

void starpu_mpi_data_set_tag(starpu_data_handle_t handle, starpu_mpi_tag_t data_tag)
{
	starpu_mpi_data_register_comm(handle, data_tag, -1, MPI_COMM_WORLD);
}

int starpu_mpi_data_get_rank(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.node.rank;
}

starpu_mpi_tag_t starpu_mpi_data_get_tag(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.data_tag;
}

void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank;
	starpu_mpi_tag_t tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register() or starpu_mpi_data_register_comm()\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return;

	tag = starpu_mpi_data_get_tag(data_handle);
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register() or starpu_mpi_data_register_comm()\n");
	}

	if (me == node)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = starpu_mpi_cached_receive_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_irecv_detached(data_handle, rank, tag, comm, callback, arg);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = starpu_mpi_cached_send_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_isend_detached(data_handle, node, tag, comm, NULL, NULL);
		}
	}
}

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank;
	starpu_mpi_tag_t tag;

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
		int already_received = starpu_mpi_cached_receive_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_recv(data_handle, rank, tag, comm, &status);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = starpu_mpi_cached_send_set(data_handle, node);
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
	starpu_task_wait_for_all();
	starpu_mpi_barrier(comm);
	return 0;
}
