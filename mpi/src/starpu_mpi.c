/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019-2021  Federal University of Rio Grande do Sul (UFRGS)
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
#include <starpu_mpi_task_insert.h>
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

int _starpu_mpi_choose_node(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	if (mode & STARPU_W)
	{
		/* Receiving */

		/* TODO: lookup NIC location */
		/* Where to receive the data? */
		if (handle->home_node >= 0 && starpu_node_get_kind(handle->home_node) == STARPU_CPU_RAM)
			/* For now, better use the home node to avoid duplicates */
			return handle->home_node;

		/* Several potential places */
		unsigned i;
		if (_starpu_mpi_has_cuda || _starpu_mpi_has_hip)
			for (i = 0; i < STARPU_MAXNODES; i++)
			{
				/* Note: We take as a hint that it's allocated on the GPU as
				 * a clue that we want to push directly to the GPU */
				if (_starpu_mpi_has_cuda
				        && starpu_node_get_kind(i) == STARPU_CUDA_RAM
					&& handle->per_node[i].allocated
					&& (_starpu_mpi_cuda_devid == -1 || _starpu_mpi_cuda_devid == starpu_memory_node_get_devid(i)))
					/* This node already has allocated buffers, let's just use it */
					return i;
				if (_starpu_mpi_has_hip
				        && starpu_node_get_kind(i) == STARPU_HIP_RAM
					&& handle->per_node[i].allocated
					&& (_starpu_mpi_hip_devid == -1 || _starpu_mpi_hip_devid == starpu_memory_node_get_devid(i)))
					/* This node already has allocated buffers, let's just use it */
					return i;
			}

		for (i = 0; i < STARPU_MAXNODES; i++)
		{
			/* Note: We take as a hint that it's allocated on a NUMA node as
			 * a clue that we want to push directly to that NUMA node */
			if (starpu_node_get_kind(i) == STARPU_CPU_RAM
				&& handle->per_node[i].allocated)
				/* This node already has allocated buffers, let's just use it */
				return i;
		}

		/* No luck, take the least loaded node */
		starpu_ssize_t maximum = 0;
		starpu_ssize_t needed = starpu_data_get_alloc_size(handle);
		unsigned node = STARPU_MAIN_RAM;

		for (i = 0; i < STARPU_MAXNODES; i++)
		{
                       if (starpu_node_get_kind(i) == STARPU_CPU_RAM
                               || (_starpu_mpi_has_cuda && starpu_node_get_kind(i) == STARPU_CUDA_RAM)
                               || (_starpu_mpi_has_hip  && starpu_node_get_kind(i) == STARPU_HIP_RAM))
			{
				starpu_ssize_t size = starpu_memory_get_available(i);
				if (size >= needed && size > maximum)
				{
					node = i;
					maximum = size;
				}
			}
		}
		return node;
	}
	else
	{
		/* Sending */

		/* Several potential places */
		unsigned i;
		for (i = 0; i < STARPU_MAXNODES; i++)
		{
			if (handle->per_node[i].state != STARPU_INVALID)
			{
				/* If this node already has the value, let's just use it */
				/* TODO: rather pick up place next to NIC */
				if (starpu_node_get_kind(i) == STARPU_CPU_RAM)
					return i;
				if (_starpu_mpi_has_cuda
				    && starpu_node_get_kind(i) == STARPU_CUDA_RAM
				    && (_starpu_mpi_cuda_devid == -1 || _starpu_mpi_cuda_devid == starpu_memory_node_get_devid(i)))
					return i;
				if (_starpu_mpi_has_hip
				    && starpu_node_get_kind(i) == STARPU_HIP_RAM
				    && (_starpu_mpi_hip_devid == -1 || _starpu_mpi_hip_devid == starpu_memory_node_get_devid(i)))
					return i;
			}
		}

		/* No luck, take the least loaded node, to transfer from e.g. GPU */
		starpu_ssize_t maximum = 0;
		starpu_ssize_t needed = starpu_data_get_alloc_size(handle);
		unsigned node = STARPU_MAIN_RAM;

		for (i = 0; i < STARPU_MAXNODES; i++)
		{
			if (starpu_node_get_kind(i) == STARPU_CPU_RAM)
			{
				starpu_ssize_t size = starpu_memory_get_available(i);
				if (size >= needed && size > maximum)
				{
					node = i;
					maximum = size;
				}
			}
		}
		return node;
	}
}

static void _starpu_mpi_acquired_callback(void *arg, int *nodep, enum starpu_data_access_mode mode)
{
	struct _starpu_mpi_req *req = arg;
	int node = *nodep;

	/* The data was acquired in terms of dependencies, we can now look the
	 * current state of the handle and decide which node we prefer for the data
	 * fetch */

	if (node < 0)
		node = _starpu_mpi_choose_node(req->data_handle, mode);

	req->node = *nodep = node;
}

void _starpu_mpi_isend_irecv_common(struct _starpu_mpi_req *req, enum starpu_data_access_mode mode, int sequential_consistency)
{

	int node = -1;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, _starpu_mpi_submit_ready_request(req) is called and
	 * the request is actually submitted */

	if (_starpu_mpi_mem_throttle && mode & STARPU_W && !req->data_handle->initialized)
	{
		/* We will trigger allocation, pre-reserve for it */
		size_t size = starpu_data_get_size(req->data_handle);
		if (size)
		{
			/* FIXME: rather take the less-loaded NUMA node */
			node = STARPU_MAIN_RAM;

			/* This will potentially block */
			starpu_memory_allocate(node, size, STARPU_MEMORY_WAIT);
			req->reserved_size = size;
			/* This also decides where we will store the data */
			req->node = node;
		}
	}

	/* TODO: use soon_callback */
	if (sequential_consistency)
	{
		starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(req->data_handle, node, mode, NULL, _starpu_mpi_acquired_callback, _starpu_mpi_submit_ready_request, (void *)req, 1 /*sequential consistency*/, 1, &req->pre_sync_jobid, &req->post_sync_jobid, req->prio);
	}
	else
	{
		/* post_sync_job_id has already been filled */
		starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(req->data_handle, node, mode, NULL, _starpu_mpi_acquired_callback, _starpu_mpi_submit_ready_request, (void *)req, 0 /*sequential consistency*/, 1, &req->pre_sync_jobid, NULL, req->prio);
	}
}

struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg, int sequential_consistency)
{
	if (STARPU_UNLIKELY(_starpu_mpi_fake_world_size != -1))
	{
		/* Don't actually do the communication */
		return NULL;
	}

#ifdef STARPU_MPI_PEDANTIC_ISEND
	enum starpu_data_access_mode mode = STARPU_RW;
#else
	enum starpu_data_access_mode mode = STARPU_R;
#endif

	struct _starpu_mpi_req *req = _starpu_mpi_request_fill(data_handle, dest, data_tag, comm, detached, sync, prio, callback, arg, SEND_REQ, _mpi_backend._starpu_mpi_backend_isend_size_func, sequential_consistency, 0, 0, 0);
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
	int ret;

	_STARPU_MPI_LOG_IN();
	ret  = starpu_mpi_isend_prio(data_handle, &req, dest, data_tag, prio, comm);
	if (ret)
		return ret;

	memset(&status, 0, sizeof(MPI_Status));
	ret = starpu_mpi_wait(&req, &status);

	_STARPU_MPI_LOG_OUT();
	return ret;
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

struct _starpu_mpi_req* _starpu_mpi_isend_cache_aware(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *_arg, int sequential_consistency, int* cache_flag)
{
	struct _starpu_mpi_req* req = NULL;
	int already_sent = starpu_mpi_cached_send_set(data_handle, dest);
	if (already_sent == 0)
	{
		*cache_flag = 0;
		if (data_tag == -1)
			_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
		_STARPU_MPI_DEBUG(1, "Send data %p to %d\n", data_handle, dest);
		req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, detached, sync, prio, callback, _arg, sequential_consistency);
	}
	else
	{
		_STARPU_MPI_DEBUG(1, "STARPU CACHE: Data already sent\n");
		*cache_flag = 1;
		if (callback)
			callback(_arg);
	}
	return req;
}

struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_mpi_comm internal_comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count, int prio)
{
	if (_starpu_mpi_fake_world_size != -1)
	{
		/* Don't actually do the communication */
		return NULL;
	}

	struct _starpu_mpi_req *req = _starpu_mpi_request_fill(data_handle, source, data_tag, comm, detached, sync, prio, callback, arg, RECV_REQ, _mpi_backend._starpu_mpi_backend_irecv_size_func, sequential_consistency, is_internal_req,  internal_comm, count);
	_starpu_mpi_req_willpost(req);

	if (sequential_consistency == 0)
	{
		/* Synchronization task jobid from redux is used */
		_starpu_mpi_redux_fill_post_sync_jobid(arg, &(req->post_sync_jobid));
	}

	_starpu_mpi_isend_irecv_common(req, STARPU_W, sequential_consistency);
	return req;
}

int _starpu_mpi_irecv_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_irecv needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(source, data_tag);
	req = _starpu_mpi_irecv_common(data_handle, source, data_tag, comm, MPI_COMM_NULL, 0, 0, NULL, NULL, 1, 0, 0, prio);
	_STARPU_MPI_TRACE_IRECV_COMPLETE_END(source, data_tag);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_irecv_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm)
{
	return _starpu_mpi_irecv_prio(data_handle, public_req, source, data_tag, STARPU_DEFAULT_PRIO, comm);
}

int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, MPI_COMM_NULL, 1, 0, callback, arg, 1, 0, 0, STARPU_DEFAULT_PRIO);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached_prio(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, MPI_COMM_NULL, 1, 0, callback, arg, 1, 0, 0, prio);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, MPI_COMM_NULL, 1, 0, callback, arg, sequential_consistency, 0, 0, STARPU_DEFAULT_PRIO);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int _starpu_mpi_recv_prio(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, MPI_Status *status)
{
	STARPU_ASSERT_MSG(status != NULL || status == MPI_STATUS_IGNORE, "MPI_Status value cannot be NULL or different from MPI_STATUS_IGNORE");

	starpu_mpi_req req;
	int ret;

	_STARPU_MPI_LOG_IN();

	ret = _starpu_mpi_irecv_prio(data_handle, &req, source, data_tag, prio, comm);
	if (ret)
		return ret;
	ret = starpu_mpi_wait(&req, status);

	_STARPU_MPI_LOG_OUT();
	return ret;
}

int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, MPI_Status *status)
{
	return _starpu_mpi_recv_prio(data_handle, source, data_tag, STARPU_DEFAULT_PRIO, comm, status);
}

int starpu_mpi_recv_prio(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, MPI_Status *status)
{
	return _starpu_mpi_recv_prio(data_handle, source, data_tag, prio, comm, status);
}

struct _starpu_mpi_req* _starpu_mpi_irecv_cache_aware(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *_arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count, int* cache_flag)
{
	struct _starpu_mpi_req* req = NULL;
	int already_received = starpu_mpi_cached_cp_receive_set(data_handle);
	if (already_received == 0)
	{
		if (data_tag == -1)
			_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
		_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, source);
		req = _starpu_mpi_irecv_common(data_handle, source, data_tag, comm, MPI_COMM_NULL, detached, sync, callback, _arg, sequential_consistency, is_internal_req, count, STARPU_DEFAULT_PRIO); //TODO: Allow to pass prio in args
		*cache_flag = 0;
	}
	else
	{
		_STARPU_MPI_DEBUG(1, "STARPU CACHE: Data already received\n");
		*cache_flag =1;
		if (callback)
			callback(_arg);
	}
	return req;
}

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	STARPU_ASSERT_MSG(status != NULL || status == MPI_STATUS_IGNORE, "MPI_Status value cannot be NULL or different from MPI_STATUS_IGNORE");
	return _mpi_backend._starpu_mpi_backend_wait(public_req, status);
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	STARPU_ASSERT_MSG(status != NULL || status == MPI_STATUS_IGNORE, "MPI_Status value cannot be NULL or different from MPI_STATUS_IGNORE");
	return _mpi_backend._starpu_mpi_backend_test(public_req, flag, status);
}

int starpu_mpi_barrier(MPI_Comm comm)
{
	return _mpi_backend._starpu_mpi_backend_barrier(comm);
}

void _starpu_mpi_data_clear(starpu_data_handle_t data_handle)
{
	struct _starpu_mpi_data *data = data_handle->mpi_data;
	_mpi_backend._starpu_mpi_backend_data_clear(data_handle);
	_starpu_mpi_cache_data_clear(data_handle);
	_starpu_spin_destroy(&data->coop_lock);
	free(data->redux_map);
	data->redux_map = NULL;
	free(data);
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
		mpi_data->nb_future_sends = 0;
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

char* starpu_mpi_data_get_redux_map(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->redux_map;
}

int starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank;
	starpu_mpi_tag_t data_tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register() or starpu_mpi_data_register_comm()\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return 0;

	data_tag = starpu_mpi_data_get_tag(data_handle);
	if (data_tag == -1)
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
			return starpu_mpi_irecv_detached(data_handle, rank, data_tag, comm, callback, arg);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = starpu_mpi_cached_send_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			return starpu_mpi_isend_detached(data_handle, node, data_tag, comm, NULL, NULL);
		}
	}
	return 0;
}

int starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank;
	starpu_mpi_tag_t data_tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return 0;

	data_tag = starpu_mpi_data_get_tag(data_handle);
	if (data_tag == -1)
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
			return starpu_mpi_recv(data_handle, rank, data_tag, comm, &status);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = starpu_mpi_cached_send_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			return starpu_mpi_send(data_handle, node, data_tag, comm);
		}
	}
	return 0;
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
	/* TODO: or at least remember that the previous owner has the data, that's an easy case to support */
	starpu_mpi_cache_flush(comm, data);
	return;
}

int starpu_mpi_wait_for_all(MPI_Comm comm)
{
	/* If the user forgets to call mpi_redux_data or insert R tasks on the reduced handles */
	/* then, we wrap reduction patterns for them. This is typical of benchmarks */
	_starpu_mpi_redux_wrapup_data_all();
	return _mpi_backend._starpu_mpi_backend_wait_for_all(comm);
}

int starpu_mpi_wait_for_all_in_ctx(MPI_Comm comm, unsigned sched_ctx)
{
	/* If the user forgets to call mpi_redux_data or insert R tasks on the reduced handles */
	/* then, we wrap reduction patterns for them. This is typical of benchmarks */
	_starpu_mpi_redux_wrapup_data_all();
	return _mpi_backend._starpu_mpi_backend_wait_for_all_in_ctx(comm, sched_ctx);
}

void starpu_mpi_comm_stats_disable()
{
	_starpu_mpi_comm_stats_disable();
}

void starpu_mpi_comm_stats_enable()
{
	_starpu_mpi_comm_stats_enable();
}

int _starpu_mpi_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle, MPI_Comm comm, int asynchronous, void (*callback_func)(void *), void *callback_arg, int priority)
{
	if (dst_handle == src_handle)
	{
		if (callback_func)
			callback_func(callback_arg);
		return 0;
	}

	int ret = 0;
	int src_rank = starpu_mpi_data_get_rank(src_handle);
	int dst_rank = starpu_mpi_data_get_rank(dst_handle);

	if (src_rank == dst_rank)
		// Both data are on the same node, no need to transfer data
		ret = starpu_data_cpy_priority(dst_handle, src_handle, asynchronous, callback_func, callback_arg, priority);
	else
	{
		// We need to transfer data
		int my_rank;
		starpu_mpi_comm_rank(comm, &my_rank);
		starpu_mpi_tag_t tag = starpu_mpi_data_get_tag(dst_handle);

		if (my_rank == src_rank)
		{
			if (asynchronous == 1)
				ret = starpu_mpi_isend_detached_prio(src_handle, dst_rank, tag, priority, comm, NULL, NULL);
			else
				ret = starpu_mpi_send_prio(src_handle, dst_rank, tag, priority, comm);
		}
		else if (my_rank == dst_rank)
		{
			if (asynchronous == 1)
				ret = starpu_mpi_irecv_detached_prio(dst_handle, src_rank, tag, priority, comm, callback_func, callback_arg);
			else
			{
				ret = starpu_mpi_recv_prio(dst_handle, src_rank, tag, priority, comm, MPI_STATUS_IGNORE);
				if (callback_func)
					callback_func(callback_arg);
			}
			return ret;
		}
	}
	starpu_mpi_cache_flush(comm, dst_handle);
	return ret;
}

int starpu_mpi_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle, MPI_Comm comm, int asynchronous, void (*callback_func)(void *), void *callback_arg)
{
	return _starpu_mpi_data_cpy(dst_handle, src_handle, comm, asynchronous, callback_func, callback_arg, STARPU_DEFAULT_PRIO);
}

int starpu_mpi_data_cpy_priority(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle, MPI_Comm comm, int asynchronous, void (*callback_func)(void *), void *callback_arg, int priority)
{
	return _starpu_mpi_data_cpy(dst_handle, src_handle, comm, asynchronous, callback_func, callback_arg, priority);
}
