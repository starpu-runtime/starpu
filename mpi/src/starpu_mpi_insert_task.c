/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdarg.h>
#include <mpi.h>

#include <starpu.h>
#include <starpu_data.h>
#include <common/utils.h>
#include <util/starpu_insert_task_utils.h>
#include <datawizard/coherency.h>
#include <core/task.h>

#include <starpu_mpi_cache.h>
#include <starpu_mpi_cache_stats.h>
#include <starpu_mpi_private.h>

typedef void (*_starpu_callback_func_t)(void *);

#define _SEND_DATA(data, mode, dest, mpi_tag, comm, callback, arg)	\
	if (mode & STARPU_SSEND) \
		starpu_mpi_issend_detached(data, dest, mpi_tag, comm, callback, arg); \
	else \
		starpu_mpi_isend_detached(data, dest, mpi_tag, comm, callback, arg);

static
int _starpu_mpi_find_executee_node(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int *do_execute, int *inconsistent_execute, int *dest, size_t *size_on_nodes)
{
	if (data && mode & STARPU_R)
	{
		struct starpu_data_interface_ops *ops;
		int rank = starpu_mpi_data_get_rank(data);

		ops = data->ops;
		size_on_nodes[rank] += ops->get_size(data);
	}

	if (mode & STARPU_W)
	{
		if (!data)
		{
			/* We don't have anything allocated for this.
			 * The application knows we won't do anything
			 * about this task */
			/* Yes, the app could actually not call
			 * insert_task at all itself, this is just a
			 * safeguard. */
			_STARPU_MPI_DEBUG(3, "oh oh\n");
			_STARPU_MPI_LOG_OUT();
			return -EINVAL;
		}
		int mpi_rank = starpu_mpi_data_get_rank(data);
		if (mpi_rank == me)
		{
			if (*do_execute == 0)
			{
				*inconsistent_execute = 1;
			}
			else
			{
				*do_execute = 1;
			}
		}
		else if (mpi_rank != -1)
		{
			if (*do_execute == 1)
			{
				*inconsistent_execute = 1;
			}
			else
			{
				*do_execute = 0;
				*dest = mpi_rank;
				/* That's the rank which needs the data to be sent to */
			}
		}
		else
		{
			_STARPU_ERROR("rank %d invalid\n", mpi_rank);
		}
	}
	return 0;
}

static
void _starpu_mpi_exchange_data_before_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int dest, int do_execute, MPI_Comm comm)
{
	if (data && mode & STARPU_R)
	{
		int mpi_rank = starpu_mpi_data_get_rank(data);
		int mpi_tag = starpu_mpi_data_get_tag(data);
		if (mpi_rank == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
			STARPU_ABORT();
		}
		if (mpi_tag == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
			STARPU_ABORT();
		}

		/* The task needs to read this data */
		if (do_execute && mpi_rank != me)
 		{
			/* The node is going to execute the codelet, but it does not own the data, it needs to receive the data from the owner node */
			void *already_received = _starpu_mpi_cache_received_data_set(data, mpi_rank);
			if (already_received == NULL)
			{
				_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data, mpi_rank);
				starpu_mpi_irecv_detached(data, mpi_rank, mpi_tag, comm, NULL, NULL);
			}
			// else the node has already received the data

		}

		if (!do_execute && mpi_rank == me)
		{
			/* The node owns the data, but another node is going to execute the codelet, the node needs to send the data to the executee node. */
			void *already_sent = _starpu_mpi_cache_sent_data_set(data, dest);
			if (already_sent == NULL)
			{
				_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data, dest);
				_SEND_DATA(data, mode, dest, mpi_tag, comm, NULL, NULL);
			}
			// Else the data has already been sent

		}
	}
}

static
void _starpu_mpi_exchange_data_after_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int xrank, int dest, int do_execute, MPI_Comm comm)
{
	if (mode & STARPU_W)
	{
		int mpi_rank = starpu_mpi_data_get_rank(data);
		int mpi_tag = starpu_mpi_data_get_tag(data);
		if(mpi_rank == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
			STARPU_ABORT();
		}
		if(mpi_tag == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
			STARPU_ABORT();
		}
		if (mpi_rank == me)
		{
			if (xrank != -1 && me != xrank)
			{
				_STARPU_MPI_DEBUG(1, "Receive data %p back from the task %d which executed the codelet ...\n", data, dest);
				starpu_mpi_irecv_detached(data, dest, mpi_tag, comm, NULL, NULL);
			}
		}
		else if (do_execute)
		{
			_STARPU_MPI_DEBUG(1, "Send data %p back to its owner %d...\n", data, mpi_rank);
			_SEND_DATA(data, mode, mpi_rank, mpi_tag, comm, NULL, NULL);
		}
	}
}

void _starpu_mpi_clear_data_after_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int do_execute, MPI_Comm comm)
{
	if (_starpu_cache_enabled)
	{
		if (mode & STARPU_W || mode & STARPU_REDUX)
		{
			/* The data has been modified, it MUST be removed from the cache */
			_starpu_mpi_cache_sent_data_clear(comm, data);
			_starpu_mpi_cache_received_data_clear(data);
		}
	}
	else
	{
		/* We allocated a temporary buffer for the received data, now drop it */
		if ((mode & STARPU_R) && do_execute)
		{
			int mpi_rank = starpu_mpi_data_get_rank(data);
			if (mpi_rank != me && mpi_rank != -1)
			{
				starpu_data_invalidate_submit(data);
			}
		}
	}
}

int starpu_mpi_insert_task(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	int arg_type;
	va_list varg_list;
	int me, do_execute, xrank, nb_nodes;
	size_t *size_on_nodes;
	size_t arg_buffer_size = 0;
	void *arg_buffer = NULL;
	int dest=0, inconsistent_execute;
	int current_data = 0;

	_STARPU_MPI_LOG_IN();

	MPI_Comm_rank(comm, &me);
	MPI_Comm_size(comm, &nb_nodes);

	size_on_nodes = (size_t *)calloc(1, nb_nodes * sizeof(size_t));

	/* Find out whether we are to execute the data because we own the data to be written to. */
	inconsistent_execute = 0;
	do_execute = -1;
	xrank = -1;
	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			xrank = va_arg(varg_list, int);
			_STARPU_MPI_DEBUG(1, "Executing on node %d\n", xrank);
			do_execute = 1;
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			starpu_data_handle_t data = va_arg(varg_list, starpu_data_handle_t);
			xrank = starpu_mpi_data_get_rank(data);
			STARPU_ASSERT_MSG(xrank != -1, "Rank of the data must be set using starpu_mpi_data_register() or starpu_data_set_rank()");
			_STARPU_MPI_DEBUG(1, "Executing on data node %d\n", xrank);
			STARPU_ASSERT_MSG(xrank <= nb_nodes, "Node %d to execute codelet is not a valid node (%d)", xrank, nb_nodes);
			do_execute = 1;
		}
		else if (arg_type & STARPU_R || arg_type & STARPU_W || arg_type & STARPU_RW || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;
			int ret = _starpu_mpi_find_executee_node(data, mode, me, &do_execute, &inconsistent_execute, &dest, size_on_nodes);
			if (ret == -EINVAL)
			{
				free(size_on_nodes);
				return ret;
			}
			current_data ++;
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list, int);
			int i;
			for(i=0 ; i<nb_handles ; i++)
			{
				enum starpu_data_access_mode mode = STARPU_CODELET_GET_MODE(codelet, current_data);
				int ret = _starpu_mpi_find_executee_node(datas[i], mode, me, &do_execute, &inconsistent_execute, &dest, size_on_nodes);
				if (ret == -EINVAL)
				{
					free(size_on_nodes);
					return ret;
				}
				current_data ++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			va_arg(varg_list, void *);
			va_arg(varg_list, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list, _starpu_callback_func_t);
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list, int);
		}
		/* STARPU_EXECUTE_ON_NODE handled above */
		/* STARPU_EXECUTE_ON_DATA handled above */
		/* STARPU_DATA_ARRAY handled above */
		else if (arg_type==STARPU_TAG)
		{
			(void)va_arg(varg_list, starpu_tag_t);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list, double);
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
		{
			(void)va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			va_arg(varg_list, int);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}

	}
	va_end(varg_list);

	if (do_execute == -1)
	{
		int i;
		size_t max_size = 0;
		for(i=0 ; i<nb_nodes ; i++)
		{
			if (size_on_nodes[i] > max_size)
			{
				max_size = size_on_nodes[i];
				xrank = i;
			}
		}
		if (xrank != -1)
		{
			_STARPU_MPI_DEBUG(1, "Node %d is having the most R data\n", xrank);
			do_execute = 1;
		}
	}
	free(size_on_nodes);

	STARPU_ASSERT_MSG(do_execute != -1, "StarPU needs to see a W or a REDUX data which will tell it where to execute the task");

	if (inconsistent_execute == 1)
	{
		if (xrank == -1)
		{
			_STARPU_MPI_DEBUG(1, "Different tasks are owning W data. Needs to specify which one is to execute the codelet, using STARPU_EXECUTE_ON_NODE or STARPU_EXECUTE_ON_DATA\n");
			return -EINVAL;
		}
		else
		{
			do_execute = (me == xrank);
			dest = xrank;
		}
	}
	else if (xrank != -1)
	{
		do_execute = (me == xrank);
		dest = xrank;
	}

	/* Send and receive data as requested */
	va_start(varg_list, codelet);
	current_data = 0;
	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type & STARPU_R || arg_type & STARPU_W || arg_type & STARPU_RW || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;

			_starpu_mpi_exchange_data_before_execution(data, mode, me, dest, do_execute, comm);
			current_data ++;

		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list, int);
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				_starpu_mpi_exchange_data_before_execution(datas[i], STARPU_CODELET_GET_MODE(codelet, current_data), me, dest, do_execute, comm);
				current_data++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			va_arg(varg_list, void *);
			va_arg(varg_list, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list, _starpu_callback_func_t);
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			va_arg(varg_list, starpu_data_handle_t);
		}
		/* STARPU_DATA_ARRAY handled above */
		else if (arg_type==STARPU_TAG)
		{
			(void)va_arg(varg_list, starpu_tag_t);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list, double);
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
		{
			(void)va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			va_arg(varg_list, int);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
	}
	va_end(varg_list);

	if (do_execute)
	{
		/* Get the number of buffers and the size of the arguments */
		va_start(varg_list, codelet);
		arg_buffer_size = _starpu_insert_task_get_arg_size(varg_list);

		/* Pack arguments if needed */
		if (arg_buffer_size)
		{
			va_start(varg_list, codelet);
			_starpu_codelet_pack_args(&arg_buffer, arg_buffer_size, varg_list);
		}

		_STARPU_MPI_DEBUG(1, "Execution of the codelet %p (%s)\n", codelet, codelet->name);
		va_start(varg_list, codelet);

		struct starpu_task *task = starpu_task_create();
		task->cl_arg_free = 1;

		if (codelet->nbuffers > STARPU_NMAXBUFS)
		{
			task->dyn_handles = malloc(codelet->nbuffers * sizeof(starpu_data_handle_t));
		}
		int ret = _starpu_insert_task_create_and_submit(arg_buffer, arg_buffer_size, codelet, &task, varg_list);
		STARPU_ASSERT_MSG(ret==0, "_starpu_insert_task_create_and_submit failure %d", ret);
	}

	va_start(varg_list, codelet);
	current_data = 0;
	while ((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type & STARPU_R || arg_type & STARPU_W || arg_type & STARPU_RW || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;

			_starpu_mpi_exchange_data_after_execution(data, mode, me, xrank, dest, do_execute, comm);
			_starpu_mpi_clear_data_after_execution(data, mode, me, do_execute, comm);
			current_data++;
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list, int);
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				_starpu_mpi_exchange_data_after_execution(datas[i], STARPU_CODELET_GET_MODE(codelet, current_data), me, xrank, dest, do_execute, comm);
				_starpu_mpi_clear_data_after_execution(datas[i], STARPU_CODELET_GET_MODE(codelet, current_data), me, do_execute, comm);
				current_data++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			va_arg(varg_list, void *);
			va_arg(varg_list, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list, _starpu_callback_func_t);
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			va_arg(varg_list, starpu_data_handle_t);
		}
		/* STARPU_DATA_ARRAY handled above */
		else if (arg_type==STARPU_TAG)
		{
			(void)va_arg(varg_list, starpu_tag_t);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list, double);
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
		{
			(void)va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			va_arg(varg_list, int);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
	}
	va_end(varg_list);


	_STARPU_MPI_LOG_OUT();
	return 0;
}

void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	tag = starpu_mpi_data_get_tag(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
	}
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
	}
	MPI_Comm_rank(comm, &me);

	if (node == rank) return;

	if (me == node)
	{
		starpu_mpi_irecv_detached(data_handle, rank, tag, comm, callback, arg);
	}
	else if (me == rank)
	{
		starpu_mpi_isend_detached(data_handle, node, tag, comm, NULL, NULL);
	}
}

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	tag = starpu_mpi_data_get_tag(data_handle);
	if (rank == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
		STARPU_ABORT();
	}
	if (tag == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
		STARPU_ABORT();
	}
	MPI_Comm_rank(comm, &me);

	if (node == rank) return;

	if (me == node)
	{
		MPI_Status status;
		starpu_mpi_recv(data_handle, rank, tag, comm, &status);
	}
	else if (me == rank)
	{
		starpu_mpi_send(data_handle, node, tag, comm);
	}
}

/* TODO: this should rather be implicitly called by starpu_mpi_insert_task when
 * a data previously accessed in REDUX mode gets accessed in R mode. */
void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int me, rank, tag, nb_nodes;

	rank = starpu_mpi_data_get_rank(data_handle);
	tag = starpu_mpi_data_get_tag(data_handle);
	if (rank == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
		STARPU_ABORT();
	}
	if (tag == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
		STARPU_ABORT();
	}

	MPI_Comm_rank(comm, &me);
	MPI_Comm_size(comm, &nb_nodes);

	_STARPU_MPI_DEBUG(1, "Doing reduction for data %p on node %d with %d nodes ...\n", data_handle, rank, nb_nodes);

	// need to count how many nodes have the data in redux mode
	if (me == rank)
	{
		int i;

		for(i=0 ; i<nb_nodes ; i++)
		{
			if (i != rank)
			{
				starpu_data_handle_t new_handle;

				starpu_data_register_same(&new_handle, data_handle);

				_STARPU_MPI_DEBUG(1, "Receiving redux handle from %d in %p ...\n", i, new_handle);

				/* FIXME: we here allocate a lot of data: one
				 * instance per MPI node and per number of
				 * times we are called. We should rather do
				 * that much later, e.g. after data_handle
				 * finished its last read access, by submitting
				 * an empty task A reading data_handle whose
				 * callback submits the mpi comm, whose
				 * callback submits the redux_cl task B with
				 * sequential consistency set to 0, and submit
				 * an empty task C writing data_handle and
				 * depending on task B, just to replug with
				 * implicit data dependencies with tasks
				 * inserted after this reduction.
				 */
				starpu_mpi_irecv_detached(new_handle, i, tag, comm, NULL, NULL);
				starpu_insert_task(data_handle->redux_cl,
						   STARPU_RW, data_handle,
						   STARPU_R, new_handle,
						   0);
				starpu_data_unregister_submit(new_handle);
			}
		}
	}
	else
	{
		_STARPU_MPI_DEBUG(1, "Sending redux handle to %d ...\n", rank);
		starpu_mpi_isend_detached(data_handle, rank, tag, comm, NULL, NULL);
		starpu_insert_task(data_handle->init_cl, STARPU_W, data_handle, 0);
	}
}
