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
#include <util/starpu_task_insert_utils.h>
#include <datawizard/coherency.h>
#include <core/task.h>

#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <starpu_mpi_select_node.h>

#include "starpu_mpi_task_insert.h"

#define _SEND_DATA(data, mode, dest, data_tag, prio, comm, callback, arg)     \
	do {									\
		if (mode & STARPU_SSEND)					\
			starpu_mpi_issend_detached_prio(data, dest, data_tag, prio, comm, callback, arg); 	\
		else												\
			starpu_mpi_isend_detached_prio(data, dest, data_tag, prio, comm, callback, arg);	\
	} while (0)

static void (*pre_submit_hook)(struct starpu_task *task) = NULL;

int starpu_mpi_pre_submit_hook_register(void (*f)(struct starpu_task *))
{
	if (pre_submit_hook)
		_STARPU_MSG("Warning: a pre_submit_hook has already been registered. Please check if you really want to erase the previously registered hook.\n");
	pre_submit_hook = f;
	return 0;
}

int starpu_mpi_pre_submit_hook_unregister()
{
	pre_submit_hook = NULL;
	return 0;
}

int _starpu_mpi_find_executee_node(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int *do_execute, int *inconsistent_execute, int *xrank)
{
	if (mode & STARPU_W || mode & STARPU_REDUX)
	{
		if (!data)
		{
			/* We don't have anything allocated for this.
			 * The application knows we won't do anything
			 * about this task */
			/* Yes, the app could actually not call
			 * task_insert at all itself, this is just a
			 * safeguard. */
			_STARPU_MPI_DEBUG(3, "oh oh\n");
			_STARPU_MPI_LOG_OUT();
			return -EINVAL;
		}

		int mpi_rank = starpu_mpi_data_get_rank(data);
		if (mpi_rank == -1)
		{
			_STARPU_ERROR("Data %p with mode STARPU_W needs to have a valid rank", data);
		}

		if (*xrank == -1)
		{
			// No node has been selected yet
			*xrank = mpi_rank;
			_STARPU_MPI_DEBUG(100, "Codelet is going to be executed by node %d\n", *xrank);
			*do_execute = mpi_rank == STARPU_MPI_PER_NODE || (mpi_rank == me);
		}
		else if (mpi_rank != *xrank)
		{
			_STARPU_MPI_DEBUG(100, "Another node %d had already been selected to execute the codelet, can't now set %d\n", *xrank, mpi_rank);
			*inconsistent_execute = 1;
		}
	}
	_STARPU_MPI_DEBUG(100, "Executing: inconsistent=%d, do_execute=%d, xrank=%d\n", *inconsistent_execute, *do_execute, *xrank);
	return 0;
}

void _starpu_mpi_exchange_data_before_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int xrank, int do_execute, int prio, MPI_Comm comm)
{
	if (data && xrank == STARPU_MPI_PER_NODE)
	{
		STARPU_ASSERT_MSG(starpu_mpi_data_get_rank(data) == STARPU_MPI_PER_NODE, "If task is replicated, it has to access only per-node data");
	}
	if (data && mode & STARPU_R)
	{
		int mpi_rank = starpu_mpi_data_get_rank(data);
		starpu_mpi_tag_t data_tag = starpu_mpi_data_get_tag(data);
		if (mpi_rank == -1)
		{
			_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
		}

		if (do_execute && mpi_rank != STARPU_MPI_PER_NODE && mpi_rank != me)
		{
			/* The node is going to execute the codelet, but it does not own the data, it needs to receive the data from the owner node */
			int already_received = starpu_mpi_cached_receive_set(data);
			if (already_received == 0)
			{
				if (data_tag == -1)
					_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
				_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data, mpi_rank);
				starpu_mpi_irecv_detached(data, mpi_rank, data_tag, comm, NULL, NULL);
			}
			// else the node has already received the data
		}

		if (!do_execute && mpi_rank == me)
		{
			/* The node owns the data, but another node is going to execute the codelet, the node needs to send the data to the executee node. */
			int already_sent = starpu_mpi_cached_send_set(data, xrank);
			if (already_sent == 0)
			{
				if (data_tag == -1)
					_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
				_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data, xrank);
				_SEND_DATA(data, mode, xrank, data_tag, prio, comm, NULL, NULL);
			}
			// Else the data has already been sent
		}
	}
}

static
void _starpu_mpi_exchange_data_after_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int xrank, int do_execute, int prio, MPI_Comm comm)
{
	if (mode & STARPU_W)
	{
		int mpi_rank = starpu_mpi_data_get_rank(data);
		starpu_mpi_tag_t data_tag = starpu_mpi_data_get_tag(data);
		if(mpi_rank == -1)
		{
			_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
		}
		if (mpi_rank == STARPU_MPI_PER_NODE)
		{
			mpi_rank = me;
		}
		if (mpi_rank == me)
		{
			if (xrank != -1 && (xrank != STARPU_MPI_PER_NODE && me != xrank))
			{
				_STARPU_MPI_DEBUG(1, "Receive data %p back from the task %d which executed the codelet ...\n", data, xrank);
				if(data_tag == -1)
					_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
				starpu_mpi_irecv_detached(data, xrank, data_tag, comm, NULL, NULL);
			}
		}
		else if (do_execute)
		{
			if(data_tag == -1)
				_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
			_STARPU_MPI_DEBUG(1, "Send data %p back to its owner %d...\n", data, mpi_rank);
			_SEND_DATA(data, mode, mpi_rank, data_tag, prio, comm, NULL, NULL);
		}
	}
}

static
void _starpu_mpi_clear_data_after_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int do_execute)
{
	if (_starpu_cache_enabled)
	{
		if (mode & STARPU_W || mode & STARPU_REDUX)
		{
			/* The data has been modified, it MUST be removed from the cache */
			starpu_mpi_cached_send_clear(data);
			starpu_mpi_cached_receive_clear(data);
		}
	}
	else
	{
		/* We allocated a temporary buffer for the received data, now drop it */
		if ((mode & STARPU_R) && do_execute)
		{
			int mpi_rank = starpu_mpi_data_get_rank(data);
			if (mpi_rank == STARPU_MPI_PER_NODE)
			{
				mpi_rank = me;
			}
			if (mpi_rank != me && mpi_rank != -1)
			{
				starpu_data_invalidate_submit(data);
			}
		}
	}
}

static
int _starpu_mpi_task_decode_v(struct starpu_codelet *codelet, int me, int nb_nodes, int *xrank, int *do_execute, struct starpu_data_descr **descrs_p, int *nb_data_p, int *prio_p, va_list varg_list)
{
	/* XXX: _fstarpu_mpi_task_decode_v needs to be updated at the same time */
	va_list varg_list_copy;
	int inconsistent_execute = 0;
	int arg_type;
	int node_selected = 0;
	int nb_allocated_data = 16;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio = 0;
	int select_node_policy = STARPU_MPI_NODE_SELECTION_CURRENT_POLICY;

	_STARPU_TRACE_TASK_MPI_DECODE_START();

	_STARPU_MPI_MALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
	nb_data = 0;
	*do_execute = -1;
	*xrank = -1;

	va_copy(varg_list_copy, varg_list);
	while ((arg_type = va_arg(varg_list_copy, int)) != 0)
	{
		int arg_type_nocommute = arg_type & ~STARPU_COMMUTE;
		if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			*xrank = va_arg(varg_list_copy, int);
			if (node_selected == 0)
			{
				_STARPU_MPI_DEBUG(100, "Executing on node %d\n", *xrank);
				*do_execute = 1;
				node_selected = 1;
				inconsistent_execute = 0;
			}
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			starpu_data_handle_t data = va_arg(varg_list_copy, starpu_data_handle_t);
			if (node_selected == 0)
			{
				*xrank = starpu_mpi_data_get_rank(data);
				STARPU_ASSERT_MSG(*xrank != -1, "Rank of the data must be set using starpu_mpi_data_register() or starpu_data_set_rank()");
				_STARPU_MPI_DEBUG(100, "Executing on data node %d\n", *xrank);
				STARPU_ASSERT_MSG(*xrank <= nb_nodes, "Node %d to execute codelet is not a valid node (%d)", *xrank, nb_nodes);
				*do_execute = 1;
				node_selected = 1;
				inconsistent_execute = 0;
			}
		}
		else if (arg_type_nocommute & STARPU_R || arg_type_nocommute & STARPU_W || arg_type_nocommute & STARPU_RW || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list_copy, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;
			if (node_selected == 0)
			{
				int ret = _starpu_mpi_find_executee_node(data, mode, me, do_execute, &inconsistent_execute, xrank);
				if (ret == -EINVAL)
				{
					free(descrs);
					va_end(varg_list_copy);
					_STARPU_TRACE_TASK_MPI_DECODE_END();
					return ret;
				}
			}
			if (nb_data >= nb_allocated_data)
			{
				nb_allocated_data *= 2;
				_STARPU_MPI_REALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
			}
			descrs[nb_data].handle = data;
			descrs[nb_data].mode = mode;
			nb_data ++;
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list_copy, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list_copy, int);
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				STARPU_ASSERT_MSG(codelet->nbuffers == STARPU_VARIABLE_NBUFFERS || nb_data < codelet->nbuffers, "Too many data passed to starpu_mpi_task_insert");
				enum starpu_data_access_mode mode = STARPU_CODELET_GET_MODE(codelet, nb_data);
				if (node_selected == 0)
				{
					int ret = _starpu_mpi_find_executee_node(datas[i], mode, me, do_execute, &inconsistent_execute, xrank);
					if (ret == -EINVAL)
					{
						free(descrs);
						va_end(varg_list_copy);
						_STARPU_TRACE_TASK_MPI_DECODE_END();
						return ret;
					}
				}
				if (nb_data >= nb_allocated_data)
				{
					nb_allocated_data *= 2;
					_STARPU_MPI_REALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
				}
				descrs[nb_data].handle = datas[i];
				descrs[nb_data].mode = mode;
				nb_data ++;
			}
		}
		else if (arg_type == STARPU_DATA_MODE_ARRAY)
		{
			struct starpu_data_descr *_descrs = va_arg(varg_list_copy, struct starpu_data_descr*);
			int nb_handles = va_arg(varg_list_copy, int);
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				enum starpu_data_access_mode mode = _descrs[i].mode;
				if (node_selected == 0)
				{
					int ret = _starpu_mpi_find_executee_node(_descrs[i].handle, mode, me, do_execute, &inconsistent_execute, xrank);
					if (ret == -EINVAL)
					{
						free(descrs);
						va_end(varg_list_copy);
						_STARPU_TRACE_TASK_MPI_DECODE_END();
						return ret;
					}
				}
				if (nb_data >= nb_allocated_data)
				{
					nb_allocated_data *= 2;
					_STARPU_MPI_REALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
				}
				descrs[nb_data].handle = _descrs[i].handle;
				descrs[nb_data].mode = mode;
				nb_data ++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			(void)va_arg(varg_list_copy, void *);
			(void)va_arg(varg_list_copy, size_t);
		}
		else if (arg_type==STARPU_CL_ARGS)
		{
			(void)va_arg(varg_list_copy, void *);
			(void)va_arg(varg_list_copy, size_t);
		}
		else if (arg_type==STARPU_CL_ARGS_NFREE)
		{
			(void)va_arg(varg_list_copy, void *);
			(void)va_arg(varg_list_copy, size_t);
		}
		else if (arg_type==STARPU_TASK_DEPS_ARRAY)
		{
			(void)va_arg(varg_list_copy, unsigned);
			(void)va_arg(varg_list_copy, struct starpu_task **);
		}
		else if (arg_type==STARPU_TASK_END_DEPS_ARRAY)
		{
			(void)va_arg(varg_list_copy, unsigned);
			(void)va_arg(varg_list_copy, struct starpu_task **);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			(void)va_arg(varg_list_copy, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			(void)va_arg(varg_list_copy, _starpu_callback_func_t);
			(void)va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG_NFREE)
		{
			(void)va_arg(varg_list_copy, _starpu_callback_func_t);
			(void)va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			(void)va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG_NFREE)
		{
			(void)va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			prio = va_arg(varg_list_copy, int);
		}
		/* STARPU_EXECUTE_ON_NODE handled above */
		/* STARPU_EXECUTE_ON_DATA handled above */
		/* STARPU_DATA_ARRAY handled above */
		/* STARPU_DATA_MODE_ARRAY handled above */
		else if (arg_type==STARPU_TAG)
		{
			(void)va_arg(varg_list_copy, starpu_tag_t);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list_copy, double);
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			(void)va_arg(varg_list_copy, unsigned);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
                {
			(void)va_arg(varg_list_copy, _starpu_callback_func_t);
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
                {
                        (void)va_arg(varg_list_copy, void *);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG_NFREE)
                {
                        (void)va_arg(varg_list_copy, void *);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
                {
			(void)va_arg(varg_list_copy, _starpu_callback_func_t);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
                {
                        (void)va_arg(varg_list_copy, void *);
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE)
                {
                        (void)va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_EXECUTE_WHERE)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			(void)va_arg(varg_list_copy, unsigned long long);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			(void)va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_TAG_ONLY)
		{
			(void)va_arg(varg_list_copy, starpu_tag_t);
		}
		else if (arg_type==STARPU_NAME)
		{
			(void)va_arg(varg_list_copy, const char *);
		}
		else if (arg_type==STARPU_POSSIBLY_PARALLEL)
		{
			(void)va_arg(varg_list_copy, unsigned);
		}
		else if (arg_type==STARPU_WORKER_ORDER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			(void)va_arg(varg_list_copy, unsigned);
		}
		else if (arg_type==STARPU_NODE_SELECTION_POLICY)
		{
			select_node_policy = va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_TASK_COLOR)
                {
                        (void)va_arg(varg_list_copy, int);
                }
		else if (arg_type==STARPU_TASK_SYNCHRONOUS)
                {
                        (void)va_arg(varg_list_copy, int);
                }
		else if (arg_type==STARPU_HANDLES_SEQUENTIAL_CONSISTENCY)
                {
                        (void)va_arg(varg_list_copy, char *);
                }
		else if (arg_type==STARPU_TASK_END_DEP)
		{
			(void)va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_TASK_WORKERIDS)
		{
			(void)va_arg(varg_list_copy, unsigned);
			(void)va_arg(varg_list_copy, uint32_t*);
		}
		else if (arg_type==STARPU_SEQUENTIAL_CONSISTENCY)
		{
			(void)va_arg(varg_list_copy, unsigned);
		}
		else if (arg_type==STARPU_TASK_PROFILING_INFO)
		{
			(void)va_arg(varg_list_copy, struct starpu_profiling_task_info *);
		}
		else if (arg_type==STARPU_TASK_NO_SUBMITORDER)
		{
			(void)va_arg(varg_list_copy, unsigned);
		}
		else if (arg_type==STARPU_TASK_SCHED_DATA)
		{
			(void)va_arg(varg_list_copy, void *);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d, did you perhaps forget to end arguments with 0?\n", arg_type);
		}

	}
	va_end(varg_list_copy);

	if (inconsistent_execute == 1 || *xrank == -1)
	{
		// We need to find out which node is going to execute the codelet.
		_STARPU_MPI_DEBUG(100, "Different nodes are owning W data. The node to execute the codelet is going to be selected with the current selection node policy. See starpu_mpi_node_selection_set_current_policy() to change the policy, or use STARPU_EXECUTE_ON_NODE or STARPU_EXECUTE_ON_DATA to specify the node\n");
		*xrank = _starpu_mpi_select_node(me, nb_nodes, descrs, nb_data, select_node_policy);
		*do_execute = *xrank == STARPU_MPI_PER_NODE || (me == *xrank);
	}
	else
	{
		_STARPU_MPI_DEBUG(100, "Inconsistent=%d - xrank=%d\n", inconsistent_execute, *xrank);
		*do_execute = *xrank == STARPU_MPI_PER_NODE || (me == *xrank);
	}
	_STARPU_MPI_DEBUG(100, "do_execute=%d\n", *do_execute);

	*descrs_p = descrs;
	*nb_data_p = nb_data;
	*prio_p = prio;

	_STARPU_TRACE_TASK_MPI_DECODE_END();
	return 0;
}

static
int _starpu_mpi_task_build_v(MPI_Comm comm, struct starpu_codelet *codelet, struct starpu_task **task, int *xrank_p, struct starpu_data_descr **descrs_p, int *nb_data_p, int *prio_p, va_list varg_list)
{
	int me, do_execute, xrank, nb_nodes;
	int ret;
	int i;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio;

	_STARPU_MPI_LOG_IN();

	starpu_mpi_comm_rank(comm, &me);
	starpu_mpi_comm_size(comm, &nb_nodes);

	/* Find out whether we are to execute the data because we own the data to be written to. */
	ret = _starpu_mpi_task_decode_v(codelet, me, nb_nodes, &xrank, &do_execute, &descrs, &nb_data, &prio, varg_list);
	if (ret < 0)
		return ret;

	_STARPU_TRACE_TASK_MPI_PRE_START();
	/* Send and receive data as requested */
	for(i=0 ; i<nb_data ; i++)
	{
		_starpu_mpi_exchange_data_before_execution(descrs[i].handle, descrs[i].mode, me, xrank, do_execute, prio, comm);
	}

	if (xrank_p)
		*xrank_p = xrank;
	if (nb_data_p)
		*nb_data_p = nb_data;
	if (prio_p)
		*prio_p = prio;

	if (descrs_p)
		*descrs_p = descrs;
	else
		free(descrs);


	_STARPU_TRACE_TASK_MPI_PRE_END();

	if (do_execute == 0)
	{
		return 1;
	}
	else
	{
		va_list varg_list_copy;
		_STARPU_MPI_DEBUG(100, "Execution of the codelet %p (%s)\n", codelet, codelet?codelet->name:NULL);

		*task = starpu_task_create();
		(*task)->cl_arg_free = 1;
		(*task)->callback_arg_free = 1;
		(*task)->prologue_callback_arg_free = 1;
		(*task)->prologue_callback_pop_arg_free = 1;

		va_copy(varg_list_copy, varg_list);
		_starpu_task_insert_create(codelet, *task, varg_list_copy);
		va_end(varg_list_copy);

		return 0;
	}
}

int _starpu_mpi_task_postbuild_v(MPI_Comm comm, int xrank, int do_execute, struct starpu_data_descr *descrs, int nb_data, int prio)
{
	int me, i;

	_STARPU_TRACE_TASK_MPI_POST_START();
	starpu_mpi_comm_rank(comm, &me);

	for(i=0 ; i<nb_data ; i++)
	{
		_starpu_mpi_exchange_data_after_execution(descrs[i].handle, descrs[i].mode, me, xrank, do_execute, prio, comm);
		_starpu_mpi_clear_data_after_execution(descrs[i].handle, descrs[i].mode, me, do_execute);
	}

	free(descrs);

	_STARPU_TRACE_TASK_MPI_POST_END();
	_STARPU_MPI_LOG_OUT();
	return 0;
}

static
int _starpu_mpi_task_insert_v(MPI_Comm comm, struct starpu_codelet *codelet, va_list varg_list)
{
	struct starpu_task *task;
	int ret;
	int xrank;
	int do_execute = 0;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio;

	ret = _starpu_mpi_task_build_v(comm, codelet, &task, &xrank, &descrs, &nb_data, &prio, varg_list);
	if (ret < 0)
		return ret;

	if (ret == 0)
	{
		do_execute = 1;
		ret = starpu_task_submit(task);

		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			_STARPU_MSG("submission of task %p wih codelet %p failed (symbol `%s') (err: ENODEV)\n",
				    task, task->cl,
				    (codelet == NULL) ? "none" :
				    task->cl->name ? task->cl->name :
				    (task->cl->model && task->cl->model->symbol)?task->cl->model->symbol:"none");

			task->destroy = 0;
			starpu_task_destroy(task);
		}
	}

	int val = _starpu_mpi_task_postbuild_v(comm, xrank, do_execute, descrs, nb_data, prio);

	if (ret == 0 && pre_submit_hook)
		pre_submit_hook(task);

	return val;
}

int starpu_mpi_task_insert(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	va_list varg_list;
	int ret;

	va_start(varg_list, codelet);
	ret = _starpu_mpi_task_insert_v(comm, codelet, varg_list);
	va_end(varg_list);
	return ret;
}

int starpu_mpi_insert_task(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	va_list varg_list;
	int ret;

	va_start(varg_list, codelet);
	ret = _starpu_mpi_task_insert_v(comm, codelet, varg_list);
	va_end(varg_list);
	return ret;
}

struct starpu_task *starpu_mpi_task_build(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	va_list varg_list;
	struct starpu_task *task;
	int ret;

	va_start(varg_list, codelet);
	ret = _starpu_mpi_task_build_v(comm, codelet, &task, NULL, NULL, NULL, NULL, varg_list);
	va_end(varg_list);
	STARPU_ASSERT(ret >= 0);
	return (ret > 0) ? NULL : task;
}

int starpu_mpi_task_post_build(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	int xrank, do_execute;
	int ret, me, nb_nodes;
	va_list varg_list;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio;

	starpu_mpi_comm_rank(comm, &me);
	starpu_mpi_comm_size(comm, &nb_nodes);

	va_start(varg_list, codelet);
	/* Find out whether we are to execute the data because we own the data to be written to. */
	ret = _starpu_mpi_task_decode_v(codelet, me, nb_nodes, &xrank, &do_execute, &descrs, &nb_data, &prio, varg_list);
	va_end(varg_list);
	if (ret < 0)
		return ret;

	return _starpu_mpi_task_postbuild_v(comm, xrank, do_execute, descrs, nb_data, prio);
}

struct _starpu_mpi_redux_data_args
{
	starpu_data_handle_t data_handle;
	starpu_data_handle_t new_handle;
	starpu_mpi_tag_t data_tag;
	int node;
	MPI_Comm comm;
	struct starpu_task *taskB;
};

void _starpu_mpi_redux_data_dummy_func(void *buffers[], void *cl_arg)
{
	(void)buffers;
	(void)cl_arg;
}

/* Dummy cost function for simgrid */
static double cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	return 0.000001;
}
static struct starpu_perfmodel dumb_model =
{
	.type		= STARPU_COMMON,
	.cost_function	= cost_function
};

static
struct starpu_codelet _starpu_mpi_redux_data_read_cl =
{
	.cpu_funcs = {_starpu_mpi_redux_data_dummy_func},
	.cuda_funcs = {_starpu_mpi_redux_data_dummy_func},
	.opencl_funcs = {_starpu_mpi_redux_data_dummy_func},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &dumb_model,
	.name = "_starpu_mpi_redux_data_read_cl"
};

struct starpu_codelet _starpu_mpi_redux_data_readwrite_cl =
{
	.cpu_funcs = {_starpu_mpi_redux_data_dummy_func},
	.cuda_funcs = {_starpu_mpi_redux_data_dummy_func},
	.opencl_funcs = {_starpu_mpi_redux_data_dummy_func},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &dumb_model,
	.name = "_starpu_mpi_redux_data_write_cl"
};

static
void _starpu_mpi_redux_data_detached_callback(void *arg)
{
	struct _starpu_mpi_redux_data_args *args = (struct _starpu_mpi_redux_data_args *) arg;

	STARPU_TASK_SET_HANDLE(args->taskB, args->new_handle, 1);
	int ret = starpu_task_submit(args->taskB);
	STARPU_ASSERT(ret == 0);

	starpu_data_unregister_submit(args->new_handle);
	free(args);
}

static
void _starpu_mpi_redux_data_recv_callback(void *callback_arg)
{
	struct _starpu_mpi_redux_data_args *args = (struct _starpu_mpi_redux_data_args *) callback_arg;
	starpu_data_register_same(&args->new_handle, args->data_handle);

	starpu_mpi_irecv_detached_sequential_consistency(args->new_handle, args->node, args->data_tag, args->comm, _starpu_mpi_redux_data_detached_callback, args, 0);
}

/* TODO: this should rather be implicitly called by starpu_mpi_task_insert when
 * a data previously accessed in REDUX mode gets accessed in R mode. */
void starpu_mpi_redux_data_prio(MPI_Comm comm, starpu_data_handle_t data_handle, int prio)
{
	int me, rank, nb_nodes;
	starpu_mpi_tag_t tag;

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

	starpu_mpi_comm_rank(comm, &me);
	starpu_mpi_comm_size(comm, &nb_nodes);

	_STARPU_MPI_DEBUG(1, "Doing reduction for data %p on node %d with %d nodes ...\n", data_handle, rank, nb_nodes);

	// need to count how many nodes have the data in redux mode
	if (me == rank)
	{
		int i;

		// taskC depends on all taskBs created
		struct starpu_task *taskC = starpu_task_create();
		taskC->cl = &_starpu_mpi_redux_data_readwrite_cl;
		STARPU_TASK_SET_HANDLE(taskC, data_handle, 0);

		for(i=0 ; i<nb_nodes ; i++)
		{
			if (i != rank)
			{
				/* We need to make sure all is
				 * executed after data_handle finished
				 * its last read access, we hence do
				 * the following:
				 * - submit an empty task A reading
				 * data_handle whose callback submits
				 * the mpi comm with sequential
				 * consistency set to 0, whose
				 * callback submits the redux_cl task
				 * B with sequential consistency set
				 * to 0,
				 * - submit an empty task C reading
				 * and writing data_handle and
				 * depending on task B, just to replug
				 * with implicit data dependencies
				 * with tasks inserted after this
				 * reduction.
				 */

				struct _starpu_mpi_redux_data_args *args;
				_STARPU_MPI_MALLOC(args, sizeof(struct _starpu_mpi_redux_data_args));
				args->data_handle = data_handle;
				args->data_tag = tag;
				args->node = i;
				args->comm = comm;

				// We need to create taskB early as
				// taskC declares a dependancy on it
				args->taskB = starpu_task_create();
				args->taskB->cl = args->data_handle->redux_cl;
				args->taskB->sequential_consistency = 0;
				STARPU_TASK_SET_HANDLE(args->taskB, args->data_handle, 0);

				starpu_task_declare_deps_array(taskC, 1, &args->taskB);

				// Submit taskA
				starpu_task_insert(&_starpu_mpi_redux_data_read_cl,
						   STARPU_R, data_handle,
						   STARPU_CALLBACK_WITH_ARG_NFREE, _starpu_mpi_redux_data_recv_callback, args,
						   0);
			}
		}

		int ret = starpu_task_submit(taskC);
		STARPU_ASSERT(ret == 0);
	}
	else
	{
		_STARPU_MPI_DEBUG(1, "Sending redux handle to %d ...\n", rank);
		starpu_mpi_isend_detached_prio(data_handle, rank, tag, prio, comm, NULL, NULL);
		starpu_task_insert(data_handle->init_cl, STARPU_W, data_handle, 0);
	}
	/* FIXME: In order to prevent simultaneous receive submissions
	 * on the same handle, we need to wait that all the starpu_mpi
	 * tasks are done before submitting next tasks. The current
	 * version of the implementation does not support multiple
	 * simultaneous receive requests on the same handle.*/
	starpu_task_wait_for_all();

}
void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	return starpu_mpi_redux_data_prio(comm, data_handle, 0);
}
