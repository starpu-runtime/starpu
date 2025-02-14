/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi.h>
#include <common/config.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_task_insert.h>
#include <starpu_mpi_select_node.h>
#include <util/starpu_task_insert_utils.h>
#include <datawizard/coherency.h>
#include <core/task.h>
#include <core/workers.h>

#ifdef HAVE_MPI_COMM_F2C
static
int _fstarpu_mpi_task_decode_v(struct starpu_codelet *codelet, int me, int nb_nodes, int *xrank, int *do_execute, struct starpu_data_descr **descrs_p, int *nb_data_p, int *prio_p, void **arglist)
{
	int arg_i = 0;
	int inconsistent_execute = 0;
	int node_selected = 0;
	int nb_allocated_data = 16;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio = 0;
	int select_node_policy = STARPU_MPI_NODE_SELECTION_CURRENT_POLICY;

	_starpu_trace_task_mpi_decode_start();

	_STARPU_MPI_MALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
	nb_data = 0;
	*do_execute = -1;
	*xrank = -1;

	while (arglist[arg_i] != NULL)
	{
		int arg_type = (int)(intptr_t)arglist[arg_i];
		int arg_type_nocommute = arg_type & ~STARPU_COMMUTE;

		if (arg_type_nocommute & STARPU_R || arg_type_nocommute & STARPU_W || arg_type_nocommute & STARPU_RW || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX || arg_type & STARPU_MPI_REDUX)
		{
			arg_i++;
			starpu_data_handle_t data = arglist[arg_i];
			arg_i++;
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;
			if (node_selected == 0)
			{
				int ret = _starpu_mpi_find_executee_node(data, mode, me, do_execute, &inconsistent_execute, xrank);
				if (ret == -EINVAL)
				{
					free(descrs);
					_starpu_trace_task_mpi_decode_end();
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
			continue;
		}
		switch(arg_type)
		{
		case STARPU_CALLBACK:
		case STARPU_CALLBACK_ARG:
		case STARPU_CALLBACK_ARG_NFREE:
		{
			arg_i++;
			break;
		}
		case STARPU_CALLBACK_WITH_ARG:
		case STARPU_CALLBACK_WITH_ARG_NFREE:
		case STARPU_CL_ARGS:
		case STARPU_CL_ARGS_NFREE:
		{
			arg_i++;
			arg_i++;
			break;
		}
		case  STARPU_DATA_ARRAY:
		{
			arg_i++;
			starpu_data_handle_t *data = arglist[arg_i];
			arg_i++;
			int nb_handles = *(int *)arglist[arg_i];
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				STARPU_ASSERT_MSG(codelet->nbuffers == STARPU_VARIABLE_NBUFFERS || nb_data < codelet->nbuffers, "Too many data passed to starpu_mpi_task_insert");
				enum starpu_data_access_mode mode = STARPU_CODELET_GET_MODE(codelet, nb_data);
				if (node_selected == 0)
				{
					int ret = _starpu_mpi_find_executee_node(data[i], mode, me, do_execute, &inconsistent_execute, xrank);
					if (ret == -EINVAL)
					{
						free(descrs);
						_starpu_trace_task_mpi_decode_end();
						return ret;
					}
				}
				if (nb_data >= nb_allocated_data)
				{
					nb_allocated_data *= 2;
					_STARPU_MPI_REALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
				}
				descrs[nb_data].handle = data[i];
				descrs[nb_data].mode = mode;
				nb_data ++;
			}
			break;
		}
		case  STARPU_DATA_MODE_ARRAY:
		{
			arg_i++;
			struct starpu_data_descr *_descrs = arglist[arg_i];
			arg_i++;
			int nb_handles = *(int *)arglist[arg_i];
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
						_starpu_trace_task_mpi_decode_end();
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
			break;
		}
		case STARPU_EPILOGUE_CALLBACK:
		case STARPU_EPILOGUE_CALLBACK_ARG:
		{
			arg_i++;
			break;
		}
		case STARPU_EXECUTE_ON_DATA:
		{
			arg_i++;
			starpu_data_handle_t data = arglist[arg_i];
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
			break;
		}
		case STARPU_EXECUTE_ON_NODE:
		{
			arg_i++;
			int rank = *(int *)arglist[arg_i];
			if (rank != -1)
			{
				*xrank = rank;
				if (node_selected == 0)
				{
					_STARPU_MPI_DEBUG(100, "Executing on node %d\n", *xrank);
					*do_execute = 1;
					node_selected = 1;
					inconsistent_execute = 0;
				}
			}
			break;
		}
		case STARPU_EXECUTE_ON_WORKER:
		case STARPU_EXECUTE_WHERE:
		case STARPU_HANDLES_SEQUENTIAL_CONSISTENCY:
		case STARPU_FLOPS:
		case STARPU_HYPERVISOR_TAG:
		case STARPU_NAME:
		case STARPU_NODE_SELECTION_POLICY:
		case STARPU_NONE:
		case STARPU_POSSIBLY_PARALLEL:
		{
			arg_i++;
			break;
		}
		case STARPU_PRIORITY:
		{
			arg_i++;
			prio = *(int *)arglist[arg_i];
			break;
		}
		case STARPU_PROLOGUE_CALLBACK:
		case STARPU_PROLOGUE_CALLBACK_ARG:
		case STARPU_PROLOGUE_CALLBACK_ARG_NFREE:
		case STARPU_PROLOGUE_CALLBACK_POP:
		case STARPU_PROLOGUE_CALLBACK_POP_ARG:
		case STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE:
		{
			arg_i++;
			break;
		}
#ifdef STARPU_RECURSIVE_TASKS
		case STARPU_RECURSIVE_TASK_FUNC:
		case STARPU_RECURSIVE_TASK_FUNC_ARG:
		case STARPU_RECURSIVE_TASK_GEN_DAG_FUNC:
		case STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG:
		case STARPU_RECURSIVE_TASK_PARENT:
		{
			STARPU_ASSERT_MSG(0, "Recursive Tasks + MPI not supported yet\n");
			arg_i++;
			break;
		}
#endif
		case STARPU_SCHED_CTX:
		case STARPU_SEQUENTIAL_CONSISTENCY:
		case STARPU_SOON_CALLBACK:
		case STARPU_SOON_CALLBACK_ARG:
		case STARPU_SOON_CALLBACK_ARG_NFREE:
		case STARPU_TAG:
		case STARPU_TAG_ONLY:
		case STARPU_TASK_COLOR:
		{
			arg_i++;
			break;
		}
		case STARPU_TASK_DEPS_ARRAY:
		{
			arg_i++;
			arg_i++;
			break;
		}
		case STARPU_TASK_END_DEP:
		{
			arg_i++;
			break;
		}
		case STARPU_TASK_END_DEPS_ARRAY:
		{
			arg_i++;
			arg_i++;
			break;
		}
		case STARPU_TASK_FILE:
		case STARPU_TASK_LINE:
		case STARPU_TASK_NO_SUBMITORDER:
		case STARPU_TASK_PROFILING_INFO:
		case STARPU_TASK_SCHED_DATA:
		case STARPU_TASK_SYNCHRONOUS:
		{
			arg_i++;
			break;
		}
		case STARPU_TASK_WORKERIDS:
		{
			arg_i++;
			arg_i++;
			break;
		}
		case STARPU_TRANSACTION:
		{
			arg_i++;
			break;
		}
		case STARPU_VALUE:
		{
			arg_i++;
			arg_i++;
			break;
		}
		case STARPU_WORKER_ORDER:
		{
			arg_i++;
			break;
		}
		default:
		{
			STARPU_ABORT_MSG("Unrecognized argument %d, did you perhaps forget to end arguments with 0?\n", arg_type);
		}
		}
		arg_i++;
	}

	if (inconsistent_execute == 1 || *xrank == -1)
	{
		// We need to find out which node is going to execute the codelet.
		_STARPU_MPI_DISP("Different nodes are owning W data. The node to execute the codelet is going to be selected with the current selection node policy. See starpu_mpi_node_selection_set_current_policy() to change the policy, or use STARPU_EXECUTE_ON_NODE or STARPU_EXECUTE_ON_DATA to specify the node\n");
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

	_starpu_trace_task_mpi_decode_end();
	return 0;
}

static
int _fstarpu_mpi_task_build_v(MPI_Comm comm, struct starpu_codelet *codelet, struct starpu_task **task, int *xrank_p, struct starpu_data_descr **descrs_p, int *nb_data_p, int *prio_p, int *exchange_needed, void **arglist)
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
	ret = _fstarpu_mpi_task_decode_v(codelet, me, nb_nodes, &xrank, &do_execute, &descrs, &nb_data, &prio, arglist);
	if (ret < 0)
		return ret;

	_starpu_trace_task_mpi_pre_start();
	/* Send and receive data as requested */
	for(i=0 ; i<nb_data ; i++)
	{
		_starpu_mpi_exchange_data_before_execution(descrs[i].handle, descrs[i].mode, me, xrank, do_execute, prio, comm, exchange_needed);
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

	if (do_execute == 1)
	{
		_STARPU_MPI_DEBUG(100, "Execution of the codelet %p (%s)\n", codelet, codelet?codelet->name:NULL);

		*task = starpu_task_create();
		(*task)->cl_arg_free = 1;
		(*task)->callback_arg_free = 1;
		(*task)->prologue_callback_arg_free = 1;
		(*task)->prologue_callback_pop_arg_free = 1;

		_fstarpu_task_insert_create(codelet, *task, arglist);

		if ((*task)->cl)
		{
			/* we suppose the current context is not going to change between now and the execution of the task */
			(*task)->sched_ctx = _starpu_sched_ctx_get_current_context();
			/* Check the type of worker(s) required by the task exist */
			if (STARPU_UNLIKELY(!_starpu_worker_exists(*task)))
			{
				_STARPU_MPI_DEBUG(0, "There is no worker to execute the codelet %p (%s)\n", codelet, codelet?codelet->name:NULL);
				return -ENODEV;
			}

			/* In case we require that a task should be explicitly
			 * executed on a specific worker, we make sure that the worker
			 * is able to execute this task.  */
			if (STARPU_UNLIKELY((*task)->execute_on_a_specific_worker && !starpu_combined_worker_can_execute_task((*task)->workerid, *task, 0)))
			{
				_STARPU_MPI_DEBUG(0, "The specified worker %d cannot execute the codelet %p (%s)\n", (*task)->workerid, codelet, codelet?codelet->name:NULL);
				return -ENODEV;
			}
		}
	}

	_starpu_trace_task_mpi_pre_end();

	return do_execute;
}

static
int _fstarpu_mpi_task_insert_v(MPI_Comm comm, struct starpu_codelet *codelet, void **arglist)
{
	struct starpu_task *task;
	int ret;
	int me;
	int xrank;
	int do_execute = 0;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio;
	int exchange_needed;

	starpu_mpi_comm_rank(comm, &me);
	ret = _fstarpu_mpi_task_build_v(comm, codelet, &task, &xrank, &descrs, &nb_data, &prio, &exchange_needed, arglist);
	if (ret < 0)
		return ret;

	if (ret == 1)
	{
		do_execute = 1;
		ret = starpu_task_submit(task);

		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			_STARPU_MSG("submission of task %p with codelet %p failed (symbol `%s') (err: ENODEV)\n",
				    task, task->cl,
				    (codelet == NULL) ? "none" :
				    task->cl->name ? task->cl->name :
				    (task->cl->model && task->cl->model->symbol)?task->cl->model->symbol:"none");

			task->destroy = 0;
			starpu_task_destroy(task);
			free(descrs);
			return -ENODEV;
		}
	}

	int val = _starpu_mpi_task_postbuild_v(comm, me, xrank, do_execute, descrs, nb_data, prio, exchange_needed);
	free(descrs);

	if (ret == 1)
		_starpu_mpi_pre_submit_hook_call(task);

	return val;
}

void fstarpu_mpi_task_insert(void **arglist)
{
	MPI_Fint comm = *((MPI_Fint *)arglist[0]);
	struct starpu_codelet *codelet = arglist[1];
	if (codelet == NULL)
	{
		STARPU_ABORT_MSG("task without codelet");
	}

	int ret;
	ret = _fstarpu_mpi_task_insert_v(MPI_Comm_f2c(comm), codelet, arglist+2);
	STARPU_ASSERT(ret >= 0);
}

/* fstarpu_mpi_insert_task: aliased to fstarpu_mpi_task_insert in fstarpu_mpi_mod.f90 */

struct starpu_task *fstarpu_mpi_task_build(void **arglist)
{
	MPI_Fint comm = *((MPI_Fint *)arglist[0]);
	struct starpu_codelet *codelet = arglist[1];
	if (codelet == NULL)
	{
		STARPU_ABORT_MSG("task without codelet");
	}
	struct starpu_task *task;
	int ret;

	ret = _fstarpu_mpi_task_build_v(MPI_Comm_f2c(comm), codelet, &task, NULL, NULL, NULL, NULL, NULL, arglist+2);
	return (ret == 1 || ret == -ENODEV) ? task : NULL;
}

void fstarpu_mpi_task_post_build(void **arglist)
{
	MPI_Fint comm = *((MPI_Fint *)arglist[0]);
	struct starpu_codelet *codelet = arglist[1];
	if (codelet == NULL)
	{
		STARPU_ABORT_MSG("task without codelet");
	}
	int xrank, do_execute;
	int ret, me, nb_nodes;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio;

	starpu_mpi_comm_rank(MPI_Comm_f2c(comm), &me);
	starpu_mpi_comm_size(MPI_Comm_f2c(comm), &nb_nodes);

	/* Find out whether we are to execute the data because we own the data to be written to. */
	ret = _fstarpu_mpi_task_decode_v(codelet, me, nb_nodes, &xrank, &do_execute, &descrs, &nb_data, &prio, arglist+2);
	STARPU_ASSERT(ret >= 0);

	ret = _starpu_mpi_task_postbuild_v(MPI_Comm_f2c(comm), me, xrank, do_execute, descrs, nb_data, prio, 1);
	free(descrs);
	STARPU_ASSERT(ret >= 0);
}

#endif /* HAVE_MPI_COMM_F2C */
