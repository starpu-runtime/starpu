/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	_STARPU_TRACE_TASK_MPI_DECODE_START();

	_STARPU_MPI_MALLOC(descrs, nb_allocated_data * sizeof(struct starpu_data_descr));
	nb_data = 0;
	*do_execute = -1;
	*xrank = -1;

	while (arglist[arg_i] != NULL)
	{
		int arg_type = (int)(intptr_t)arglist[arg_i];
		int arg_type_nocommute = arg_type & ~STARPU_COMMUTE;

		if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			arg_i++;
			*xrank = *(int *)arglist[arg_i];
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
		}
		else if (arg_type_nocommute & STARPU_R || arg_type_nocommute & STARPU_W || arg_type_nocommute & STARPU_RW || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			arg_i++;
			starpu_data_handle_t data = arglist[arg_i];
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;
			if (node_selected == 0)
			{
				int ret = _starpu_mpi_find_executee_node(data, mode, me, do_execute, &inconsistent_execute, xrank);
				if (ret == -EINVAL)
				{
					free(descrs);
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
			arg_i++;
			starpu_data_handle_t *datas = arglist[arg_i];
			arg_i++;
			int nb_handles = *(int *)arglist[arg_i];
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
			arg_i++;
			/* void* */
			arg_i++;
			/* size_t */
		}
		else if (arg_type==STARPU_CL_ARGS)
		{
			arg_i++;
			/* void* */
			arg_i++;
			/* size_t */
		}
		else if (arg_type==STARPU_CL_ARGS_NFREE)
		{
			arg_i++;
			/* void* */
			arg_i++;
			/* size_t */
		}
		else if (arg_type==STARPU_TASK_DEPS_ARRAY)
		{
			arg_i++;
			/* unsigned */
			arg_i++;
			/* struct starpu_task ** */
		}
		else if (arg_type==STARPU_TASK_END_DEPS_ARRAY)
		{
			arg_i++;
			/* unsigned */
			arg_i++;
			/* struct starpu_task ** */
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			arg_i++;
			/* _starpu_callback_func_t */
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			arg_i++;
			/* _starpu_callback_func_t */
			arg_i++;
			/* void* */
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG_NFREE)
		{
			arg_i++;
			/* _starpu_callback_func_t */
			arg_i++;
			/* void* */
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			arg_i++;
			/* void* */
		}
		else if (arg_type==STARPU_CALLBACK_ARG_NFREE)
		{
			arg_i++;
			/* void* */
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			arg_i++;
			prio = *(int *)arglist[arg_i];
			/* int* */
		}
		/* STARPU_EXECUTE_ON_NODE handled above */
		/* STARPU_EXECUTE_ON_DATA handled above */
		/* STARPU_DATA_ARRAY handled above */
		/* STARPU_DATA_MODE_ARRAY handled above */
		else if (arg_type==STARPU_TAG)
		{
			arg_i++;
			/* starpu_tag_t* */
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			arg_i++;
			/* int* */
		}
		else if (arg_type==STARPU_FLOPS)
		{
			arg_i++;
			/* double* */
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			arg_i++;
			/* unsigned* */
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
                {
			arg_i++;
			/* _starpu_callback_func_t */
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
                {
			arg_i++;
			/* void* */
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG_NFREE)
                {
			arg_i++;
			/* void* */
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
                {
			arg_i++;
			/* _starpu_callback_func_t */
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
                {
			arg_i++;
			/* void* */
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE)
                {
			arg_i++;
			/* void* */
		}
		else if (arg_type==STARPU_EXECUTE_WHERE)
		{
			arg_i++;
			/* int* */
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			arg_i++;
			/* int* */
		}
		else if (arg_type==STARPU_TAG_ONLY)
		{
			arg_i++;
			/* starpu_tag_t* */
		}
		else if (arg_type==STARPU_NAME)
		{
			arg_i++;
			/* char* */
		}
		else if (arg_type==STARPU_POSSIBLY_PARALLEL)
		{
			arg_i++;
			/* unsigned* */
		}
		else if (arg_type==STARPU_WORKER_ORDER)
		{
			arg_i++;
			/* unsigned* */
		}
		else if (arg_type==STARPU_NODE_SELECTION_POLICY)
		{
			arg_i++;
			/* int* */
		}
		else if (arg_type==STARPU_TASK_COLOR)
		{
			arg_i++;
			/* int* */
		}
		else if (arg_type==STARPU_TASK_SYNCHRONOUS)
		{
			arg_i++;
			/* int* */
		}
		else if (arg_type==STARPU_HANDLES_SEQUENTIAL_CONSISTENCY)
		{
			arg_i++;
			/* char* */
		}
		else if (arg_type==STARPU_TASK_END_DEP)
		{
			arg_i++;
			/* int */
		}
		else if (arg_type==STARPU_TASK_WORKERIDS)
		{
			arg_i++;
			/* unsigned */
			arg_i++;
			/* uint32_t* */
		}
		else if (arg_type==STARPU_SEQUENTIAL_CONSISTENCY)
		{
			arg_i++;
			/* unsigned */
		}
		else if (arg_type==STARPU_TASK_PROFILING_INFO)
		{
			arg_i++;
			/* struct starpu_profiling_task_info * */
		}
		else if (arg_type==STARPU_TASK_NO_SUBMITORDER)
		{
			arg_i++;
			/* unsigned */
		}
		else if (arg_type==STARPU_TASK_SCHED_DATA)
		{
			arg_i++;
			/* void * */
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d, did you perhaps forget to end arguments with 0?\n", arg_type);
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

	_STARPU_TRACE_TASK_MPI_DECODE_END();
	return 0;
}

static
int _fstarpu_mpi_task_build_v(MPI_Comm comm, struct starpu_codelet *codelet, struct starpu_task **task, int *xrank_p, struct starpu_data_descr **descrs_p, int *nb_data_p, int *prio_p, void **arglist)
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
		_STARPU_MPI_DEBUG(100, "Execution of the codelet %p (%s)\n", codelet, codelet?codelet->name:NULL);

		*task = starpu_task_create();
		(*task)->cl_arg_free = 1;
		(*task)->callback_arg_free = 1;
		(*task)->prologue_callback_arg_free = 1;
		(*task)->prologue_callback_pop_arg_free = 1;

		_fstarpu_task_insert_create(codelet, *task, arglist);
		return 0;
	}
}

static
int _fstarpu_mpi_task_insert_v(MPI_Comm comm, struct starpu_codelet *codelet, void **arglist)
{
	struct starpu_task *task;
	int ret;
	int xrank;
	int do_execute = 0;
	struct starpu_data_descr *descrs;
	int nb_data;
	int prio;

	ret = _fstarpu_mpi_task_build_v(comm, codelet, &task, &xrank, &descrs, &nb_data, &prio, arglist);
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
	return _starpu_mpi_task_postbuild_v(comm, xrank, do_execute, descrs, nb_data, prio);
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

	ret = _fstarpu_mpi_task_build_v(MPI_Comm_f2c(comm), codelet, &task, NULL, NULL, NULL, NULL, arglist+2);
	STARPU_ASSERT(ret >= 0);
	return (ret > 0) ? NULL : task;
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

	ret = _starpu_mpi_task_postbuild_v(MPI_Comm_f2c(comm), xrank, do_execute, descrs, nb_data, prio);
	STARPU_ASSERT(ret >= 0);
}

#endif /* HAVE_MPI_COMM_F2C */
