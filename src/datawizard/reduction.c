/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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

#include <starpu.h>
#include <common/utils.h>
#include <util/starpu_data_cpy.h>
#include <core/task.h>
#include <datawizard/datawizard.h>
#include <drivers/mp_common/source_common.h>
#include <datawizard/memory_nodes.h>

void starpu_data_set_reduction_methods(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, struct starpu_codelet *init_cl)
{
	starpu_data_set_reduction_methods_with_args(handle, redux_cl, NULL, init_cl, NULL);
}

void starpu_data_set_reduction_methods_with_args(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, void *redux_cl_arg, struct starpu_codelet *init_cl, void *init_cl_arg)
{
	_starpu_spin_lock(&handle->header_lock);

	if (init_cl)
	{
		STARPU_ASSERT_MSG(init_cl->nbuffers == 1, "The initialization method has to take one STARPU_W parameter");
		STARPU_ASSERT_MSG(init_cl->modes[0] == STARPU_W, "The initialization method has to take one STARPU_W parameter");
	}
	if (redux_cl)
	{
		STARPU_ASSERT_MSG(redux_cl->nbuffers == 2, "The reduction method has to take one STARPU_RW|STARPU_COMMUTE parameter and one STARPU_R parameter");
		if (!(redux_cl->modes[0] & STARPU_COMMUTE))
		{
			static int _warned = 0;
			STARPU_HG_DISABLE_CHECKING(_warned);
			if (!_warned)
			{
				_STARPU_DISP("Warning: The reduction method should use STARPU_COMMUTE for its first parameter\n");
				_warned = 1;
			}
			redux_cl->modes[0] |= STARPU_COMMUTE;
		}
		STARPU_ASSERT_MSG(redux_cl->modes[0] == (STARPU_RW | STARPU_COMMUTE), "The first parameter of the reduction method has to use STARPU_RW|STARPU_COMMUTE");
		STARPU_ASSERT_MSG(redux_cl->modes[1] == STARPU_R, "The second parameter of the reduction method has to use STARPU_R");
	}

	_starpu_codelet_check_deprecated_fields(redux_cl);
	_starpu_codelet_check_deprecated_fields(init_cl);

	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure that the flags are applied to the children as well */
		starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
		if (child_handle->nchildren > 0)
			starpu_data_set_reduction_methods_with_args(child_handle, redux_cl, redux_cl_arg, init_cl, init_cl_arg);
	}

	handle->redux_cl = redux_cl;
	handle->init_cl = init_cl;
	handle->redux_cl_arg = redux_cl_arg;
	handle->init_cl_arg = init_cl_arg;

	_starpu_spin_unlock(&handle->header_lock);
}

void _starpu_init_data_replicate(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, int workerid)
{
	STARPU_ASSERT(replicate);
	STARPU_ASSERT(replicate->allocated || replicate->mapped != STARPU_UNMAPPED);

	struct starpu_codelet *init_cl = handle->init_cl;
	STARPU_ASSERT_MSG(init_cl, "There is no initialisation codelet for the reduction of the handle %p. Maybe you forget to call starpu_data_set_reduction_methods() ?", handle->root_handle);

	_starpu_cl_func_t init_func = NULL;

	/* TODO Check that worker may execute the codelet */

	switch (starpu_worker_get_type(workerid))
	{
		case STARPU_CPU_WORKER:
			init_func = _starpu_task_get_cpu_nth_implementation(init_cl, 0);
			break;

		case STARPU_CUDA_WORKER:
			init_func = _starpu_task_get_cuda_nth_implementation(init_cl, 0);
#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
			/* We make sure we do manipulate the proper device */
			starpu_cuda_set_device(starpu_worker_get_devid(workerid));
#endif
			break;
		case STARPU_HIP_WORKER:
			init_func = _starpu_task_get_hip_nth_implementation(init_cl, 0);
#if defined(STARPU_HAVE_HIP_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
			/* We make sure we do manipulate the proper device */
			starpu_hip_set_device(starpu_worker_get_devid(workerid));
#endif
			break;
		case STARPU_OPENCL_WORKER:
			init_func = _starpu_task_get_opencl_nth_implementation(init_cl, 0);
			break;

#ifdef STARPU_USE_MPI_MASTER_SLAVE
		case STARPU_MPI_MS_WORKER:
			init_func = _starpu_src_common_get_cpu_func_from_codelet(init_cl, 0);
			break;
#endif

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		case STARPU_TCPIP_MS_WORKER:
			init_func = _starpu_src_common_get_cpu_func_from_codelet(init_cl, 0);
			break;
#endif

		default:
			STARPU_ABORT();
			break;
	}

	STARPU_ASSERT(init_func);

	switch (starpu_worker_get_type(workerid))
	{
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		case STARPU_MPI_MS_WORKER:
		{
			struct _starpu_mp_node *node = _starpu_mpi_ms_src_get_actual_thread_mp_node();
			int subworkerid = _starpu_get_worker_struct(workerid)->subworkerid;
			void * arg;
			int arg_size;

			_starpu_src_common_execute_kernel(node,
					(void(*)(void))init_func, subworkerid,
					STARPU_SEQ, 0, 0, &handle,
					&(replicate->data_interface), 1,
					NULL, 0 , 1);

			_starpu_src_common_wait_completed_execution(node,subworkerid,&arg,&arg_size);
			break;
		}
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		case STARPU_TCPIP_MS_WORKER:
		{
			struct _starpu_mp_node *node = _starpu_tcpip_ms_src_get_actual_thread_mp_node();
			int subworkerid = _starpu_get_worker_struct(workerid)->subworkerid;
			void * arg;
			int arg_size;

			_starpu_src_common_execute_kernel(node,
					(void(*)(void))init_func, subworkerid,
					STARPU_SEQ, 0, 0, &handle,
					&(replicate->data_interface), 1,
					NULL, 0 , 1);

			_starpu_src_common_wait_completed_execution(node,subworkerid,&arg,&arg_size);
			break;
		}
#endif
		default:
			init_func(&replicate->data_interface, NULL);
			break;
	}

	replicate->initialized = 1;
}

/* Enable reduction mode. This function must be called with the header lock
 * taken. */
void _starpu_data_start_reduction_mode(starpu_data_handle_t handle)
{
	STARPU_ASSERT(handle->reduction_refcnt == 0);

	if (!handle->per_worker)
		_starpu_data_initialize_per_worker(handle);

	unsigned worker;

	unsigned nworkers = starpu_worker_get_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_data_replicate *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;
		replicate->relaxed_coherency = 2;
		if (replicate->mc)
			replicate->mc->relaxed_coherency = 2;
	}
}

//#define NO_TREE_REDUCTION

/* Force reduction. The lock should already have been taken.  */
void _starpu_data_end_reduction_mode(starpu_data_handle_t handle, int priority)
{
	unsigned worker;
	unsigned node;
	unsigned empty; /* Whether the handle is initially unallocated */

	/* Put every valid replicate in the same array */
	unsigned replicate_count = 0;
	starpu_data_handle_t replicate_array[1 + STARPU_NMAXWORKERS];

	_starpu_spin_checklocked(&handle->header_lock);

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (handle->per_node[node].state != STARPU_INVALID)
			break;
	}
	empty = node == STARPU_MAXNODES;

#ifndef NO_TREE_REDUCTION
	if (!empty)
		/* Include the initial value into the reduction tree */
		replicate_array[replicate_count++] = handle;
#endif

	/* Register all valid per-worker replicates */
	unsigned nworkers = starpu_worker_get_count();
	STARPU_ASSERT(!handle->reduction_tmp_handles);
	_STARPU_MALLOC(handle->reduction_tmp_handles, nworkers*sizeof(handle->reduction_tmp_handles[0]));
	for (worker = 0; worker < nworkers; worker++)
	{
		if (handle->per_worker[worker].initialized)
		{
			/* Make sure the replicate is not removed */
			handle->per_worker[worker].refcnt++;

			unsigned home_node = starpu_worker_get_memory_node(worker);
			starpu_data_register(&handle->reduction_tmp_handles[worker],
				home_node, handle->per_worker[worker].data_interface, handle->ops);

			starpu_data_set_sequential_consistency_flag(handle->reduction_tmp_handles[worker], 0);

			replicate_array[replicate_count++] = handle->reduction_tmp_handles[worker];
		}
		else
		{
			handle->reduction_tmp_handles[worker] = NULL;
		}
	}

#ifndef NO_TREE_REDUCTION
	if (empty)
	{
		/* Only the final copy will touch the actual handle */
		handle->reduction_refcnt = 1;
	}
	else
	{
		unsigned step = 1;
		handle->reduction_refcnt = 0;
		while (step < replicate_count)
		{
			/* Each stage will touch the actual handle */
			handle->reduction_refcnt++;
			step *= 2;
		}
	}
#else
	/* We know that in this reduction algorithm there is exactly one task per valid replicate. */
	handle->reduction_refcnt = replicate_count + empty;
#endif

//	fprintf(stderr, "REDUX REFCNT = %d\n", handle->reduction_refcnt);

	if (replicate_count >
#ifndef NO_TREE_REDUCTION
			!empty
#else
			0
#endif
			)
	{
		/* Temporarily unlock the handle */
		_starpu_spin_unlock(&handle->header_lock);

#ifndef NO_TREE_REDUCTION
		/* We will store a pointer to the last task which should modify the
		 * replicate */
		struct starpu_task *last_replicate_deps[replicate_count];
		memset(last_replicate_deps, 0, replicate_count*sizeof(struct starpu_task *));
		struct starpu_task *redux_tasks[replicate_count];

		/* Redux step-by-step for step from 1 to replicate_count/2, i.e.
		 * 1-by-1, then 2-by-2, then 4-by-4, etc. */
		unsigned step;
		unsigned redux_task_idx = 0;
		for (step = 1; step < replicate_count; step *=2)
		{
			unsigned i;
			for (i = 0; i < replicate_count; i+=2*step)
			{
				if (i + step < replicate_count)
				{
					/* Perform the reduction between replicates i
					 * and i+step and put the result in replicate i */
					struct starpu_task *redux_task = starpu_task_create();
					redux_task->name = "redux_task_between_replicates";
					redux_task->priority = priority;

					/* Mark these tasks so that StarPU does not block them
					 * when they try to access the handle (normal tasks are
					 * data requests to that handle are frozen until the
					 * data is coherent again). */
					struct _starpu_job *j = _starpu_get_job_associated_to_task(redux_task);
					j->reduction_task = 1;

					redux_task->cl = handle->redux_cl;
					redux_task->cl_arg = handle->redux_cl_arg;
					STARPU_ASSERT(redux_task->cl);
					if (!(STARPU_CODELET_GET_MODE(redux_task->cl, 0)))
						STARPU_CODELET_SET_MODE(redux_task->cl, STARPU_RW|STARPU_COMMUTE, 0);
					if (!(STARPU_CODELET_GET_MODE(redux_task->cl, 1)))
						STARPU_CODELET_SET_MODE(redux_task->cl, STARPU_R, 1);

					if (!(STARPU_CODELET_GET_MODE(redux_task->cl, 0) & STARPU_COMMUTE))
					{
						static int warned;
						STARPU_HG_DISABLE_CHECKING(warned);
						if (!warned)
						{
							warned = 1;
							_STARPU_DISP("Warning: for reductions, codelet %p should have STARPU_COMMUTE along STARPU_RW\n", redux_task->cl);
						}
					}

					STARPU_TASK_SET_HANDLE(redux_task, replicate_array[i], 0);
					STARPU_TASK_SET_HANDLE(redux_task, replicate_array[i+step], 1);

					int ndeps = 0;
					struct starpu_task *task_deps[2];

					if (last_replicate_deps[i])
						task_deps[ndeps++] = last_replicate_deps[i];

					if (last_replicate_deps[i+step])
						task_deps[ndeps++] = last_replicate_deps[i+step];

					/* i depends on this task */
					last_replicate_deps[i] = redux_task;

					/* we don't perform the reduction until both replicates are ready */
					starpu_task_declare_deps_array(redux_task, ndeps, task_deps);

					/* We cannot submit tasks here : we do
					 * not want to depend on tasks that have
					 * been completed, so we juste store
					 * this task : it will be submitted
					 * later. */
					redux_tasks[redux_task_idx++] = redux_task;
				}
			}
		}

		if (empty)
			/* The handle was empty, we just need to copy the reduced value. */
			_starpu_data_cpy(handle, replicate_array[0], 1, NULL, 0, 1, last_replicate_deps[0], priority);

		/* Let's submit all the reduction tasks. */
		unsigned i;
		for (i = 0; i < redux_task_idx; i++)
		{
			int ret = _starpu_task_submit_internally(redux_tasks[i]);
			STARPU_ASSERT(ret == 0);
		}
#else
		if (empty)
		{
			struct starpu_task *redux_task = starpu_task_create();
			redux_task->name = "redux_task_empty";
			redux_task->priority = priority;

			/* Mark these tasks so that StarPU does not block them
			 * when they try to access the handle (normal tasks are
			 * data requests to that handle are frozen until the
			 * data is coherent again). */
			struct _starpu_job *j = _starpu_get_job_associated_to_task(redux_task);
			j->reduction_task = 1;

			redux_task->cl = handle->init_cl;
			redux_task->cl_arg = handle->init_cl_arg;
			STARPU_ASSERT(redux_task->cl);

			if (!(STARPU_CODELET_GET_MODE(redux_task->cl, 0)))
				STARPU_CODELET_SET_MODE(redux_task->cl, STARPU_W, 0);

			STARPU_TASK_SET_HANDLE(redux_task, handle, 0);

			int ret = _starpu_task_submit_internally(redux_task);
			STARPU_ASSERT(!ret);
		}

		/* Create a set of tasks to perform the reduction */
		unsigned replicate;
		for (replicate = 0; replicate < replicate_count; replicate++)
		{
			struct starpu_task *redux_task = starpu_task_create();
			redux_task->name = "redux_task_reduction";
			redux_task->priority = priority;

			/* Mark these tasks so that StarPU does not block them
			 * when they try to access the handle (normal tasks are
			 * data requests to that handle are frozen until the
			 * data is coherent again). */
			struct _starpu_job *j = _starpu_get_job_associated_to_task(redux_task);
			j->reduction_task = 1;

			redux_task->cl = handle->redux_cl;
			STARPU_ASSERT(redux_task->cl);

			if (!(STARPU_CODELET_GET_MODE(redux_task->cl, 0)))
				STARPU_CODELET_SET_MODE(redux_task->cl, STARPU_RW, 0);
			if (!(STARPU_CODELET_GET_MODE(redux_task->cl, 1)))
				STARPU_CODELET_SET_MODE(redux_task->cl, STARPU_R, 1);

			STARPU_TASK_SET_HANDLE(redux_task, handle, 0);
			STARPU_TASK_SET_HANDLE(redux_task, replicate_array[replicate], 1);

			int ret = _starpu_task_submit_internally(redux_task);
			STARPU_ASSERT(!ret);
		}
#endif
		/* Get the header lock back */
		_starpu_spin_lock(&handle->header_lock);

	}

	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_data_replicate *replicate;
		replicate = &handle->per_worker[worker];
		replicate->relaxed_coherency = 1;
		if (replicate->mc)
			replicate->mc->relaxed_coherency = 1;
	}
}

void _starpu_data_end_reduction_mode_terminate(starpu_data_handle_t handle)
{
	unsigned nworkers = starpu_worker_get_count();

//	fprintf(stderr, "_starpu_data_end_reduction_mode_terminate\n");
	unsigned worker;

	_starpu_spin_checklocked(&handle->header_lock);

	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_data_replicate *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;

		if (handle->reduction_tmp_handles[worker])
		{
//			fprintf(stderr, "unregister handle %p\n", handle);
			_starpu_spin_lock(&handle->reduction_tmp_handles[worker]->header_lock);
			handle->reduction_tmp_handles[worker]->lazy_unregister = 1;
			_starpu_spin_unlock(&handle->reduction_tmp_handles[worker]->header_lock);
			starpu_data_unregister_no_coherency(handle->reduction_tmp_handles[worker]);
			handle->per_worker[worker].refcnt--;
			/* TODO put in cache */
		}
	}
	free(handle->reduction_tmp_handles);
	handle->reduction_tmp_handles = NULL;
}
