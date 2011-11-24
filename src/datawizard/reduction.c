/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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
#include <core/task.h>
#include <datawizard/datawizard.h>

void starpu_data_set_reduction_methods(starpu_data_handle_t handle,
					struct starpu_codelet *redux_cl,
					struct starpu_codelet *init_cl)
{
	_starpu_spin_lock(&handle->header_lock);

	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure that the flags are applied to the children as well */
		struct _starpu_data_state *child_handle = &handle->children[child];
		if (child_handle->nchildren > 0)
			starpu_data_set_reduction_methods(child_handle, redux_cl, init_cl);
	}

	handle->redux_cl = redux_cl;
	handle->init_cl = init_cl;

	_starpu_spin_unlock(&handle->header_lock);
}

void _starpu_redux_init_data_replicate(starpu_data_handle_t handle, struct starpu_data_replicate_s *replicate, int workerid)
{
	STARPU_ASSERT(replicate);
	STARPU_ASSERT(replicate->allocated);

	struct starpu_codelet *init_cl = handle->init_cl;
	STARPU_ASSERT(init_cl);

	cl_func init_func = NULL;
	
	/* TODO Check that worker may execute the codelet */

	switch (starpu_worker_get_type(workerid)) {
		case STARPU_CPU_WORKER:
			init_func = init_cl->cpu_func;
			break;
		case STARPU_CUDA_WORKER:
			init_func = init_cl->cuda_func;
			break;
		case STARPU_OPENCL_WORKER:
			init_func = init_cl->opencl_func;
			break;
		default:
			STARPU_ABORT();
			break;
	}

	STARPU_ASSERT(init_func);

	init_func(&replicate->data_interface, NULL);

	replicate->initialized = 1;
}

/* Enable reduction mode. This function must be called with the header lock
 * taken. */
void starpu_data_start_reduction_mode(starpu_data_handle_t handle)
{
	STARPU_ASSERT(handle->reduction_refcnt == 0);

	unsigned worker;

	unsigned nworkers = starpu_worker_get_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;
	}
}

//#define NO_TREE_REDUCTION

/* Force reduction. The lock should already have been taken.  */
void starpu_data_end_reduction_mode(starpu_data_handle_t handle)
{
	unsigned worker;

	/* Put every valid replicate in the same array */
	unsigned replicate_count = 0;
	starpu_data_handle_t replicate_array[STARPU_NMAXWORKERS];

	/* Register all valid per-worker replicates */
	unsigned nworkers = starpu_worker_get_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		if (handle->per_worker[worker].initialized)
		{
			/* Make sure the replicate is not removed */
			handle->per_worker[worker].refcnt++;

			uint32_t home_node = starpu_worker_get_memory_node(worker); 
			starpu_data_register(&handle->reduction_tmp_handles[worker],
				home_node, handle->per_worker[worker].data_interface, handle->ops);

			starpu_data_set_sequential_consistency_flag(handle->reduction_tmp_handles[worker], 0);

			replicate_array[replicate_count++] = handle->reduction_tmp_handles[worker];
		}
		else {
			handle->reduction_tmp_handles[worker] = NULL;
		}
	}

#ifndef NO_TREE_REDUCTION
	handle->reduction_refcnt = 1;
#else
	/* We know that in this reduction algorithm there is exactly one task per valid replicate. */
	handle->reduction_refcnt = replicate_count;
#endif

//	fprintf(stderr, "REDUX REFCNT = %d\n", handle->reduction_refcnt);
	
	if (replicate_count > 0)
	{
		/* Temporarily unlock the handle */
		_starpu_spin_unlock(&handle->header_lock);

#ifndef NO_TREE_REDUCTION
		/* We will store a pointer to the last task which should modify the
		 * replicate */
		struct starpu_task *last_replicate_deps[replicate_count];
		memset(last_replicate_deps, 0, replicate_count*sizeof(struct starpu_task *));
	
		unsigned step = 1;
		while (step <= replicate_count)
		{
			unsigned i;
			for (i = 0; i < replicate_count; i+=2*step)
			{
				if (i + step < replicate_count)
				{
					/* Perform the reduction between replicates i
					 * and i+step and put the result in replicate i */
					struct starpu_task *redux_task = starpu_task_create();
		
					redux_task->cl = handle->redux_cl;
					STARPU_ASSERT(redux_task->cl);
		
					redux_task->buffers[0].handle = replicate_array[i];
					redux_task->buffers[0].mode = STARPU_RW;
		
					redux_task->buffers[1].handle = replicate_array[i+step];
					redux_task->buffers[1].mode = STARPU_R;
	
					redux_task->detach = 0;
	
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
		
					int ret = starpu_task_submit(redux_task);
					STARPU_ASSERT(!ret);
		
				}
			}

			step *= 2;
		}
	
		struct starpu_task *redux_task = starpu_task_create();

		/* Mark these tasks so that StarPU does not block them
		 * when they try to access the handle (normal tasks are
		 * data requests to that handle are frozen until the
		 * data is coherent again). */
		starpu_job_t j = _starpu_get_job_associated_to_task(redux_task);
		j->reduction_task = 1;

		redux_task->cl = handle->redux_cl;
		STARPU_ASSERT(redux_task->cl);

		redux_task->buffers[0].handle = handle;
		redux_task->buffers[0].mode = STARPU_RW;

		redux_task->buffers[1].handle = replicate_array[0];
		redux_task->buffers[1].mode = STARPU_R;

		if (last_replicate_deps[0])
			starpu_task_declare_deps_array(redux_task, 1, &last_replicate_deps[0]);

		int ret = starpu_task_submit(redux_task);
		STARPU_ASSERT(!ret);

#else
		/* Create a set of tasks to perform the reduction */
		unsigned replicate;
		for (replicate = 0; replicate < replicate_count; replicate++)
		{
			struct starpu_task *redux_task = starpu_task_create();
	
			/* Mark these tasks so that StarPU does not block them
			 * when they try to access the handle (normal tasks are
			 * data requests to that handle are frozen until the
			 * data is coherent again). */
			starpu_job_t j = _starpu_get_job_associated_to_task(redux_task);
			j->reduction_task = 1;
	
			redux_task->cl = handle->redux_cl;
			STARPU_ASSERT(redux_task->cl);
	
			redux_task->buffers[0].handle = handle;
			redux_task->buffers[0].mode = STARPU_RW;
	
			redux_task->buffers[1].handle = replicate_array[replicate];
			redux_task->buffers[1].mode = STARPU_R;
	
			int ret = starpu_task_submit(redux_task);
			STARPU_ASSERT(!ret);
		}
#endif
	/* Get the header lock back */
	_starpu_spin_lock(&handle->header_lock);

	}
}

void starpu_data_end_reduction_mode_terminate(starpu_data_handle_t handle)
{
	unsigned nworkers = starpu_worker_get_count();

//	fprintf(stderr, "starpu_data_end_reduction_mode_terminate\n");
	unsigned worker;
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;

		if (handle->reduction_tmp_handles[worker])
		{
//			fprintf(stderr, "unregister handle %p\n", handle);
			handle->reduction_tmp_handles[worker]->lazy_unregister = 1;
			starpu_data_unregister_no_coherency(handle->reduction_tmp_handles[worker]);
			handle->per_worker[worker].refcnt--;
			/* TODO put in cache */
		}
	}
}
