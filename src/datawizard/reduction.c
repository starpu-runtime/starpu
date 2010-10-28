/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/utils.h>
#include <core/task.h>
#include <datawizard/datawizard.h>

void starpu_data_set_reduction_methods(starpu_data_handle handle,
					struct starpu_codelet_t *redux_cl,
					struct starpu_codelet_t *init_cl)
{
	_starpu_spin_lock(&handle->header_lock);

	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure that the flags are applied to the children as well */
		struct starpu_data_state_t *child_handle = &handle->children[child];
		if (child_handle->nchildren > 0)
			starpu_data_set_reduction_methods(child_handle, redux_cl, init_cl);
	}

	handle->redux_cl = redux_cl;
	handle->init_cl = init_cl;

	_starpu_spin_unlock(&handle->header_lock);
}

void _starpu_redux_init_data_replicate(starpu_data_handle handle, struct starpu_data_replicate_s *replicate, int workerid)
{
	STARPU_ASSERT(replicate);
	STARPU_ASSERT(replicate->allocated);

	struct starpu_codelet_t *init_cl = handle->init_cl;
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

	init_func(&replicate->interface, NULL);

	replicate->initialized = 1;
}

/* Enable reduction mode. This function must be called with the header lock
 * taken. */
void starpu_data_start_reduction_mode(starpu_data_handle handle)
{
	STARPU_ASSERT(handle->reduction_refcnt == 0);

	unsigned worker;

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;
	}
}

/* Force reduction. The lock should already have been taken.  */
void starpu_data_end_reduction_mode(starpu_data_handle handle)
{
	unsigned worker;

	handle->reduction_refcnt = 0;

	/* Register all valid per-worker replicates */
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (handle->per_worker[worker].initialized)
		{
			/* Make sure the replicate is not removed */
			handle->per_worker[worker].refcnt++;

			uint32_t home_node = starpu_worker_get_memory_node(worker); 
			starpu_data_register(&handle->reduction_tmp_handles[worker],
				home_node, handle->per_worker[worker].interface, handle->ops);

			/* We know that in this reduction algorithm there is exactly one task per valid replicate. */
			handle->reduction_refcnt++;
		}
		else {
			handle->reduction_tmp_handles[worker] = NULL;
		}
	}

//	fprintf(stderr, "REDUX REFCNT = %d\n", handle->reduction_refcnt);
	
	/* Temporarily unlock the handle */
	_starpu_spin_unlock(&handle->header_lock);

	/* Create a set of tasks to perform the reduction */
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (handle->reduction_tmp_handles[worker])
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

			redux_task->buffers[1].handle = handle->reduction_tmp_handles[worker];
			redux_task->buffers[1].mode = STARPU_R;

			int ret = starpu_task_submit(redux_task);
			STARPU_ASSERT(!ret);
		}
	}

	/* Get the header lock back */
	_starpu_spin_lock(&handle->header_lock);
}

void starpu_data_end_reduction_mode_terminate(starpu_data_handle handle)
{
//	fprintf(stderr, "starpu_data_end_reduction_mode_terminate\n");
	unsigned worker;
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
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
