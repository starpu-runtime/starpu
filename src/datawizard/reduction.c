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

/* Enable reduction mode */
void starpu_data_start_reduction_mode(starpu_data_handle handle)
{
	unsigned worker;

	_starpu_spin_lock(&handle->header_lock);

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;
	}

	_starpu_spin_unlock(&handle->header_lock);
}

/* Force reduction */
void starpu_data_end_reduction_mode(starpu_data_handle handle)
{
	unsigned worker;

	_starpu_spin_lock(&handle->header_lock);

	/* Register all valid per-worker replicates */
	starpu_data_handle tmp_handles[STARPU_NMAXWORKERS];

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (handle->per_worker[worker].initialized)
		{
			/* Make sure the replicate is not removed */
			handle->per_worker[worker].refcnt++;

			uint32_t home_node = starpu_worker_get_memory_node(worker); 
			starpu_data_register(&tmp_handles[worker], home_node, handle->per_worker[worker].interface, handle->ops);
		}
		else {
			tmp_handles[worker] = NULL;
		}
	}
	
	_starpu_spin_unlock(&handle->header_lock);

	/* Create a set of tasks to perform the reduction */
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (tmp_handles[worker])
		{
			struct starpu_task *redux_task = starpu_task_create();

			redux_task->cl = handle->redux_cl;
			STARPU_ASSERT(redux_task->cl);

			redux_task->buffers[0].handle = handle;
			redux_task->buffers[0].mode = STARPU_RW;

			redux_task->buffers[1].handle = tmp_handles[worker];
			redux_task->buffers[1].mode = STARPU_R;

			int ret = starpu_task_submit(redux_task);
			STARPU_ASSERT(!ret);
		}
	}

	/* TODO have a better way to synchronize */
	starpu_task_wait_for_all();

	_starpu_spin_lock(&handle->header_lock);
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];
		replicate->initialized = 0;

		if (tmp_handles[worker])
		{
			starpu_data_unregister_no_coherency(tmp_handles[worker]);

			handle->per_worker[worker].refcnt--;
			/* TODO put in cache */
		}
	}
	_starpu_spin_unlock(&handle->header_lock);
}
