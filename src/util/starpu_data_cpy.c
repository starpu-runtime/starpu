/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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
#include <common/config.h>
#include <datawizard/datawizard.h>

static void data_cpy_func(void *descr[], void *cl_arg)
{
	const struct starpu_data_copy_methods *copy_methods = (const struct starpu_data_copy_methods *) cl_arg;

	int workerid = starpu_worker_get_id();
	enum starpu_archtype type = starpu_worker_get_type(workerid);
	unsigned memory_node = starpu_worker_get_memory_node(workerid);

	void *dst_interface = descr[0];
	void *src_interface = descr[1];

	switch (type) {
		case STARPU_CPU_WORKER:
			STARPU_ASSERT(copy_methods->ram_to_ram);
			copy_methods->ram_to_ram(src_interface, memory_node, dst_interface, memory_node);
			break;
		case STARPU_CUDA_WORKER:
			STARPU_ASSERT(copy_methods->cuda_to_cuda);
			copy_methods->cuda_to_cuda(src_interface, memory_node, dst_interface, memory_node);
			break;
		case STARPU_OPENCL_WORKER:
			STARPU_ASSERT(copy_methods->opencl_to_opencl);
			copy_methods->opencl_to_opencl(src_interface, memory_node, dst_interface, memory_node);
			break;
		default:
			/* unknown architecture */
			STARPU_ABORT();
	}

}

struct starpu_perfmodel copy_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "starpu_data_cpy"
};

static struct starpu_codelet copy_cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cpu_func = data_cpy_func,
	.cuda_func = data_cpy_func,
	.opencl_func = data_cpy_func,
	.nbuffers = 2,
	.model = &copy_model
};

int starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
			int asynchronous, void (*callback_func)(void*), void *callback_arg)
{
	const struct starpu_data_copy_methods *copy_methods = dst_handle->ops->copy_methods;

	struct starpu_task *task = starpu_task_create();
	STARPU_ASSERT(task);

	task->cl = &copy_cl;
	task->cl_arg = (void *)copy_methods;

	task->callback_func = callback_func;
	task->callback_arg = callback_arg;

	task->buffers[0].handle = dst_handle;
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = src_handle;
	task->buffers[1].mode = STARPU_R;

	task->synchronous = !asynchronous;

	int ret = starpu_task_submit(task);
	STARPU_ASSERT(!ret);

	return 0;
}
