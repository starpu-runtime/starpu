/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#include <core/task.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>
#include <util/starpu_data_cpy.h>
#include <starpu_mic.h>

static void common_data_cpy_func(void *descr[], void *cl_arg)
{
	unsigned interface_id = *(unsigned *)cl_arg;

	const struct starpu_data_interface_ops *interface_ops = _starpu_data_interface_get_ops(interface_id);
	const struct starpu_data_copy_methods *copy_methods = interface_ops->copy_methods;

	int workerid = starpu_worker_get_id_check();
	enum starpu_worker_archtype type = starpu_worker_get_type(workerid);
	unsigned memory_node = starpu_worker_get_memory_node(workerid);

	void *dst_interface = descr[0];
	void *src_interface = descr[1];

	switch (type)
	{
		case STARPU_CPU_WORKER:
			if (copy_methods->ram_to_ram)
			{
				copy_methods->ram_to_ram(src_interface, memory_node, dst_interface, memory_node);
				return;
			}
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_WORKER:
		{
			cudaStream_t stream = starpu_cuda_get_local_stream();
			if (copy_methods->cuda_to_cuda_async)
			{
				copy_methods->cuda_to_cuda_async(src_interface, memory_node, dst_interface, memory_node, stream);
				return;
			}
			else if (copy_methods->cuda_to_cuda)
			{
				copy_methods->cuda_to_cuda(src_interface, memory_node, dst_interface, memory_node);
				return;
			}
			break;
		}
#endif
		case STARPU_OPENCL_WORKER:
			if (copy_methods->opencl_to_opencl)
			{
				copy_methods->opencl_to_opencl(src_interface, memory_node, dst_interface, memory_node);
				return;
			}
			break;
		default:
			/* unknown architecture */
			STARPU_ABORT();
	}
	STARPU_ASSERT(copy_methods->any_to_any);
	copy_methods->any_to_any(src_interface, memory_node, dst_interface, memory_node, NULL);

}

void mp_cpy_kernel(void *descr[], void *cl_arg)
{
	unsigned interface_id = *(unsigned *)cl_arg;

	const struct starpu_data_interface_ops *interface_ops = _starpu_data_interface_get_ops(interface_id);
	const struct starpu_data_copy_methods *copy_methods = interface_ops->copy_methods;
	
	void *dst_interface = descr[0];
	void *src_interface = descr[1];

	if(copy_methods->ram_to_ram)
		copy_methods->ram_to_ram(src_interface, STARPU_MAIN_RAM, dst_interface, STARPU_MAIN_RAM);
	else if(copy_methods->any_to_any)
		copy_methods->any_to_any(src_interface, STARPU_MAIN_RAM, dst_interface, STARPU_MAIN_RAM, NULL);
	else
		STARPU_ABORT();

}

static starpu_mic_kernel_t mic_cpy_func()
{
#ifdef STARPU_USE_MIC
	starpu_mic_func_symbol_t mic_symbol = NULL;
	starpu_mic_register_kernel(&mic_symbol, "mp_cpy_kernel");

	return starpu_mic_get_kernel(mic_symbol);
#else
	STARPU_ABORT();
	return NULL;
#endif
}

struct starpu_perfmodel copy_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "starpu_data_cpy"
};

static struct starpu_codelet copy_cl =
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL|STARPU_MIC,
	.cpu_funcs = {common_data_cpy_func},
	.cuda_funcs = {common_data_cpy_func},
	.opencl_funcs = {common_data_cpy_func},
	.mic_funcs = {mic_cpy_func},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_R},
	.model = &copy_model
};

int _starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
		     int asynchronous, void (*callback_func)(void*), void *callback_arg,
		     int reduction, struct starpu_task *reduction_dep_task)
{

	struct starpu_task *task = starpu_task_create();
	STARPU_ASSERT(task);
	task->name = "data_cpy";

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	if (reduction)
	{
		j->reduction_task = reduction;
		if (reduction_dep_task)
			starpu_task_declare_deps_array(task, 1, &reduction_dep_task);
	}

	task->cl = &copy_cl;

	unsigned *interface_id;
	_STARPU_MALLOC(interface_id, sizeof(*interface_id));
	*interface_id = dst_handle->ops->interfaceid; 
	task->cl_arg = interface_id;
	task->cl_arg_size = sizeof(*interface_id);
	task->cl_arg_free = 1;

	task->callback_func = callback_func;
	task->callback_arg = callback_arg;

	/* FIXME: priority!! */
	STARPU_TASK_SET_HANDLE(task, dst_handle, 0);
	STARPU_TASK_SET_HANDLE(task, src_handle, 1);

	task->synchronous = !asynchronous;

	int ret = _starpu_task_submit_internally(task);
	STARPU_ASSERT(!ret);

	return 0;
}

int starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
			int asynchronous, void (*callback_func)(void*), void *callback_arg)
{
	return _starpu_data_cpy(dst_handle, src_handle, asynchronous, callback_func, callback_arg, 0, NULL);
}
