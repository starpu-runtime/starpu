/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <datawizard/memory_nodes.h>

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

	/* Driver porters: adding your driver here is very optional, any_to_any will be enough.  */

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
#ifdef STARPU_USE_HIP
		case STARPU_HIP_WORKER:
		{
			hipStream_t stream = starpu_hip_get_local_stream();
			if (copy_methods->hip_to_hip_async)
			{
				copy_methods->hip_to_hip_async(src_interface, memory_node, dst_interface, memory_node, stream);
				return;
			}
			else if (copy_methods->hip_to_hip)
			{
				copy_methods->hip_to_hip(src_interface, memory_node, dst_interface, memory_node);
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

static struct starpu_perfmodel copy_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "starpu_data_cpy"
};

static struct starpu_codelet copy_cl =
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_HIP|STARPU_OPENCL,
	.cpu_funcs = {common_data_cpy_func},
	.cuda_funcs = {common_data_cpy_func},
	.opencl_funcs = {common_data_cpy_func},
	.hip_funcs = {common_data_cpy_func},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_R},
	.model = &copy_model
};

int _starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
		     int asynchronous, void (*callback_func)(void*), void *callback_arg,
		     int reduction, struct starpu_task *reduction_dep_task, int priority)
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
	task->priority = priority;
	task->callback_func = callback_func;
	task->callback_arg = callback_arg;

	/* FIXME: priority!! */
	STARPU_TASK_SET_HANDLE(task, dst_handle, 0);
	STARPU_TASK_SET_HANDLE(task, src_handle, 1);

	task->synchronous = !asynchronous;

	int ret = _starpu_task_submit_internally(task);
	STARPU_ASSERT_MSG(ret != -ENODEV, "Implementation of _starpu_data_cpy is needed for this only available architecture\n");
	STARPU_ASSERT_MSG(!ret, "Task data copy failed with code: %d\n", ret);

	return 0;
}

int starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
		    int asynchronous, void (*callback_func)(void*), void *callback_arg)
{
	return _starpu_data_cpy(dst_handle, src_handle, asynchronous, callback_func, callback_arg, 0, NULL, STARPU_DEFAULT_PRIO);
}

int starpu_data_cpy_priority(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
			     int asynchronous, void (*callback_func)(void*), void *callback_arg, int priority)
{
	return _starpu_data_cpy(dst_handle, src_handle, asynchronous, callback_func, callback_arg, 0, NULL, priority);
}

/* TODO: implement copy on write, and introduce starpu_data_dup as well */
int starpu_data_dup_ro(starpu_data_handle_t *dst_handle, starpu_data_handle_t src_handle, int asynchronous)
{
	_starpu_spin_lock(&src_handle->header_lock);
	if (src_handle->readonly_dup)
	{
		/* Already a ro duplicate, just return it with one more ref */
		*dst_handle = src_handle->readonly_dup;
		_starpu_spin_unlock(&src_handle->header_lock);
		_starpu_spin_lock(&(*dst_handle)->header_lock);
		(*dst_handle)->aliases++;
		_starpu_spin_unlock(&(*dst_handle)->header_lock);
		return 0;
	}
	if (src_handle->readonly)
	{
		src_handle->aliases++;
		_starpu_spin_unlock(&src_handle->header_lock);
		*dst_handle = src_handle;
		return 0;
	}
	_starpu_spin_unlock(&src_handle->header_lock);

	starpu_data_register_same(dst_handle, src_handle);
	_starpu_data_cpy(*dst_handle, src_handle, asynchronous, NULL, NULL, 0, NULL, STARPU_DEFAULT_PRIO);
	(*dst_handle)->readonly = 1;

	_starpu_spin_lock(&src_handle->header_lock);
	src_handle->readonly_dup = (*dst_handle);
	(*dst_handle)->readonly_dup_of = src_handle;
	_starpu_spin_unlock(&src_handle->header_lock);

	return 0;
}
