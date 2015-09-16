/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012, 2014-2015  Université de Bordeaux
 * Copyright (C) 2011, 2012, 2013, 2014, 2015  CNRS
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

/* This file provides an interface that is very similar to that of the Quark
 * scheduler from the PLASMA project (see http://icl.cs.utk.edu/plasma/). */

#include <starpu.h>
#include <common/config.h>
#include <stdarg.h>
#include <util/starpu_task_insert_utils.h>

void starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, ...)
{
	va_list varg_list;

	va_start(varg_list, arg_buffer_size);
	_starpu_codelet_pack_args(arg_buffer, arg_buffer_size, varg_list);
	va_end(varg_list);
}

void starpu_codelet_unpack_args(void *_cl_arg, ...)
{
	char *cl_arg = (char *) _cl_arg;
	int current_arg_offset = 0;
	int nargs, arg;
	va_list varg_list;

	STARPU_ASSERT(cl_arg);
	va_start(varg_list, _cl_arg);

	/* We fill the different pointers with the appropriate arguments */
	memcpy(&nargs, cl_arg, sizeof(nargs));
	current_arg_offset += sizeof(nargs);

	for (arg = 0; arg < nargs; arg++)
	{
		void *argptr = va_arg(varg_list, void *);

		size_t arg_size;
		memcpy(&arg_size, cl_arg+current_arg_offset, sizeof(arg_size));
		current_arg_offset += sizeof(arg_size);

		memcpy(argptr, cl_arg+current_arg_offset, arg_size);
		current_arg_offset += arg_size;
	}

	va_end(varg_list);
}

static
struct starpu_task *_starpu_task_build_v(struct starpu_codelet *cl, const char* task_name, int cl_arg_free, va_list varg_list)
{
	va_list varg_list_copy;

	struct starpu_task *task = starpu_task_create();
	task->name = task_name;
	task->cl_arg_free = cl_arg_free;

	va_copy(varg_list_copy, varg_list);
	_starpu_task_insert_create(cl, &task, varg_list_copy);
	va_end(varg_list_copy);

	return task;
}

static
int _starpu_task_insert_v(struct starpu_codelet *cl, va_list varg_list)
{
	struct starpu_task *task;
	int ret;

	task = _starpu_task_build_v(cl, NULL, 1, varg_list);
	ret = starpu_task_submit(task);

	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		fprintf(stderr, "submission of task %p wih codelet %p failed (symbol `%s') (err: ENODEV)\n",
			task, task->cl,
			(cl == NULL) ? "none" :
			task->cl->name ? task->cl->name :
			(task->cl->model && task->cl->model->symbol)?task->cl->model->symbol:"none");

		task->destroy = 0;
		starpu_task_destroy(task);
	}
	return ret;
}

int starpu_task_insert(struct starpu_codelet *cl, ...)
{
	va_list varg_list;
	int ret;

	va_start(varg_list, cl);
	ret = _starpu_task_insert_v(cl, varg_list);
	va_end(varg_list);
	return ret;
}

int starpu_insert_task(struct starpu_codelet *cl, ...)
{
	va_list varg_list;
	int ret;

	va_start(varg_list, cl);
	ret = _starpu_task_insert_v(cl, varg_list);
	va_end(varg_list);
	return ret;
}

struct starpu_task *starpu_task_build(struct starpu_codelet *cl, ...)
{
	struct starpu_task *task;
	va_list varg_list;

	va_start(varg_list, cl);
	task = _starpu_task_build_v(cl, "task_build", 0, varg_list);
	if (task->cl_arg)
	{
		task->cl_arg_free = 1;
}
	va_end(varg_list);

	return task;
}
