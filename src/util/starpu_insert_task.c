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

/* This file provides an interface that is very similar to that of the Quark
 * scheduler from the PLASMA project (see http://icl.cs.utk.edu/plasma/). */

#include <starpu.h>
#include <common/config.h>
#include <stdarg.h>

void starpu_unpack_cl_args(void *_cl_arg, ...)
{
	unsigned char *cl_arg = _cl_arg;

	unsigned current_arg_offset = 0;
	va_list varg_list;

	va_start(varg_list, _cl_arg);

	/* We fill the different pointers with the appropriate arguments */
	unsigned char nargs = cl_arg[0];
	current_arg_offset += sizeof(char);

	unsigned arg;
	for (arg = 0; arg < nargs; arg++)
	{
		void *argptr = va_arg(varg_list, void *);
		size_t arg_size = *(size_t *)&cl_arg[current_arg_offset];
		current_arg_offset += sizeof(size_t);

		memcpy(argptr, &cl_arg[current_arg_offset], arg_size); 
		current_arg_offset += arg_size;
	}

	va_end(varg_list);

	/* XXX this should not really be done in StarPU but well .... */
	free(_cl_arg);
}

void starpu_insert_task(starpu_codelet *cl, ...)
{
	struct starpu_task *task = starpu_task_create();
	int arg_type;
	va_list varg_list;

	/* The buffer will contain : nargs, {size, content} (x nargs)*/

	/* Compute the size */
	size_t arg_buffer_size = 0;

	arg_buffer_size += sizeof(char);

	va_start(varg_list, cl);

	while( (arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
			va_arg(varg_list, starpu_data_handle);
		}
		else if (arg_type==STARPU_VALUE) {
			va_arg(varg_list, void *);
			size_t cst_size = va_arg(varg_list, size_t);

			arg_buffer_size += sizeof(size_t);
			arg_buffer_size += cst_size;
		}
		else if (arg_type==STARPU_CALLBACK) {
			va_arg(varg_list, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY) {
			va_arg(varg_list, int);
		}
	}

	va_end(varg_list);

	char *arg_buffer = malloc(arg_buffer_size);
	unsigned current_arg_offset = 0;

	/* We will begin the buffer with the number of args (which is stored as a char) */
	current_arg_offset += sizeof(char);
	unsigned current_buffer = 0;
	unsigned char nargs = 0;

	va_start(varg_list, cl);

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH)
		{
			/* We have an access mode : we expect to find a handle */
			starpu_data_handle handle = va_arg(varg_list, starpu_data_handle);

			starpu_access_mode mode = arg_type;

			task->buffers[current_buffer].handle = handle;
			task->buffers[current_buffer].mode = mode;

			current_buffer++;
		}
		else if (arg_type==STARPU_VALUE)
		{
			/* We have a constant value: this should be followed by a pointer to the cst value and the size of the constant */
			void *ptr = va_arg(varg_list, void *);
			size_t cst_size = va_arg(varg_list, size_t);

			*(size_t *)(&arg_buffer[current_arg_offset]) = cst_size;
			current_arg_offset += sizeof(size_t);

			memcpy(&arg_buffer[current_arg_offset], ptr, cst_size);
			current_arg_offset += cst_size;

			nargs++;
			STARPU_ASSERT(current_arg_offset <= arg_buffer_size);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			void (*callback_func)(void *);
			callback_func = va_arg(varg_list, void (*)(void *));
			task->callback_func = callback_func;
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			void *callback_arg;
			callback_arg = va_arg(varg_list, void *);
			task->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			/* Followed by a priority level */
			int prio = va_arg(varg_list, int); 
			task->priority = prio;
		}
	}

	va_end(varg_list);

	arg_buffer[0] = nargs;

	STARPU_ASSERT(current_buffer == cl->nbuffers);

	task->cl = cl;
	task->cl_arg = arg_buffer;

	int ret = starpu_task_submit(task);

	if (STARPU_UNLIKELY(ret == -ENODEV))
		fprintf(stderr, "No one can execute task %p wih cl %p (symbol %s)\n", task, task->cl, (task->cl->model && task->cl->model->symbol)?task->cl->model->symbol:"none");

	STARPU_ASSERT(!ret);
}
