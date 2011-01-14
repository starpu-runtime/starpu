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

#include <util/starpu_insert_task_utils.h>
#include <common/config.h>
#include <common/utils.h>

size_t starpu_insert_task_get_arg_size(va_list varg_list)
{
	int arg_type;
        size_t arg_buffer_size;

        arg_buffer_size = 0;

	arg_buffer_size += sizeof(char);

	while ((arg_type = va_arg(varg_list, int)) != 0) {
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
        return arg_buffer_size;
}

int starpu_insert_task_create_and_submit(size_t arg_buffer_size, starpu_codelet *cl, struct starpu_task **task, va_list varg_list) {
        int arg_type;
	unsigned current_buffer = 0;
	unsigned char nargs = 0;
	char *arg_buffer = malloc(arg_buffer_size);
	unsigned current_arg_offset = 0;

	/* We will begin the buffer with the number of args (which is stored as a char) */
	current_arg_offset += sizeof(char);

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH)
		{
			/* We have an access mode : we expect to find a handle */
			starpu_data_handle handle = va_arg(varg_list, starpu_data_handle);

			starpu_access_mode mode = arg_type;

			(*task)->buffers[current_buffer].handle = handle;
			(*task)->buffers[current_buffer].mode = mode;

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
			(*task)->callback_func = callback_func;
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			void *callback_arg;
			callback_arg = va_arg(varg_list, void *);
			(*task)->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			/* Followed by a priority level */
			int prio = va_arg(varg_list, int); 
			(*task)->priority = prio;
		}
	}

	va_end(varg_list);

	arg_buffer[0] = nargs;

	STARPU_ASSERT(current_buffer == cl->nbuffers);

	(*task)->cl = cl;
	(*task)->cl_arg = arg_buffer;

	int ret = starpu_task_submit(*task);

	if (STARPU_UNLIKELY(ret == -ENODEV))
          fprintf(stderr, "No one can execute task %p wih cl %p (symbol %s)\n", *task, (*task)->cl, ((*task)->cl->model && (*task)->cl->model->symbol)?(*task)->cl->model->symbol:"none");

	STARPU_ASSERT(!ret);
        return ret;
}
