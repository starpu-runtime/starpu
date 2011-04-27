/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
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

#include <util/starpu_insert_task_utils.h>
#include <common/config.h>
#include <common/utils.h>

/* Deal with callbacks. The unpack function may be called multiple times when
 * we have a parallel task, and we should not free the cl_arg parameter from
 * the callback function. */
struct insert_task_cb_wrapper {
	void (*callback_func)(void *);
	void *callback_arg;
	void *arg_stack;
};

static
void starpu_task_insert_callback_wrapper(void *_cl_arg_wrapper)
{
	struct insert_task_cb_wrapper *cl_arg_wrapper = _cl_arg_wrapper;

	/* Execute the callback specified by the application */
	if (cl_arg_wrapper->callback_func)
		cl_arg_wrapper->callback_func(cl_arg_wrapper->callback_arg);

	/* Free the stack of arguments */
	free(cl_arg_wrapper->arg_stack);
}

size_t _starpu_insert_task_get_arg_size(va_list varg_list)
{
	int arg_type;
        size_t arg_buffer_size;

        arg_buffer_size = 0;

	arg_buffer_size += sizeof(char);

	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH || arg_type == STARPU_REDUX) {
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
		else if (arg_type==STARPU_EXECUTE) {
			va_arg(varg_list, int);
		}
	}

	va_end(varg_list);
        return arg_buffer_size;
}

int _starpu_pack_cl_args(size_t arg_buffer_size, char **arg_buffer, va_list varg_list)
{
        int arg_type;
	unsigned current_arg_offset = 0;
	unsigned char nargs = 0;

	/* The buffer will contain : nargs, {size, content} (x nargs)*/

	*arg_buffer = malloc(arg_buffer_size);

	/* We will begin the buffer with the number of args (which is stored as a char) */
	current_arg_offset += sizeof(char);

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH || arg_type == STARPU_REDUX)
		{
			va_arg(varg_list, starpu_data_handle);
		}
		else if (arg_type==STARPU_VALUE)
		{
			/* We have a constant value: this should be followed by a pointer to the cst value and the size of the constant */
			void *ptr = va_arg(varg_list, void *);
			size_t cst_size = va_arg(varg_list, size_t);

			*(size_t *)(&(*arg_buffer)[current_arg_offset]) = cst_size;
			current_arg_offset += sizeof(size_t);

			memcpy(&(*arg_buffer)[current_arg_offset], ptr, cst_size);
			current_arg_offset += cst_size;

			nargs++;
			STARPU_ASSERT(current_arg_offset <= arg_buffer_size);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE) {
			va_arg(varg_list, int);
		}
	}

	(*arg_buffer)[0] = nargs;
	va_end(varg_list);
	return 0;
}

int _starpu_insert_task_create_and_submit(char *arg_buffer, starpu_codelet *cl, struct starpu_task **task, va_list varg_list) {
        int arg_type;
	unsigned current_buffer = 0;

	struct insert_task_cb_wrapper *cl_arg_wrapper = malloc(sizeof(struct insert_task_cb_wrapper));
	STARPU_ASSERT(cl_arg_wrapper);

	cl_arg_wrapper->callback_func = NULL;
	cl_arg_wrapper->arg_stack = arg_buffer;

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH || arg_type == STARPU_REDUX)
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
			va_arg(varg_list, void *);
			va_arg(varg_list, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			void (*callback_func)(void *);
			callback_func = va_arg(varg_list, void (*)(void *));
			cl_arg_wrapper->callback_func = callback_func;
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			void *callback_arg = va_arg(varg_list, void *);
			cl_arg_wrapper->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			/* Followed by a priority level */
			int prio = va_arg(varg_list, int); 
			(*task)->priority = prio;
		}
		else if (arg_type==STARPU_EXECUTE) {
			va_arg(varg_list, int);
		}
	}

	va_end(varg_list);

	STARPU_ASSERT(current_buffer == cl->nbuffers);

	(*task)->cl = cl;
	(*task)->cl_arg = arg_buffer;

	/* The callback will free the argument stack and execute the
	 * application's callback, if any. */
	(*task)->callback_func = starpu_task_insert_callback_wrapper;
	(*task)->callback_arg = cl_arg_wrapper;

	int ret = starpu_task_submit(*task);

	if (STARPU_UNLIKELY(ret == -ENODEV))
          fprintf(stderr, "submission of task %p wih codelet %p failed (symbol `%s')\n", *task, (*task)->cl, ((*task)->cl->model && (*task)->cl->model->symbol)?(*task)->cl->model->symbol:"none");

        return ret;
}
