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

/* This file provides an interface that is very similar to that of the Quark
 * scheduler from the PLASMA project (see http://icl.cs.utk.edu/plasma/). */

#include <starpu.h>
#include <common/config.h>
#include <stdarg.h>
#include <util/starpu_insert_task_utils.h>

void starpu_pack_cl_args(va_list varg_list)
{
	va_list varg_list;

	/* TODO use a single malloc to allocate the memory for arg_buffer and
	 * the callback argument wrapper */
	char *arg_buffer = malloc(arg_buffer_size);
	STARPU_ASSERT(arg_buffer);
	unsigned current_arg_offset = 0;

	/* We will begin the buffer with the number of args (which is stored as a char) */
	current_arg_offset += sizeof(char);

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type==STARPU_VALUE)
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
		else if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH)
		{
			/* We have an access mode : we expect to find a handle */
			va_arg(varg_list, starpu_data_handle);
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
	arg_buffer[0] = nargs;

	va_end(varg_list);
}

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
}

void starpu_insert_task(starpu_codelet *cl, ...)
{
	va_list varg_list;

	/* The buffer will contain : nargs, {size, content} (x nargs)*/

	va_start(varg_list, cl);
        starpu_pack_cl_args(varg_list);

	va_start(varg_list, cl);
        struct starpu_task *task = starpu_task_create();
        starpu_insert_task_create_and_submit(arg_buffer_size, cl, &task, varg_list);
}
