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
#include <util/starpu_insert_task_utils.h>

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
	va_list varg_list;

	/* The buffer will contain : nargs, {size, content} (x nargs)*/

	/* Compute the size */
	size_t arg_buffer_size = 0;

	va_start(varg_list, cl);
        arg_buffer_size = starpu_insert_task_get_arg_size(varg_list);

	va_start(varg_list, cl);
        struct starpu_task *task = starpu_task_create();
        starpu_insert_task_create_and_submit(arg_buffer_size, cl, &task, varg_list);

}
