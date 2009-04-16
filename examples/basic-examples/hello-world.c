/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <stdio.h>
#include <starpu.h>

void callback_func(void *callback_arg)
{
	printf("Callback function got argument %x\n", callback_arg);
}

void core_func(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	printf("Hello world\n");
}

int main(int argc, char **argv)
{
	/* initialize StarPU */
	starpu_init(NULL);

	starpu_codelet cl =
	{
		.where = CORE,
		.core_func = core_func,
		.nbuffers = 0
	};

	struct starpu_task *task = starpu_task_create();

	task->cl = &cl;
		
	task->callback_func = callback_func;
	task->callback_arg = 0x42;

	/* starpu_submit_task will be a blocking call */
	task->synchronous = 1;
	
	/* submit the task to StarPU */
	starpu_submit_task(task);
	
	/* terminate StarPU */
	starpu_shutdown();

	return 0;
}
