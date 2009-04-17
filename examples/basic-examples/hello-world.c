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
#include <stdint.h>
#include <starpu.h>

void callback_func(void *callback_arg)
{
	printf("Callback function got argument %p\n", callback_arg);
}

void cpu_func(starpu_data_interface_t *buffers, void *func_arg)
{
	float *array = func_arg;

	printf("Hello world (array = {%f, %f} )\n", array[0], array[1]);
}

starpu_codelet cl =
{
	.where = CORE,
	.core_func = cpu_func,
	.nbuffers = 0
};

int main(int argc, char **argv)
{
	/* initialize StarPU */
	starpu_init(NULL);
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl;

	float array[2] = {1.0f, -1.0f};
	task->cl_arg = &array;
	task->cl_arg_size = 2*sizeof(float);
		
	task->callback_func = callback_func;
	task->callback_arg = (void*) (uintptr_t) 0x42;

	/* starpu_submit_task will be a blocking call */
	task->synchronous = 1;
	
	/* submit the task to StarPU */
	starpu_submit_task(task);
	
	/* terminate StarPU */
	starpu_shutdown();

	return 0;
}
