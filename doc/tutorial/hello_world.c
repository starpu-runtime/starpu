/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

struct params
{
	int i;
	float f;
};

void cpu_func(void *buffers[], void *cl_arg)
{
	struct params *params = cl_arg;

	printf("Hello world (params = {%i, %f} )\n", params->i, params->f);
}

struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_func},
	.nbuffers = 0
};

void callback_func(void *callback_arg)
{
	printf("Callback function (arg %p)\n", callback_arg);
}

int main(int argc, char **argv)
{
	int ret;

	/* initialize StarPU */
	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task *task = starpu_task_create();

	task->cl = &cl; /* Pointer to the codelet defined above */

	struct params params = { 1, 2.0f };
	task->cl_arg = &params;
	task->cl_arg_size = sizeof(params);

	task->callback_func = callback_func;
	task->callback_arg = (void*) (uintptr_t) 0x42;

	/* starpu_task_submit will be a blocking call */
	task->synchronous = 1;

	/* submit the task to StarPU */
	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* terminate StarPU */
	starpu_shutdown();

	return 0;
}
