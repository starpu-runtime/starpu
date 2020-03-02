/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

void func_cpu(void *descr[], void *_args)
{
	(void) descr;
	(void) _args;
}

struct starpu_codelet codelet =
{
	.cpu_funcs = { func_cpu },
	.cpu_funcs_name = { "func_cpu" }
};

int main(int argc, char **argv)
{
        int ret;
	void *cl_arg = NULL;
	size_t cl_arg_size = 0;
	struct starpu_task *task;

	(void)argv;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_codelet_pack_args(&cl_arg, &cl_arg_size,
				 STARPU_VALUE, &argc, sizeof(argc),
				 0);

	task = starpu_task_build(&codelet,
				 STARPU_CL_ARGS, cl_arg, cl_arg_size,
				 STARPU_VALUE, &argc, sizeof(argc),
				 0);
	starpu_shutdown();

	FPRINTF(stderr, "Task %p\n", task);
	return (task==NULL)?0:1;
}
