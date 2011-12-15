/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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

void cpu_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	int *valin = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *valout = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*valout = *valin;
}

struct starpu_codelet cl_cpu_funcs =
{
	.where = STARPU_CPU,
	.cpu_funcs = {cpu_codelet, NULL},
	.nbuffers = 2,
	.name = "cpu_funcs",
};

struct starpu_codelet cl_cpu_func =
{
	.where = STARPU_CPU,
	.cpu_func = cpu_codelet,
	.nbuffers = 2,
	.name = "cpu_func",
};

struct starpu_codelet cl_cpu_multiple =
{
	.where = STARPU_CPU,
	.cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS,
	.cpu_funcs = {cpu_codelet, NULL},
	.nbuffers = 2,
	.name = "cpu_multiple",
};

int submit_codelet(struct starpu_codelet *cl)
{
	int x=42, y=14;
	starpu_data_handle_t handles[2];

	starpu_variable_data_register(&handles[0], 0, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&handles[1], 0, (uintptr_t)&y, sizeof(y));

	starpu_insert_task(cl,
			   STARPU_R, handles[0],
			   STARPU_W, handles[1],
			   0);
	starpu_task_wait_for_all();

	starpu_data_unregister(handles[0]);
	starpu_data_unregister(handles[1]);

	if (x != y)
	{
		FPRINTF(stderr, "error when executing codelet <%s>\n", cl->name);
	}
	else
	{
		FPRINTF(stderr, "success when executing codelet <%s>\n", cl->name);
	}
	return (x == y);
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = submit_codelet(&cl_cpu_func);
	if (ret)
	{
		submit_codelet(&cl_cpu_funcs);
	}
	if (ret)
	{
		submit_codelet(&cl_cpu_multiple);
	}

	starpu_shutdown();

	return ret;
}
