/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#define N	4

int main(int argc, char **argv)
{
	int i, ret;

	starpu_init(NULL);

	struct starpu_task **tasks = malloc(N*sizeof(struct starpu_task *));

	for (i = 0; i < N; i++)
	{
		tasks[i] = starpu_task_create();
		tasks[i]->cl = NULL;

		if (i > 0)
		{
			starpu_task_declare_deps_array(tasks[i], 1, &tasks[i-1]);
			ret = starpu_task_submit(tasks[i]);
			STARPU_ASSERT(!ret);
		}

		if (i == (N-1))
			tasks[i]->detach = 0;
	}

	ret = starpu_task_submit(tasks[0]);
	STARPU_ASSERT(!ret);

	starpu_task_wait(tasks[N-1]);

	starpu_shutdown();

	return 0;
}
