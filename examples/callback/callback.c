/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include <pthread.h>
#include <sys/time.h>

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

starpu_data_handle handle;

void cpu_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*val += 1;
}

starpu_codelet cl =
{
	.where = STARPU_CPU,
	.cpu_func = cpu_codelet,
	.nbuffers = 1
};

void callback_func(void *callback_arg)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &cl;
	task->buffers[0].handle = handle;
	task->buffers[0].mode = STARPU_RW;
	starpu_task_submit(task);
}

int main(int argc, char **argv)
{
	int v=40;

	starpu_init(NULL);
	starpu_variable_data_register(&handle, 0, (uintptr_t)&v, sizeof(int));

	struct starpu_task *task = starpu_task_create();
	task->cl = &cl;
	task->callback_func = callback_func;
	task->callback_arg = NULL;
	task->buffers[0].handle = handle;
	task->buffers[0].mode = STARPU_RW;

	starpu_task_submit(task);

	starpu_task_wait_for_all();

	FPRINTF(stderr, "v -> %d\n", v);

	starpu_shutdown();

	return 0;
}
