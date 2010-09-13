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

#include <starpu.h>

static unsigned ntasks = 400;

void increment_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(*tokenptr)++;
}

static starpu_codelet increment_cl = {
        .where = STARPU_CPU,
	.cpu_func = increment_cpu,
	.nbuffers = 1
};

unsigned token = 42;
starpu_data_handle token_handle;

void increment_token()
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &increment_cl;
	task->buffers[0].handle = token_handle;
	task->buffers[0].mode = STARPU_RW;
	starpu_task_submit(task);
}

void callback(void *arg __attribute__ ((unused)))
{
        starpu_data_release(token_handle);
}

int main(int argc, char **argv)
{
	int i;

        starpu_init(NULL);
	starpu_variable_data_register(&token_handle, 0, (uintptr_t)&token, sizeof(unsigned));

	for(i=0; i<ntasks; i++)
	{
                starpu_data_acquire(token_handle, STARPU_RW);
                increment_token();
                starpu_data_release(token_handle);

                starpu_data_acquire_cb(token_handle, STARPU_RW, callback, NULL);
	}

	starpu_shutdown();

	return 0;
}
