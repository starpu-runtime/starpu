/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
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

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

static unsigned ntasks = 40000;

#ifdef STARPU_USE_CUDA
extern void increment_cuda(void *descr[], __attribute__ ((unused)) void *_args);
#endif

void increment_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(*tokenptr)++;
}

static starpu_codelet increment_cl = {
        .where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = increment_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = increment_cuda,
#endif
	.nbuffers = 1
};

unsigned token = 0;
starpu_data_handle token_handle;

void increment_token(int synchronous)
{
	struct starpu_task *task = starpu_task_create();
        task->synchronous = synchronous;
	task->cl = &increment_cl;
	task->buffers[0].handle = token_handle;
	task->buffers[0].mode = STARPU_RW;
	starpu_task_submit(task);
}

void callback(void *arg __attribute__ ((unused)))
{
        starpu_data_release(token_handle);
}

#warning TODO add threads

int main(int argc, char **argv)
{
	int i;

        starpu_init(NULL);
	starpu_variable_data_register(&token_handle, 0, (uintptr_t)&token, sizeof(unsigned));

        FPRINTF(stderr, "Token: %u\n", token);

	for(i=0; i<ntasks; i++)
	{
                starpu_data_acquire_cb(token_handle, STARPU_W, callback, NULL);  // recv
                increment_token(0);
                starpu_data_acquire_cb(token_handle, STARPU_R, callback, NULL);  // send
	}

	starpu_data_unregister(token_handle);
        FPRINTF(stderr, "Token: %u\n", token);
        assert(token==ntasks);

	starpu_shutdown();

	return 0;
}
