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

/*
 * This test stress the memory allocation system and should force StarPU to
 * reclaim memory from time to time. 
 */

#include <assert.h>
#include <starpu.h>

static unsigned ntasks = 10000;

static void dummy_func(void *descr[], __attribute__ ((unused)) void *_args)
{
}

static starpu_codelet dummy_cl = {
        .where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dummy_func,
	.cuda_func = dummy_func,
	.nbuffers = 3
};

/* Number of 1MB chunks */
static int mb = 256;

int main(int argc, char **argv)
{
	int i;
	int task;

	/* An optional argument indicates the number of MB to allocate */
	if (argc > 1)
		mb = atoi(argv[1]);

	if (2*mb > ntasks)
		ntasks = 2*mb;

	fprintf(stderr, "Allocate %d buffers and create %d tasks\n", mb, ntasks);

        starpu_init(NULL);

	float **host_ptr_array;
	starpu_data_handle *handle_array;

	host_ptr_array = calloc(mb, sizeof(float *));
	handle_array = calloc(mb, sizeof(starpu_data_handle));

	/* Register mb buffers of 1MB */
	for (i = 0; i < mb; i++)
	{
		host_ptr_array[i] = malloc(1024*1024);
		assert(host_ptr_array[i]);
		starpu_variable_data_register(&handle_array[i], 0,
			(uintptr_t)host_ptr_array[i], 1024*1024);
		assert(handle_array[i]);
	}

	for (task = 0; task < ntasks; task++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &dummy_cl;
		task->buffers[0].handle = handle_array[i%mb];
		task->buffers[0].mode = STARPU_RW;
		task->buffers[1].handle = handle_array[(i+1)%mb];
		task->buffers[1].mode = STARPU_R;
		task->buffers[2].handle = handle_array[(i+2)%mb];
		task->buffers[2].mode = STARPU_R;
		starpu_task_submit(task);
	}

	starpu_task_wait_for_all();

	for (i = 0; i < mb; i++)
	{
		starpu_data_unregister(handle_array[i]);
		free(host_ptr_array[i]);
	}

	starpu_shutdown();

	return 0;
}
