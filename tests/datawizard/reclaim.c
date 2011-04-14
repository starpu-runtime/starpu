/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

/*
 * This test stress the memory allocation system and should force StarPU to
 * reclaim memory from time to time. 
 */

#include <assert.h>
#include <starpu.h>
#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#define BLOCK_SIZE	(64*1024*1024)
#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

static unsigned ntasks = 1000;

#ifdef STARPU_HAVE_HWLOC
static uint64_t get_total_memory_size(void)
{
	hwloc_topology_t hwtopology;
	hwloc_topology_init(&hwtopology);
	hwloc_topology_load(hwtopology);
	hwloc_obj_t root = hwloc_get_root_obj(hwtopology);	

	return root->memory.total_memory;
}
#endif

static void dummy_func(void *descr[], __attribute__ ((unused)) void *_args)
{
}

static starpu_codelet dummy_cl = {
        .where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dummy_func,
	.cuda_func = dummy_func,
	.nbuffers = 3
};

/* Number of chunks */
static int mb = 256;

int main(int argc, char **argv)
{
	int i;
	int taskid;

#ifdef STARPU_HAVE_HWLOC
	/* We allocate 50% of the memory */
	uint64_t total_size = get_total_memory_size();

	mb = (int)((0.50 * total_size)/(BLOCK_SIZE));
#endif

	/* An optional argument indicates the number of MB to allocate */
	if (argc > 1)
		mb = atoi(argv[1]);

	if (2*mb > ntasks)
		ntasks = 2*mb;

	FPRINTF(stderr, "Allocate %d buffers and create %u tasks\n", mb, ntasks);

        starpu_init(NULL);

	float **host_ptr_array;
	starpu_data_handle *handle_array;

	host_ptr_array = calloc(mb, sizeof(float *));
	handle_array = calloc(mb, sizeof(starpu_data_handle));

	/* Register mb buffers of 1MB */
	for (i = 0; i < mb; i++)
	{
		host_ptr_array[i] = malloc(BLOCK_SIZE);
		assert(host_ptr_array[i]);
		starpu_variable_data_register(&handle_array[i], 0,
			(uintptr_t)host_ptr_array[i], BLOCK_SIZE);
		assert(handle_array[i]);
	}

	for (taskid = 0; taskid < ntasks; taskid++)
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
