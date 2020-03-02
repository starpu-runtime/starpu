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

#include <assert.h>
#include <starpu.h>
#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif
#include "../helper.h"

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

/*
 * Stress the memory allocation system and force StarPU to reclaim memory from
 * time to time.
 */

#ifdef STARPU_QUICK_CHECK
#  define BLOCK_SIZE (64*1024)
static unsigned ntasks = 250;
#else
#  define BLOCK_SIZE (64*1024*1024)
static unsigned ntasks = 1000;
#endif


#ifdef STARPU_HAVE_HWLOC
static uint64_t get_total_memory_size(void)
{
	uint64_t size;
	hwloc_topology_t hwtopology;
	hwloc_topology_init(&hwtopology);
	hwloc_topology_load(hwtopology);
	hwloc_obj_t root = hwloc_get_root_obj(hwtopology);
#if HWLOC_API_VERSION >= 0x00020000
	size = root->total_memory;
#else
	size = root->memory.total_memory;
#endif
	hwloc_topology_destroy(hwtopology);
	return size;
}
#endif

void dummy_func(void *descr[], void *_args)
{
}

static unsigned int i = 0;
void f(void *arg)
{
	printf("%u\n", ++i);
}

static struct starpu_codelet dummy_cl =
{
	.cpu_funcs = {dummy_func},
	.cuda_funcs = {dummy_func},
	.opencl_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.nbuffers = 3,
	.modes = {STARPU_RW, STARPU_R, STARPU_R}
};

/* Number of chunks */
static unsigned mb = 16;

int main(int argc, char **argv)
{
	unsigned j, taskid;
	int ret;

#ifdef STARPU_HAVE_HWLOC
	/* We allocate 50% of the memory */
	uint64_t total_size = get_total_memory_size();

	/* On x86_64-freebsd8.2, hwloc 1.3 returns 0 as the total memory
	 * size, so sanity-check what we have.  */
	if (total_size > 0)
		mb = (int)((0.50 * total_size)/(BLOCK_SIZE));
#endif

	setenv("STARPU_LIMIT_OPENCL_MEM", "1000", 1);

        ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* An optional argument indicates the number of MB to allocate */
	if (argc > 1)
		mb = atoi(argv[1]);

	if (2*mb > ntasks)
		ntasks = 2*mb;

#ifdef STARPU_QUICK_CHECK
	mb /= 100;
	if (mb == 0)
		mb = 1;
#endif

	FPRINTF(stderr, "Allocate %u buffers of size %d and create %u tasks\n", mb, BLOCK_SIZE, ntasks);

	float **host_ptr_array;
	starpu_data_handle_t *handle_array;

	host_ptr_array = calloc(mb, sizeof(float *));
	STARPU_ASSERT(host_ptr_array);
	handle_array = calloc(mb, sizeof(starpu_data_handle_t));
	STARPU_ASSERT(handle_array);

	/* Register mb buffers of 1MB */
	for (j = 0; j < mb; j++)
	{
		size_t size = starpu_lrand48()%BLOCK_SIZE + 1;
		host_ptr_array[j] = calloc(size, 1);
		if (host_ptr_array[j] == NULL)
		{
			mb = j;
			FPRINTF(stderr, "Cannot allocate more than %u buffers\n", mb);
			break;
		}
		starpu_variable_data_register(&handle_array[j], STARPU_MAIN_RAM, (uintptr_t)host_ptr_array[j], size);
		STARPU_ASSERT(handle_array[j]);
	}

	for (taskid = 0; taskid < ntasks; taskid++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &dummy_cl;
		task->handles[0] = handle_array[taskid%mb];
		task->handles[1] = handle_array[(taskid+1)%mb];
		task->handles[2] = handle_array[(taskid+2)%mb];
		task->callback_func = f;
		task->callback_arg = NULL;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	for (j = 0; j < mb; j++)
	{
		if ( j%20 == 0 )
			starpu_data_unregister_submit(handle_array[j]);
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	for (j = 0; j < mb; j++)
	{
		if ( j%20 != 0 )
			starpu_data_unregister(handle_array[j]);
		free(host_ptr_array[j]);
	}

	free(host_ptr_array);
	free(handle_array);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}

#endif
