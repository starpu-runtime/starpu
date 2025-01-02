/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <datawizard/memory_manager.h>

/* Stress data allocation on a GPU, triggering eviction */

#define SIZE_LIMIT 128
#define STR_LIMIT "128"
#define SIZE_ALLOC 128

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

static
int test_prefetch(unsigned memnodes)
{
	float *buffers[4];
	starpu_data_handle_t handles[4];
	unsigned i, j;
	starpu_ssize_t available_size;

	if (starpu_getenv_number_default("STARPU_DIDUSE_BARRIER", 0))
		/* This would hang */
		return STARPU_TEST_SKIPPED;

	buffers[0] = malloc(SIZE_ALLOC*1024*512);
	STARPU_ASSERT(buffers[0]);

	/* Prefetch half the memory */
	starpu_variable_data_register(&handles[0], STARPU_MAIN_RAM, (uintptr_t)buffers[0], SIZE_ALLOC*1024*512);
	for(i=1 ; i<memnodes ; i++)
	{
		starpu_data_prefetch_on_node(handles[0], i, 0);
	}

	for(i=1 ; i<memnodes ; i++)
	{
		available_size = starpu_memory_get_available(i);
		FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
		STARPU_CHECK_RETURN_VALUE_IS((int) available_size, SIZE_ALLOC*1024*512, "starpu_memory_get_available (node %u)", i);
	}

	/* Prefetch a quarter of the memory */
	buffers[1] = malloc(SIZE_ALLOC*1024*256);
	STARPU_ASSERT(buffers[1]);

	starpu_variable_data_register(&handles[1], STARPU_MAIN_RAM, (uintptr_t)buffers[1], SIZE_ALLOC*1024*256);
	for(i=1 ; i<memnodes ; i++)
	{
		starpu_data_prefetch_on_node(handles[1], i, 0);
	}

	for(i=1 ; i<memnodes ; i++)
	{
		available_size = starpu_memory_get_available(i);
		FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
		STARPU_CHECK_RETURN_VALUE_IS((int)available_size, SIZE_ALLOC*1024*256, "starpu_memory_get_available (node %u)", i);
	}

	/* Fetch a bit more than half of the memory, it should be able to push previous data out */
	buffers[2] = malloc(SIZE_ALLOC*1024*600);
	STARPU_ASSERT(buffers[2]);

	starpu_variable_data_register(&handles[2], STARPU_MAIN_RAM, (uintptr_t)buffers[2], SIZE_ALLOC*1024*600);
	for(i=1 ; i<memnodes ; i++)
	{
		starpu_data_fetch_on_node(handles[2], i, 0);
	}

	for(i=1 ; i<memnodes ; i++)
	{
		available_size = starpu_memory_get_available(i);
		FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
		// here, we do not know which data has been cleaned, we cannot test the exact amount of available memory
		STARPU_CHECK_RETURN_VALUE((available_size == 0), "starpu_memory_get_available (node %u)", i);
	}

	/* Fetch half of the memory, it should be able to push previous data out */
	buffers[3] = malloc(SIZE_ALLOC*1024*512);
	STARPU_ASSERT(buffers[3]);

	starpu_variable_data_register(&handles[3], STARPU_MAIN_RAM, (uintptr_t)buffers[3], SIZE_ALLOC*1024*512);
	for(i=0 ; i<memnodes ; i++)
	{
		starpu_data_fetch_on_node(handles[3], i, 0);
	}

	for(i=1 ; i<memnodes ; i++)
	{
		available_size = starpu_memory_get_available(i);
		FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
		STARPU_CHECK_RETURN_VALUE_IS((int)available_size, SIZE_ALLOC*1024*512, "starpu_memory_get_available (node %u)", i);
	}

	for(i=0 ; i<4 ; i++)
	{
		starpu_data_unregister(handles[i]);
		free(buffers[i]);
	}

	i = 0;
	available_size = starpu_memory_get_available(i);
	FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
	STARPU_CHECK_RETURN_VALUE_IS((int)available_size, SIZE_ALLOC*1024*1024, "starpu_memory_get_available (node %u)", i);

	/* Write data directly in GPU without any allocation in main memory, starpu should be able to evict to main memory progressively */
	for(i=1 ; i<memnodes ; i++)
	{
		for(j=0 ; j<4 ; j++)
		{
			starpu_variable_data_register(&handles[j], -1, 0, SIZE_ALLOC*1024*400);
			starpu_data_acquire_on_node(handles[j], i, STARPU_W);
			starpu_data_release_on_node(handles[j], i);
		}
		for(j=0 ; j<4 ; j++)
		{
			starpu_data_unregister(handles[j]);
		}
	}

	/* Try again, but with different data sizes now */
	for(i=1 ; i<memnodes ; i++)
	{
		for(j=0 ; j<4 ; j++)
		{
			starpu_variable_data_register(&handles[j], -1, 0, SIZE_ALLOC*1024*400 - j);
			starpu_data_acquire_on_node(handles[j], i, STARPU_W);
			starpu_data_release_on_node(handles[j], i);
		}
		for(j=0 ; j<4 ; j++)
		{
			starpu_data_unregister(handles[j]);
		}
	}


	for(i=1 ; i<memnodes ; i++)
	{
		available_size = starpu_memory_get_available(i);
		FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
		/* STARPU_CHECK_RETURN_VALUE_IS((int)available_size, SIZE_ALLOC*1024*1024, "starpu_memory_get_available (node %u)", i); */
	}

	return 0;
}

static
void test_malloc()
{
	int ret;
	float *buffer;
	float *buffer2;
	float *buffer3;

	/* Allocate one byte */
	ret = starpu_malloc_flags((void **)&buffer, 1, STARPU_MALLOC_COUNT);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc_flags");
	FPRINTF(stderr, "Allocation successful for 1 b\n");

	/* Allocate half the memory */
	ret = starpu_malloc_flags((void **)&buffer2, SIZE_ALLOC*1024*512, STARPU_MALLOC_COUNT);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc_flags");
	FPRINTF(stderr, "Allocation successful for %d b\n", SIZE_ALLOC*1024*512);

	/* Try to allocate the other half, should fail */
	ret = starpu_malloc_flags((void **)&buffer3, SIZE_ALLOC*1024*512, STARPU_MALLOC_COUNT);
	STARPU_CHECK_RETURN_VALUE_IS(ret, -ENOMEM, "starpu_malloc_flags");
	FPRINTF(stderr, "Allocation failed for %d b\n", SIZE_ALLOC*1024*512);

	/* Try to allocate the other half without counting it, should succeed */
	ret = starpu_malloc_flags((void **)&buffer3, SIZE_ALLOC*1024*512, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc_flags");
	FPRINTF(stderr, "Allocation successful for %d b\n", SIZE_ALLOC*1024*512);
	starpu_free_flags(buffer3, SIZE_ALLOC*1024*512, 0);

	/* Free the initial half-memory allocation */
	starpu_free_flags(buffer2, SIZE_ALLOC*1024*512, STARPU_MALLOC_COUNT);
	FPRINTF(stderr, "Freeing %d b\n", SIZE_ALLOC*1024*512);

	/* Should not be able to allocate half the memory again */
	ret = starpu_malloc_flags((void **)&buffer3, SIZE_ALLOC*1024*512, STARPU_MALLOC_COUNT);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc_flags");
	FPRINTF(stderr, "Allocation successful for %d b\n", SIZE_ALLOC*1024*512);

	starpu_free_flags(buffer3, SIZE_ALLOC*1024*512, STARPU_MALLOC_COUNT);
	starpu_free_flags(buffer, 1, STARPU_MALLOC_COUNT);
}

int main(void)
{
	int ret;
	unsigned memnodes, i;

	setenv("STARPU_LIMIT_CUDA_MEM", STR_LIMIT, 1);
	setenv("STARPU_LIMIT_OPENCL_MEM", STR_LIMIT, 1);
	setenv("STARPU_LIMIT_HIP_MEM", STR_LIMIT, 1);
	setenv("STARPU_LIMIT_CPU_NUMA_MEM", STR_LIMIT, 1);

	struct starpu_conf conf;
	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return EXIT_FAILURE;
	conf.enable_map = 0;
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	memnodes = starpu_memory_nodes_get_count();
	for(i=0 ; i<memnodes ; i++)
	{
		starpu_ssize_t available_size;
		available_size = starpu_memory_get_available(i);
		if (available_size == -1)
		{
			FPRINTF(stderr, "Global memory size for node %u unavailable, skip the test\n", i);
			starpu_shutdown();
			return STARPU_TEST_SKIPPED;
		}
		FPRINTF(stderr, "Available memory size on node %u: %zd\n", i, available_size);
		STARPU_CHECK_RETURN_VALUE_IS((int)available_size, SIZE_LIMIT*1024*1024, "starpu_memory_get_available (node %u)", i);
	}

	test_malloc();
	ret = test_prefetch(memnodes);

	starpu_shutdown();
	return ret;
}

 #endif

