/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Centre National de la Recherche Scientifique
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

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

int main(int argc, char **argv)
{
	int ret;
	float *buffer;
	float *buffer2;
	float *buffer3;
	size_t global_size;

	setenv("STARPU_LIMIT_CUDA_MEM", "1", 1);
	setenv("STARPU_LIMIT_OPENCL_MEM", "1", 1);
	setenv("STARPU_LIMIT_CPU_MEM", "1", 1);

        ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	global_size = _starpu_memory_manager_get_global_memory_size(0);
	if (global_size == 0)
	{
		FPRINTF(stderr, "Global memory size unavailable, skip the test\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	STARPU_CHECK_RETURN_VALUE_IS((int)global_size, 1*1024*1024, "get_global_memory_size");
	FPRINTF(stderr, "Available memory size on node 0: %ld\n", global_size);

	ret = starpu_malloc((void **)&buffer, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
	FPRINTF(stderr, "Allocation succesfull for 1 b\n");

	ret = starpu_malloc((void **)&buffer2, 1*1024*512);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
	FPRINTF(stderr, "Allocation succesfull for %d b\n", 1*1024*512);

	ret = starpu_malloc((void **)&buffer3, 1*1024*512);
	STARPU_CHECK_RETURN_VALUE_IS(ret, -ENOMEM, "starpu_malloc");
	FPRINTF(stderr, "Allocation failed for %d b\n", 1*1024*512);

	starpu_free(buffer2);
	FPRINTF(stderr, "Freeing %d b\n", 1*1024*512);

	ret = starpu_malloc((void **)&buffer3, 1*1024*512);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
	FPRINTF(stderr, "Allocation succesfull for %d b\n", 1*1024*512);

	starpu_free(buffer3);
	starpu_free(buffer);

	starpu_shutdown();
	return 0;
}

 #endif
