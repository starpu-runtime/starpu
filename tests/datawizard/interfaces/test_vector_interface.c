/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Institut National de Recherche en Informatique et Automatique
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
#include "test_interfaces.h"

/* Prototypes */
static starpu_data_handle *register_data(void);
static void test_vector_cpu_func(void *buffers[], void *args);
#ifdef STARPU_USE_CUDA
extern void test_vector_cuda_func(void *buffers[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void test_vector_opencl_func(void *buffers[], void *args);
#endif


static starpu_data_handle *vector_handle;

struct test_config vector_config = {
	.cpu_func      = test_vector_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_vector_cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_vector_opencl_func,
#endif
	.register_func = register_data,
	.copy_failed   = 0,
	.name          = "vector_interface"
};

int n = 16;
int *vector;

static starpu_data_handle*
register_data(void)
{
	if (vector_handle)
		return vector_handle;

	/* Initializing data */
	int i;
	vector = malloc(n * sizeof(*vector));
	if (!vector)
		return NULL;
	for (i = 0; i < n; i++)
		vector[i] = i;

	/* Registering data */
	vector_handle = malloc(sizeof(*vector_handle));
	if (!vector_handle)
		return NULL;
	starpu_vector_data_register(vector_handle,
                                    0,
                                    (uintptr_t)vector,
                                     n,
                                     sizeof(int));
	return vector_handle;
}

static void test_vector_cpu_func(void *buffers[], void *args)
{
	unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
	int *val = (int *) STARPU_VECTOR_GET_PTR(buffers[0]);
	int factor = *(int*)args;
	unsigned int i;
	for (i = 0; i < n; i++) {
		if (val[i] != i*factor) {
			fprintf(stderr, "HI %d => %d\n", i, val[i]);
			vector_config.copy_failed = 1;
			return;
		}
		val[i] = -val[i];
	}
}
