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

#undef NDEBUG
#include <assert.h>

#include <starpu.h>
#include <stdlib.h>
#include <sys/types.h>

#include <stdio.h>

#define VECTOR_COUNT    12
#define VARIABLE_COUNT  42

#define VECTOR_SIZE     123

int main(int argc, char *argv[])
{
	int err;
	size_t i;
	void *vectors[VECTOR_COUNT], *variables[VARIABLE_COUNT];
	starpu_data_handle vector_handles[VECTOR_COUNT];
	starpu_data_handle variable_handles[VARIABLE_COUNT];

	starpu_init(NULL);

	/* Register data regions.  */

	for(i = 0; i < VARIABLE_COUNT; i++)
	{
		err = starpu_malloc(&variables[i], sizeof(float));
		assert(err == 0);
		starpu_variable_data_register(&variable_handles[i], 0,
					      (uintptr_t)variables[i],
					      sizeof(float));
	}

	for(i = 0; i < VECTOR_COUNT; i++)
	{
		err = starpu_malloc(&vectors[i], VECTOR_SIZE * sizeof(float));
		assert(err == 0);
		starpu_vector_data_register(&vector_handles[i], 0,
					    (uintptr_t)vectors[i],
					    VECTOR_SIZE, sizeof(float));
	}

	/* Look them up.  */

	for(i = 0; i < VARIABLE_COUNT; i++)
	{
		starpu_data_handle handle;

		handle = starpu_data_lookup(variables[i]);
		assert(handle == variable_handles[i]);
	}

	for(i = 0; i < VECTOR_COUNT; i++)
	{
		starpu_data_handle handle;

		handle = starpu_data_lookup(vectors[i]);
		assert(handle == vector_handles[i]);
	}

	/* Unregister them.  */

	for(i = 0; i < VARIABLE_COUNT; i++)
	{
		starpu_data_unregister(variable_handles[i]);
	}

	for(i = 0; i < VECTOR_COUNT; i++)
	{
		starpu_data_unregister(vector_handles[i]);
	}

	/* Make sure we can no longer find them.  */

	for(i = 0; i < VARIABLE_COUNT; i++)
	{
		starpu_data_handle handle;

		handle = starpu_data_lookup(variables[i]);
		assert(handle == NULL);
		starpu_free(variables[i]);
	}

	for(i = 0; i < VECTOR_COUNT; i++)
	{
		starpu_data_handle handle;

		handle = starpu_data_lookup(vectors[i]);
		assert(handle == NULL);
		starpu_free(vectors[i]);
	}

	starpu_shutdown();

	return EXIT_SUCCESS;
}
