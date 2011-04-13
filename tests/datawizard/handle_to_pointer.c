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

#define COUNT 123

int main(int argc, char *argv[])
{
	int err;
	void *pointer;
	starpu_data_handle handle;

	starpu_init(NULL);

	err = starpu_malloc(&pointer, COUNT * sizeof(float));
	assert((err == 0) && (pointer != NULL));

	starpu_variable_data_register(&handle, 0, (uintptr_t)pointer,
				      sizeof(float));
	assert(starpu_handle_to_pointer(handle) == pointer);
	starpu_data_unregister(handle);

	starpu_vector_data_register(&handle, 0, (uintptr_t)pointer,
				    COUNT, sizeof(float));
	assert(starpu_handle_to_pointer(handle) == pointer);
	starpu_data_unregister(handle);

	starpu_matrix_data_register(&handle, 0, (uintptr_t)pointer, 0,
				    COUNT, 1, sizeof(float));
	assert(starpu_handle_to_pointer(handle) == pointer);
	starpu_data_unregister(handle);

	starpu_shutdown();

	return EXIT_SUCCESS;
}
