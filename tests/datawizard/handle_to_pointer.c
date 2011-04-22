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

static void cpu_task(void **buffers, void *args)
{
	int *numbers;
	int i;
	size_t size;

	numbers = (int *) STARPU_VECTOR_GET_PTR(buffers[0]);
	starpu_unpack_cl_args (args, &size);

	for(i = 0; i < size; i++)
	{
		numbers[i] = i;
	}
}

#ifdef STARPU_USE_CUDA
static void cuda_task(void **buffers, void *args)
{
	int *numbers;
	int i;
	size_t size;

	numbers = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
	starpu_unpack_cl_args (args, &size);

	for(i = 0; i < size; i++)
	{
		cudaMemcpy(&numbers[i], &i, sizeof(int), cudaMemcpyHostToDevice);
	}
}
#endif

static starpu_codelet cl = {
	.where = STARPU_CPU | STARPU_CUDA,
	.cpu_func = cpu_task,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_task,
#endif
	.nbuffers = 1
};

int main(int argc, char *argv[])
{
	int err;
	size_t i;
	int *pointer;
	starpu_data_handle handle;
	static const size_t count = 123;

	starpu_init(NULL);

	err = starpu_malloc((void **)&pointer, count * sizeof(int));
	assert((err == 0) && (pointer != NULL));

	starpu_variable_data_register(&handle, 0, (uintptr_t)pointer,
				      sizeof(int));
	assert(starpu_handle_to_pointer(handle, 0) == pointer);
	starpu_data_unregister(handle);

	starpu_vector_data_register(&handle, 0, (uintptr_t)pointer,
				    count, sizeof(int));
	assert(starpu_handle_to_pointer(handle, 0) == pointer);
	starpu_data_unregister(handle);

	starpu_matrix_data_register(&handle, 0, (uintptr_t)pointer, 0,
				    count, 1, sizeof(int));
	assert(starpu_handle_to_pointer(handle, 0) == pointer);
	starpu_data_unregister(handle);

	starpu_free(pointer);
	pointer = NULL;

	/* Lazy allocation.  */
	starpu_vector_data_register(&handle, -1, 0 /* NULL */,
				    count, sizeof(int));
	assert(starpu_handle_to_pointer(handle, 0) == NULL);

	/* Pass the handle to a task.  */
	starpu_insert_task(&cl,
			   STARPU_W, handle,
			   STARPU_VALUE, &count, sizeof(count),
			   0);

	/* Acquire the handle, forcing a local allocation.  */
	starpu_data_acquire(handle, STARPU_R);

	/* Make sure we have a local pointer to it.  */
	pointer = starpu_handle_to_pointer(handle, 0);
	assert(pointer != NULL);
	for(i = 0; i < count; i++)
	{
		int *numbers = (int *)pointer;
		assert(numbers[i] == i);
	}
	starpu_data_release(handle);

	starpu_data_unregister(handle);

	starpu_shutdown();

	return EXIT_SUCCESS;
}
