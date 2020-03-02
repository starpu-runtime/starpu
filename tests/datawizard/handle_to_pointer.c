/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

/*
 * Test the value returned by starpu_handle_to_pointer
 */

void cpu_task(void **buffers, void *args)
{
	int *numbers;
	int i;
	int size;

	numbers = (int *) STARPU_VECTOR_GET_PTR(buffers[0]);
	starpu_codelet_unpack_args (args, &size);

	for(i = 0; i < (int)size; i++)
	{
		numbers[i] = i;
	}
}

#ifdef STARPU_USE_CUDA
static void cuda_task(void **buffers, void *args)
{
	int *numbers;
	int i;
	int size;

	numbers = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
	starpu_codelet_unpack_args (args, &size);

	for(i = 0; i < (int)size; i++)
	{
		cudaMemcpyAsync(&numbers[i], &i, sizeof(int), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	}
}
#endif

#ifdef STARPU_USE_OPENCL
static void opencl_task(void *buffers[], void *args)
{
	(void)args;
	cl_command_queue queue;
	int id = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(id);
	starpu_opencl_get_queue(devid, &queue);

	cl_mem numbers = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

	unsigned i;
	for (i = 0; i < size; i++)
	{
		clEnqueueWriteBuffer(queue,
				numbers,
				CL_TRUE,
				i*sizeof(int),  /* offset */
				sizeof(int),
				&i,
				0,              /* num_events_in_wait_list */
				NULL,           /* event_wait_list */
				NULL            /* event */);
	}
}
#endif

static struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_task},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_task},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_task},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"cpu_task"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

int main(int argc, char *argv[])
{
	int err, ret;
	int *pointer;
	starpu_data_handle_t handle;
	static const int count = 123;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;

	err = starpu_malloc((void **)&pointer, count * sizeof(int));
	STARPU_ASSERT((err == 0) && (pointer != NULL));

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)pointer,
				      sizeof(int));
	STARPU_ASSERT(starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM) == pointer);
	STARPU_ASSERT(starpu_data_pointer_is_inside(handle, STARPU_MAIN_RAM, pointer));
	starpu_data_unregister(handle);

	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)pointer,
				    count, sizeof(int));
	STARPU_ASSERT(starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM) == pointer);
	STARPU_ASSERT(starpu_data_pointer_is_inside(handle, STARPU_MAIN_RAM, pointer));
	starpu_data_unregister(handle);

	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)pointer, 0,
				    count, 1, sizeof(int));
	STARPU_ASSERT(starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM) == pointer);
	STARPU_ASSERT(starpu_data_pointer_is_inside(handle, STARPU_MAIN_RAM, pointer));
	starpu_data_unregister(handle);

	starpu_free(pointer);
	pointer = NULL;

	/* Lazy allocation.  */
	starpu_vector_data_register(&handle, -1, 0 /* NULL */,
				    count, sizeof(int));
	STARPU_ASSERT(starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM) == NULL);

	/* Pass the handle to a task.  */
	err = starpu_task_insert(&cl,
				 STARPU_W, handle,
				 STARPU_VALUE, &count, sizeof(count),
				 0);
	if (err == -ENODEV)
		return STARPU_TEST_SKIPPED;

	/* Acquire the handle, forcing a local allocation.  */
	starpu_data_acquire(handle, STARPU_R);

	/* Make sure we have a local pointer to it.  */
	ret = EXIT_SUCCESS;
	pointer = (int *) starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM);
	if (pointer == NULL)
	{
		FPRINTF(stderr, "pointer should be non NULL\n");
		ret = EXIT_FAILURE;
	}
	else
	{
		int i;
		for(i = 0; i < count; i++)
		{
			int *numbers = (int *)pointer;
			STARPU_ASSERT(starpu_data_pointer_is_inside(handle, STARPU_MAIN_RAM, numbers));
			if (numbers[i] != i)
			{
				FPRINTF(stderr, "Incorrect value numbers[%d] == %d should be %d\n", (int)i, numbers[i], (int)i);
				ret = EXIT_FAILURE;
			}
		}
	}
	starpu_data_release(handle);

	starpu_data_unregister(handle);

	starpu_shutdown();

	return ret;
}
