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

/*
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_vector_data_register)
 *  2- how to describe which data are accessed by a task (task->handle[0])
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */
#include <starpu.h>

#define    NX    2048

extern void vector_scal_cpu(void *buffers[], void *_args);
extern void vector_scal_cuda(void *buffers[], void *_args);
extern void vector_scal_opencl(void *buffers[], void *_args);

static struct starpu_codelet cl = {
	/* CPU implementation of the codelet */
	.cpu_funcs = {vector_scal_cpu},
#ifdef STARPU_USE_CUDA
	/* CUDA implementation of the codelet */
	.cuda_funcs = {vector_scal_cuda},
#endif
#ifdef STARPU_USE_OPENCL
	/* OpenCL implementation of the codelet */
	.opencl_funcs = {vector_scal_opencl},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program programs;
#endif

int main(int argc, char **argv)
{
	/* We consider a vector of float that is initialized just as any of C
	 * data */
	float vector[NX];
	unsigned i;
	for (i = 0; i < NX; i++)
		vector[i] = 1.0f;

	fprintf(stderr, "BEFORE : First element was %f\n", vector[0]);

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	starpu_opencl_load_opencl_from_file("vector_scal_opencl_kernel.cl", &programs, NULL);
#endif

	/* Tell StaPU to associate the "vector" vector with the "vector_handle"
	 * identifier. When a task needs to access a piece of data, it should
	 * refer to the handle that is associated to it.
	 * In the case of the "vector" data interface:
	 *  - the first argument of the registration method is a pointer to the
	 *    handle that should describe the data
	 *  - the second argument is the memory node where the data (ie. "vector")
	 *    resides initially: STARPU_MAIN_RAM stands for an address in main memory, as
	 *    opposed to an adress on a GPU for instance.
	 *  - the third argument is the adress of the vector in RAM
	 *  - the fourth argument is the number of elements in the vector
	 *  - the fifth argument is the size of each element.
	 */
	starpu_data_handle_t vector_handle;
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector,
				    NX, sizeof(vector[0]));

	float factor = 3.14;

	/* create a synchronous task: any call to starpu_task_submit will block
	 * until it is terminated */
	struct starpu_task *task = starpu_task_create();
	task->synchronous = 1;

	task->cl = &cl;

	/* the codelet manipulates one buffer in RW mode */
	task->handles[0] = vector_handle;

	/* an argument is passed to the codelet, beware that this is a
	 * READ-ONLY buffer and that the codelet may be given a pointer to a
	 * COPY of the argument */
	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(factor);

	/* execute the task on any eligible computational ressource */
	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* StarPU does not need to manipulate the array anymore so we can stop
	 * monitoring it */
	starpu_data_unregister(vector_handle);

#ifdef STARPU_USE_OPENCL
	starpu_opencl_unload_opencl(&programs);
#endif

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);

	return 0;
}
