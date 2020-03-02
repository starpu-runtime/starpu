/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 *  2- how to describe which data are accessed by a task (task->handles[0])
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define	NX	204800
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void scal_cpu_func(void *buffers[], void *_args);
extern void scal_cpu_func_icc(void *buffers[], void *_args);
extern void scal_sse_func(void *buffers[], void *_args);
extern void scal_sse_func_icc(void *buffers[], void *_args);
extern void scal_cuda_func(void *buffers[], void *_args);
extern void scal_opencl_func(void *buffers[], void *_args);

static struct starpu_perfmodel vector_scal_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "vector_scal"
};

static struct starpu_perfmodel vector_scal_energy_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "vector_scal_energy"
};

static struct starpu_codelet cl =
{
	/* CPU implementation of the codelet */
	.cpu_funcs =
	{
		scal_cpu_func
#if defined(STARPU_HAVE_ICC) && !defined(__KNC__) && !defined(__KNF__)
		, scal_cpu_func_icc
#endif
#ifdef __SSE__
		, scal_sse_func
#if defined(STARPU_HAVE_ICC) && !defined(__KNC__) && !defined(__KNF__)
		, scal_sse_func_icc
#endif
#endif
	},
	.cpu_funcs_name =
	{
		"scal_cpu_func",
#if defined(STARPU_HAVE_ICC) && !defined(__KNC__) && !defined(__KNF__)
		"scal_cpu_func_icc",
#endif
#ifdef __SSE__
		"scal_sse_func",
#if defined(STARPU_HAVE_ICC) && !defined(__KNC__) && !defined(__KNF__)
		"scal_sse_func_icc"
#endif
#endif
	},

#ifdef STARPU_USE_CUDA
	/* CUDA implementation of the codelet */
	.cuda_funcs = {scal_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	/* OpenCL implementation of the codelet */
	.opencl_funcs = {scal_opencl_func},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &vector_scal_model,
	.energy_model = &vector_scal_energy_model
};

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

static int approximately_equal(float a, float b)
{
#ifdef STARPU_HAVE_NEARBYINTF
	int ai = (int) nearbyintf(a * 1000.0);
	int bi = (int) nearbyintf(b * 1000.0);
#elif defined(STARPU_HAVE_RINTF)
	int ai = (int) rintf(a * 1000.0);
	int bi = (int) rintf(b * 1000.0);
#else
#error "Please define either nearbyintf or rintf."
#endif
	return ai == bi;
}

int main(void)
{
	/* We consider a vector of float that is initialized just as any of C
 	 * data */
	float vector[NX];
	unsigned i;
	for (i = 0; i < NX; i++)
                vector[i] = (i+1.0f);

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);
	if (ret == -ENODEV) goto enodev;

	FPRINTF(stderr, "[BEFORE] 1-th element    : %3.2f\n", vector[1]);
	FPRINTF(stderr, "[BEFORE] (NX-1)th element: %3.2f\n", vector[NX-1]);

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("examples/basic_examples/vector_scal_opencl_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
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
	starpu_memory_pin(vector, sizeof(vector));
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

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
	starpu_memory_unpin(vector, sizeof(vector));

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	ret = approximately_equal(vector[1], (1+1.0f) * factor) && approximately_equal(vector[NX-1], (NX-1+1.0f) * factor);
	FPRINTF(stderr, "[AFTER] 1-th element     : %3.2f (should be %3.2f)\n", vector[1], (1+1.0f) * factor);
	FPRINTF(stderr, "[AFTER] (NX-1)-th element: %3.2f (should be %3.2f)\n", vector[NX-1], (NX-1+1.0f) * factor);
	FPRINTF(stderr, "[AFTER] Computation is%s correct\n", ret?"":" NOT");
	return (ret ? EXIT_SUCCESS : EXIT_FAILURE);

enodev:
	return 77;
}
