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
#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif
#include "multiformat_types.h"

static struct struct_of_arrays global_struct_of_arrays;
static  starpu_data_handle global_struct_of_arrays_handle;

static void
multiformat_scal_cpu_func(void *buffers[], void *args)
{
	struct struct_of_arrays *s;
	unsigned int n, i;

	s = STARPU_MULTIFORMAT_GET_PTR(buffers[0]);
	n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);

	for (i = 0; i < n; i++) {
		s->x[i] *= s->y[i];
	}
}

#ifdef STARPU_USE_CUDA
extern starpu_codelet cpu_to_cuda_cl;
extern starpu_codelet cuda_to_cpu_cl;
#endif

#ifdef STARPU_USE_OPENCL
extern starpu_codelet cpu_to_opencl_cl;
extern starpu_codelet opencl_to_cpu_cl;
#endif

static struct starpu_multiformat_data_interface_ops format_ops = {
#ifdef STARPU_USE_CUDA
	.cuda_elemsize = sizeof(struct point),
	.cpu_to_cuda_cl = &cpu_to_cuda_cl,
	.cuda_to_cpu_cl = &cuda_to_cpu_cl,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_elemsize = sizeof(struct point),
	.cpu_to_opencl_cl = &cpu_to_opencl_cl,
	.opencl_to_cpu_cl = &opencl_to_cpu_cl,
#endif
	.cpu_elemsize = sizeof(global_struct_of_arrays),

};

#ifdef STARPU_USE_CUDA
extern void multiformat_scal_cuda_func(void *buffers[], void *arg);
#endif
#ifdef STARPU_USE_OPENCL
extern void multiformat_scal_opencl_func(void *buffers[], void *arg);
#endif

static struct starpu_perfmodel_t conversion_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "multiformat_conversion_model"
};

static starpu_codelet  cl = {
	.where = STARPU_CUDA | STARPU_OPENCL,
	.cpu_func = multiformat_scal_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func = multiformat_scal_cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func = multiformat_scal_opencl_func,
#endif
	.nbuffers = 1,
	.conversion_model = &conversion_model
};

/*
 * Main functions 
 */
static void
init_problem_data(void)
{
	int i; 
	for (i = 0; i < N_ELEMENTS; i++) {
		global_struct_of_arrays.x[i] = 1.0f + i;
		global_struct_of_arrays.y[i] = 42.0;
	}
}

static void
register_data(void)
{
	starpu_multiformat_data_register(&global_struct_of_arrays_handle,
					 0,
					 &global_struct_of_arrays,
					 N_ELEMENTS,
					 &format_ops);
}

static void
create_and_submit_tasks(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &cl;
	task->synchronous = 1;
	task->buffers[0].handle = global_struct_of_arrays_handle;
	task->buffers[0].mode = STARPU_RW;
	task->cl_arg = NULL;
	task->cl_arg_size = 0;
	starpu_task_submit(task);

	struct starpu_task *task2 = starpu_task_create();
	task2->cl = &cl;
	task2->synchronous = 1;
	task2->buffers[0].handle = global_struct_of_arrays_handle;
	task2->buffers[0].mode = STARPU_RW;
	task2->cl_arg = NULL;
	task2->cl_arg_size = 0;
	starpu_task_submit(task2);
}

static void
unregister_data(void)
{
	starpu_data_unregister(global_struct_of_arrays_handle);
}

static void
print_it(void)
{
	int i;
	for (i = 0; i < N_ELEMENTS; i++) {
		fprintf(stderr, "(%.2f %.2f) ",
			global_struct_of_arrays.x[i],
			global_struct_of_arrays.y[i]);
	}
	fprintf(stderr, "\n");
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
struct starpu_opencl_program opencl_conversion_program;
#endif

int
main(void)
{
	starpu_init(NULL);

#ifdef STARPU_USE_OPENCL
	starpu_opencl_load_opencl_from_file("examples/basic_examples/multiformat_opencl_kernel.cl",
					    &opencl_program, NULL);
	starpu_opencl_load_opencl_from_file("examples/basic_examples/multiformat_conversion_codelets_opencl_kernel.cl", 
		&opencl_conversion_program, NULL);
#endif
	init_problem_data();

	print_it();

	register_data();

	create_and_submit_tasks();

	unregister_data();

	print_it();

#ifdef STARPU_USE_OPENCL
        starpu_opencl_unload_opencl(&opencl_program);
        starpu_opencl_unload_opencl(&opencl_conversion_program);
#endif
	starpu_shutdown();


	return 0;
}
