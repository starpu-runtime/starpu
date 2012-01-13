/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 INRIA
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
 * This is a really simple example intended to show how to use filters with the
 * multiformat interface. It does not do anything really useful. Since the
 * memory is not contiguous (cf. struct struct_of_arrays), the user must write
 * its own copy functions. Some of them have not been implemented here
 * (synchronous functions, for example).
 */

#include <starpu.h>
#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif
#include "multiformat_types.h"

#ifndef STARPU_USE_CPU
/* Without the CPU, there is no point in using the multiformat
 * interface, so this test is pointless. */
int
main(void)
{
	return 77;
}
#else

static int ncpu = 0;
static int ncuda = 0;
static int nopencl = 0;
static unsigned int nchunks = 1;

static struct point array_of_structs[N_ELEMENTS];
static starpu_data_handle_t array_of_structs_handle;

static void
multiformat_scal_cpu_func(void *buffers[], void *args)
{
	struct point *aos;
	unsigned int n, i;

	aos = STARPU_MULTIFORMAT_GET_PTR(buffers[0]);
	n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);

	for (i = 0; i < n; i++)
		aos[i].x *= aos[i].y;
}

#ifdef STARPU_USE_CUDA
extern struct starpu_codelet cpu_to_cuda_cl;
extern struct starpu_codelet cuda_to_cpu_cl;
#endif

#ifdef STARPU_USE_OPENCL
extern struct starpu_codelet cpu_to_opencl_cl;
extern struct starpu_codelet opencl_to_cpu_cl;
#endif

extern struct starpu_data_copy_methods my_multiformat_copy_data_methods_s;
static struct starpu_multiformat_data_interface_ops format_ops =
{
#ifdef STARPU_USE_CUDA
	.cuda_elemsize = sizeof(struct struct_of_arrays),
	.cpu_to_cuda_cl = &cpu_to_cuda_cl,
	.cuda_to_cpu_cl = &cuda_to_cpu_cl,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_elemsize = sizeof(struct struct_of_arrays),
	.cpu_to_opencl_cl = &cpu_to_opencl_cl,
	.opencl_to_cpu_cl = &opencl_to_cpu_cl,
#endif
	.cpu_elemsize = sizeof(struct point),
	.copy = &my_multiformat_copy_data_methods_s
};

#ifdef STARPU_USE_CUDA
extern void multiformat_scal_cuda_func(void *buffers[], void *arg);
#endif
#ifdef STARPU_USE_OPENCL
extern void multiformat_scal_opencl_func(void *buffers[], void *arg);
#endif

static struct starpu_codelet cpu_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = {multiformat_scal_cpu_func, NULL},
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.name = "codelet_real"
};

#ifdef STARPU_USE_CUDA
static struct starpu_codelet cuda_cl =
{
	.where = STARPU_CUDA,
	.cuda_funcs = { multiformat_scal_cuda_func, NULL },
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.name = "cuda_codelet"
};
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static struct starpu_codelet opencl_cl =
{
	.where = STARPU_OPENCL,
	.opencl_funcs = { multiformat_scal_opencl_func, NULL },
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.name = "opencl_codelet"
};
#endif /* !STARPU_USE_OPENCL */

/*
 * Main functions 
 */
static void
init_problem_data(void)
{
	int i; 
	for (i = 0; i < N_ELEMENTS; i++)
	{
		array_of_structs[i].x = 1.0 + i;
		array_of_structs[i].y = 42.0;
	}
}

static void
register_data(void)
{
	starpu_multiformat_data_register(&array_of_structs_handle,
					 0,
					 &array_of_structs,
					 N_ELEMENTS,
					 &format_ops);
}

static void
unregister_data(void)
{
	starpu_data_unregister(array_of_structs_handle);
}

static void
multiformat_divide_in_equal_chunks_filter_func(void *father,
					       void *child,
					       struct starpu_data_filter *f,
					       unsigned id,
					       unsigned nchunks)
{
	/*
	 * One chunk for a CPU device.
	 * At least one for a GPU (CUDA or OpenCL).
	 * If possible, a third chunk for another kind of GPU.
	 */ 
	assert(nchunks == 2 || nchunks == 3);
	assert (id < nchunks);

	struct starpu_multiformat_interface *mf_father, *mf_child;

	mf_father = (struct starpu_multiformat_interface *) father;
	mf_child = (struct starpu_multiformat_interface *) child;

	uint32_t length_first = f->filter_arg;
	uint32_t nx = mf_father->nx;

	assert(length_first < nx);

	mf_child->ops = mf_father->ops;
	memcpy(mf_child->ops, mf_father->ops, sizeof(mf_child->ops));


	/* The actual partitioning */
	mf_child->nx = length_first;

	if (mf_father->cpu_ptr)
	{
		struct point *tmp = (struct point *) mf_father->cpu_ptr;
		tmp += id * length_first;
		mf_child->cpu_ptr = tmp;
	}
}

static void
partition_data(void)
{
	struct starpu_data_filter f =
	{
		.filter_func = multiformat_divide_in_equal_chunks_filter_func,
		.nchildren = nchunks,
		.get_nchildren = NULL,
		.get_child_ops = NULL,
		.filter_arg = N_ELEMENTS/nchunks
	};

	starpu_data_partition(array_of_structs_handle, &f);
}

static int
create_and_submit_tasks(void)
{
	int err;
	unsigned int i;
	for (i = 0; i < nchunks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		if (i == 0)
		{
			task->cl = &cpu_cl;
		}
		else
		if (i == 1)
		{
#ifdef STARPU_USE_CUDA
			if (ncuda > 0)
				task->cl = &cuda_cl;
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
			if (ncuda == 0 && nopencl > 0)
				task->cl = &opencl_cl;
#endif /* !STARPU_USE_OPENCL */
		}
#ifdef STARPU_USE_OPENCL
		else /* i == 2 */
		{
			task->cl = &opencl_cl;
		}
#endif /* !STARPU_USE_OPENCL */

		starpu_data_handle_t handle;
		handle = starpu_data_get_sub_data(array_of_structs_handle, 1, i);
		task->handles[0] = handle;

		err = starpu_task_submit(task);
		if (err != 0)
			return err;
	}


	err = starpu_task_wait_for_all();
	if (err != 0)
		return err;

	return 0;
}

static void
print_it(void)
{
	int i;
	for (i = 0; i < N_ELEMENTS; i++)
	{
		FPRINTF(stderr, "(%.2f %.2f) ",
			array_of_structs[i].x,
			array_of_structs[i].y);
	}
	FPRINTF(stderr, "\n");
}

static int
check_it(void)
{
	int i;
	for (i = 0; i < N_ELEMENTS; i++)
	{
		float expected_value = i + 1.0;
		expected_value *= array_of_structs[i].y;
		if (array_of_structs[i].x != expected_value)
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
struct starpu_opencl_program opencl_conversion_program;
#endif

static int
gpus_available()
{
#ifdef STARPU_USE_CUDA
	if (ncuda > 0)
		return 1;
#endif
#ifdef STARPU_USE_OPENCL
	if (nopencl > 0)
		return 1;
#endif

	return 0;
}

int
main(void)
{
	int err;
	struct starpu_conf conf =
	{
		.ncpus = -1,
		.ncuda = 1,
		.nopencl = 1
	};
	starpu_init(&conf);

	ncpu = starpu_cpu_worker_get_count();
#ifdef STARPU_USE_CUDA
	ncuda = starpu_cuda_worker_get_count();
#endif
#ifdef STARPU_USE_OPENCL
	nopencl = starpu_opencl_worker_get_count();
#endif

	if (ncpu == 0 || !gpus_available())
		return 77;

	if (ncuda > 0)
		nchunks++;
	if (nopencl > 0)
		nchunks++;

	/* For the sake of simplicity. */
	assert(N_ELEMENTS % nchunks == 0);

#ifdef STARPU_USE_OPENCL
	err = starpu_opencl_load_opencl_from_file("examples/filters/multiformat/opencl.cl",
					    &opencl_program, NULL);
	assert(err == 0);
#endif
	init_problem_data();

	print_it();

	register_data();
	partition_data();

	err = create_and_submit_tasks();
	if (err != 0)
	{
		FPRINTF(stderr, "create_submit_task : %s\n",
			strerror(-err));
	}

	starpu_data_unpartition(array_of_structs_handle, 0);

	unregister_data();

	print_it();

#ifdef STARPU_USE_OPENCL
        assert(starpu_opencl_unload_opencl(&opencl_program) == CL_SUCCESS);
#endif
	starpu_shutdown();


	return check_it();
}

#endif /* STARPU_USE_CPU */
