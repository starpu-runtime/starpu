/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 inria
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
#include "custom_interface.h"
#include "custom_types.h"

#define N 20

#define DEBUG 0

static struct point array_of_structs[N];
static starpu_data_handle_t handle;
static unsigned int nchunks = 4;

#ifdef STARPU_USE_CUDA
extern struct starpu_codelet cpu_to_cuda_cl;
extern struct starpu_codelet cuda_to_cpu_cl;
#endif

static struct starpu_multiformat_data_interface_ops format_ops =
{
#ifdef STARPU_USE_CUDA
	.cuda_elemsize = sizeof(struct struct_of_arrays),
	.cpu_to_cuda_cl = &cpu_to_cuda_cl,
	.cuda_to_cpu_cl = &cuda_to_cpu_cl,
#endif
	.cpu_elemsize = sizeof(struct point),
};


static void
custom_filter(void *father, void *child, struct starpu_data_filter *f,
		unsigned id, unsigned nchunks)
{
	struct custom_data_interface *custom_father, *custom_child;
	custom_father = (struct custom_data_interface *) father;
	custom_child = (struct custom_data_interface *) child;

	assert(N % nchunks == 0); // XXX 
	ssize_t chunk_size = N/nchunks;

	if (custom_father->cpu_ptr)
	{
		struct point *tmp = (struct point *) custom_father->cpu_ptr;
		tmp += id * chunk_size;
		custom_child->cpu_ptr = tmp;
	}
#ifdef STARPU_USE_CUDA
	else if (custom_father->cuda_ptr)
	{
		struct struct_of_arrays *soa_father, *soa_child;
		soa_father = (struct struct_of_arrays*) custom_father->cuda_ptr;
		soa_child = (struct struct_of_arrays*) custom_child->cuda_ptr;
		soa_child->x = soa_father->x + chunk_size;
		soa_child->y = soa_father->y + chunk_size;
	}
#endif

	custom_child->ops = custom_father->ops;
	custom_child->nx = chunk_size;
}

static void
register_and_partition_data(void)
{
	int i;
	for (i = 0; i < N; i++)
	{
		array_of_structs[i].x = i+1.0;
		array_of_structs[i].y = 42.0;
	}
	custom_data_register(&handle, 0, &array_of_structs, N, &format_ops);

	struct starpu_data_filter f =
	{
		.filter_func   = custom_filter,
		.nchildren     = nchunks,
		.get_nchildren = NULL,
		.get_child_ops = NULL
	};
	starpu_data_partition(handle, &f);
}

static void
unpartition_and_unregister_data(void)
{
	starpu_data_unpartition(handle, 0);
	starpu_data_unregister(handle);
}

static void
custom_scal_cpu_func(void *buffers[], void *args)
{
	struct point *aos;
	unsigned int n, i;

	aos = CUSTOM_GET_CPU_PTR(buffers[0]);
	n = CUSTOM_GET_NX(buffers[0]);

	for (i = 0; i < n; i++)
		aos[i].x *= aos[i].y;
}

#ifdef STARPU_USE_CUDA
extern void custom_scal_cuda_func(void *buffers[], void *args);
#endif

static struct starpu_codelet cpu_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = { custom_scal_cpu_func, NULL},
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.name = "codelet_real"
};

#ifdef STARPU_USE_CUDA
static struct starpu_codelet cuda_cl =
{
	.where = STARPU_CUDA,
	.cuda_funcs = { custom_scal_cuda_func, NULL },
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.name = "cuda_codelet"
};
#endif /* !STARPU_USE_CUDA */

static int
create_and_submit_tasks(void)
{
	int err;
	unsigned int i;
	for (i = 0; i < nchunks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		if (i %2 == 0)
		{
			task->cl = &cpu_cl;
		}
		else
		{
#ifdef STARPU_USE_CUDA
			task->cl = &cuda_cl;
#else
			task->cl = &cpu_cl;
#endif /* !STARPU_USE_CUDA */
		}

		task->handles[0] = starpu_data_get_sub_data(handle, 1, i);
		err = starpu_task_submit(task);
		if (err != 0)
			return err;
	}


	err = starpu_task_wait_for_all();
	if (err != 0)
		return err;

	return 0;
}

#if DEBUG
static void
print_it(void)
{
	int i;
	for (i = 0; i < N; i++)
	{
		FPRINTF(stderr, "(%.2f, %.2f) ",
			array_of_structs[i].x,
			array_of_structs[i].y);
	}
	FPRINTF(stderr, "\n");
}
#endif

static int
check_it(void)
{
	int i;
	for (i = 0; i < N; i++)
	{
		float expected_value = i + 1.0;
		expected_value *= array_of_structs[i].y;
		if (array_of_structs[i].x != expected_value)
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int
main(void)
{
#ifndef STARPU_USE_CPU
	return 77;
#else
	int err;

	err = starpu_init(NULL);
	if (err == -ENODEV)
		goto enodev;


	register_and_partition_data();
#if DEBUG
	print_it();
#endif
	err = create_and_submit_tasks();
	if (err != 0)
	{
		FPRINTF(stderr, "create_submit_task : %s\n",
			strerror(-err));
		return EXIT_FAILURE;
	}
	unpartition_and_unregister_data();
#if DEBUG
	print_it();
#endif
	starpu_shutdown();		
	return check_it();


enodev:
	return 77;
#endif
}
