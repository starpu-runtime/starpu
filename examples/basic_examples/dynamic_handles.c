/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void dummy_small_kernel(void *descr[], void *cl_arg)
{
	int nb_data;
	int i;

	starpu_codelet_unpack_args(cl_arg, &nb_data);
	assert(nb_data == 1);
	FPRINTF(stderr, "Number of data: %d\n", nb_data);

	for(i=0 ; i<nb_data; i++)
	{
		int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);
		assert(*val == 42);
	}
}

void dummy_big_kernel(void *descr[], void *cl_arg)
{
	int nb_data;
	int i;

	starpu_codelet_unpack_args(cl_arg, &nb_data);
	assert(nb_data == STARPU_NMAXBUFS + 1);
	FPRINTF(stderr, "Number of data: %d\n", nb_data);

	for(i=0 ; i<nb_data; i++)
	{
		int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);
		assert(*val == 42);
	}
}

static struct starpu_codelet dummy_small_cl =
{
	.cpu_funcs = {dummy_small_kernel},
	.cpu_funcs_name = {"dummy_small_kernel"},
	.modes = {STARPU_RW},
	.nbuffers = 1
};

struct starpu_codelet dummy_big_cl =
{
	.cpu_funcs = {dummy_big_kernel},
	.cpu_funcs_name = {"dummy_big_kernel"},
	.nbuffers = STARPU_NMAXBUFS+1
};

struct starpu_codelet dummy_variable_cl =
{
	.cpu_funcs = {dummy_big_kernel},
	.cpu_funcs_name = {"dummy_big_kernel"},
	.nbuffers = STARPU_VARIABLE_NBUFFERS
};

int main(void)
{
	starpu_data_handle_t handle;
	struct starpu_data_descr *descrs;
	int ret;
	int val=42;
	int i;
	struct starpu_task *task, *task2, *task3;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&val, sizeof(int));


	/* This tests a small constant number of arguments with starpu_task_submit */
	task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &dummy_small_cl;
	starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
				 STARPU_VALUE, &(task->cl->nbuffers), sizeof(task->cl->nbuffers),
				 0);
	task->dyn_handles = malloc(sizeof(*task->dyn_handles));
	task->dyn_handles[0] = handle;
	task->cl_arg_free = 1;
	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");


	/* This tests a large constant number of arguments with starpu_task_submit */
	task2 = starpu_task_create();
	task2->synchronous = 1;
	task2->cl = &dummy_big_cl;
	task2->cl_arg_free = 1;
	starpu_codelet_pack_args(&task2->cl_arg, &task2->cl_arg_size,
				 STARPU_VALUE, &(task2->cl->nbuffers), sizeof(task2->cl->nbuffers),
				 0);
	task2->dyn_handles = malloc(task2->cl->nbuffers * sizeof(*(task2->dyn_handles)));
	task2->dyn_modes = malloc(task2->cl->nbuffers * sizeof(*(task2->dyn_modes)));
	for(i=0 ; i<task2->cl->nbuffers ; i++)
	{
		task2->dyn_handles[i] = handle;
		task2->dyn_modes[i] = STARPU_RW;
	}
	ret = starpu_task_submit(task2);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");


	/* This tests a large variable number of arguments with starpu_task_submit */
	task3 = starpu_task_create();
	task3->synchronous = 1;
	task3->cl = &dummy_variable_cl;
	task3->cl_arg_free = 1;
	starpu_codelet_pack_args(&task3->cl_arg, &task3->cl_arg_size,
				 STARPU_VALUE, &(dummy_big_cl.nbuffers), sizeof(dummy_big_cl.nbuffers),
				 0);
	task3->dyn_handles = malloc(dummy_big_cl.nbuffers * sizeof(*(task3->dyn_handles)));
	task3->dyn_modes = malloc(dummy_big_cl.nbuffers * sizeof(*(task3->dyn_modes)));
	task3->nbuffers = dummy_big_cl.nbuffers;
	for(i=0 ; i<dummy_big_cl.nbuffers ; i++)
	{
		task3->dyn_handles[i] = handle;
		task3->dyn_modes[i] = STARPU_RW;
	}
	ret = starpu_task_submit(task3);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");


	/* This tests a small number of arguments with starpu_task_insert */
	ret = starpu_task_insert(&dummy_small_cl,
				 STARPU_VALUE, &(dummy_small_cl.nbuffers), sizeof(dummy_small_cl.nbuffers),
				 STARPU_RW, handle,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");


	/* This tests a large constant number of arguments with starpu_task_insert */
	descrs = malloc(dummy_big_cl.nbuffers * sizeof(struct starpu_data_descr));
	for(i=0 ; i<dummy_big_cl.nbuffers ; i++)
	{
		descrs[i].handle = handle;
		descrs[i].mode = STARPU_RW;
	}
	ret = starpu_task_insert(&dummy_big_cl,
				 STARPU_VALUE, &(dummy_big_cl.nbuffers), sizeof(dummy_big_cl.nbuffers),
				 STARPU_DATA_MODE_ARRAY, descrs, dummy_big_cl.nbuffers,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
	free(descrs);


	/* This tests a large variable number of arguments with starpu_task_insert */
	descrs = malloc(dummy_big_cl.nbuffers * sizeof(struct starpu_data_descr));
	for(i=0 ; i<dummy_big_cl.nbuffers ; i++)
	{
		descrs[i].handle = handle;
		descrs[i].mode = STARPU_RW;
	}
	ret = starpu_task_insert(&dummy_variable_cl,
				 STARPU_VALUE, &(dummy_big_cl.nbuffers), sizeof(dummy_big_cl.nbuffers),
				 STARPU_DATA_MODE_ARRAY, descrs, dummy_big_cl.nbuffers,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
	free(descrs);


	starpu_data_unregister(handle);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	starpu_shutdown();
	return 77;
}
