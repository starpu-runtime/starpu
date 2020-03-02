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

#include <starpu.h>
#include "../helper.h"

/*
 * Try passing the same parameter twice, with various access modes
 */

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet codelet_R_R =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_R, STARPU_R}
};

static struct starpu_codelet codelet_R_W =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_R, STARPU_W}
};

static struct starpu_codelet codelet_R_RW =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW}
};

static struct starpu_codelet codelet_W_R =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_W, STARPU_R}
};

static struct starpu_codelet codelet_W_W =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_W, STARPU_W}
};

static struct starpu_codelet codelet_W_RW =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_W, STARPU_RW}
};

static struct starpu_codelet codelet_RW_R =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R}
};

static struct starpu_codelet codelet_RW_W =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_RW, STARPU_W}
};

static struct starpu_codelet codelet_RW_RW =
{
        .cpu_funcs = { dummy_func },
	.cpu_funcs_name = {"dummy_func"},
        .model = NULL,
        .nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW}
};

int main(int argc, char **argv)
{
	float foo = 0.0f;
	starpu_data_handle_t handle;
	int ret;
	struct starpu_task *task;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&foo, sizeof(foo));

#define SUBMIT(mode0, mode1) \
	{ \
		task = starpu_task_create();	\
		task->handles[0] = handle;	\
		task->handles[1] = handle;		 \
		enum starpu_data_access_mode smode0 = STARPU_##mode0;	\
		enum starpu_data_access_mode smode1 = STARPU_##mode0;	\
		if      (smode0 == STARPU_R && smode1 == STARPU_R)	\
			task->cl = &codelet_R_R;			\
		else if (smode0 == STARPU_R && smode1 == STARPU_W)	\
			task->cl = &codelet_R_W;			\
		else if (smode0 == STARPU_R && smode1 == STARPU_RW)	\
			task->cl = &codelet_R_RW;			\
		else if (smode0 == STARPU_W && smode1 == STARPU_R)	\
			task->cl = &codelet_W_R;			\
		else if (smode0 == STARPU_W && smode1 == STARPU_W)	\
			task->cl = &codelet_W_W;			\
		else if (smode0 == STARPU_W && smode1 == STARPU_RW)	\
			task->cl = &codelet_W_RW;			\
		else if (smode0 == STARPU_RW && smode1 == STARPU_R)	\
			task->cl = &codelet_RW_R;			\
		else if (smode0 == STARPU_RW && smode1 == STARPU_W)	\
			task->cl = &codelet_RW_W;			\
		else if (smode0 == STARPU_RW && smode1 == STARPU_RW)	\
			task->cl = &codelet_RW_RW;			\
									\
		ret = starpu_task_submit(task);				\
		if (ret == -ENODEV) goto enodev;			\
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");   \
	}

	SUBMIT(R,R);
	SUBMIT(R,W);
	SUBMIT(R,RW);
	SUBMIT(W,R);
	SUBMIT(W,W);
	SUBMIT(W,RW);
	SUBMIT(RW,R);
	SUBMIT(RW,W);
	SUBMIT(RW,RW);

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
	starpu_data_unregister(handle);
	starpu_shutdown();

        return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
