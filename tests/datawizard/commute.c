/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
 * Trigger various STARPU_R / STARPU_RW / STARPU_RW|COMMUTE patterns
 */

void begin(void *descr[], void *arg)
{
	(void)arg;
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*x = 0;
}

static struct starpu_codelet codelet_begin =
{
	.cpu_funcs = {begin},
	.cpu_funcs_name = {"begin"},
	.nbuffers = 1,
	.name = "begin",
};

void commute1(void *descr[], void *arg)
{
	(void)arg;
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*x = 1;
}

static struct starpu_codelet codelet_commute1 =
{
	.cpu_funcs = {commute1},
	.cpu_funcs_name = {"commute1"},
	.nbuffers = 1,
	.modes = {STARPU_RW | STARPU_COMMUTE},
	.name = "commute1",
};

void commute2(void *descr[], void *arg)
{
	(void)arg;
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*x = 2;
}

static struct starpu_codelet codelet_commute2 =
{
	.cpu_funcs = {commute2},
	.cpu_funcs_name = {"commute2"},
	.nbuffers = 1,
	.modes = {STARPU_W | STARPU_COMMUTE},
	.name = "commute2",
};

void commute3(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet codelet_commute3 =
{
	.cpu_funcs = {commute3},
	.cpu_funcs_name = {"commute3"},
	.nbuffers = 1,
	.modes = {STARPU_RW | STARPU_COMMUTE},
	.name = "commute3",
};

static struct starpu_codelet codelet_end;
void end(void *descr[], void *_args)
{
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	enum starpu_data_access_mode end_mode = *(enum starpu_data_access_mode*) _args;

	if (end_mode & STARPU_W)
		(*x)++;
}

static struct starpu_codelet codelet_end =
{
	.cpu_funcs = {end},
	.cpu_funcs_name = {"end"},
	.nbuffers = 1,
	.name = "end",
};

static int x;
static starpu_data_handle_t x_handle, f_handle;

static void test(enum starpu_data_access_mode begin_mode, enum starpu_data_access_mode end_mode, int order)
{
	struct starpu_task *begin_t, *commute1_t, *commute2_t, *end_t;
	int ret;

	codelet_begin.modes[0] = begin_mode;
	codelet_end.modes[0] = end_mode;

	begin_t = starpu_task_create();
	begin_t->cl = &codelet_begin;
	begin_t->handles[0] = x_handle;
	begin_t->use_tag = 1;
	begin_t->tag_id = (order<<20) + (begin_mode<<10) + end_mode;

	commute1_t = starpu_task_create();
	commute1_t->cl = &codelet_commute1;
	commute1_t->handles[0] = x_handle;

	commute2_t = starpu_task_create();
	commute2_t->cl = &codelet_commute2;
	commute2_t->handles[0] = x_handle;

	if (order)
		starpu_task_declare_deps_array(commute2_t, 1, &commute1_t);
	else
		starpu_task_declare_deps_array(commute1_t, 1, &commute2_t);

	end_t = starpu_task_create();
	end_t->cl = &codelet_end;
	end_t->handles[0] = x_handle;
	end_t->detach = 0;
	end_t->cl_arg = &end_mode;
	end_t->cl_arg_size = sizeof(end_mode);

	if (starpu_task_submit(begin_t) == -ENODEV)
		exit(STARPU_TEST_SKIPPED);
	if (starpu_task_submit(commute1_t) == -ENODEV)
		exit(STARPU_TEST_SKIPPED);
	if (starpu_task_submit(commute2_t) == -ENODEV)
		exit(STARPU_TEST_SKIPPED);
	starpu_task_insert(&codelet_commute3, STARPU_RW|STARPU_COMMUTE, x_handle, 0);
	if (starpu_task_submit(end_t) == -ENODEV)
		exit(STARPU_TEST_SKIPPED);

	ret = starpu_task_wait(end_t);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");
	starpu_data_acquire(x_handle, STARPU_R);
	if (x != 1 + order + !!(end_mode & STARPU_W))
		exit(EXIT_FAILURE);
	starpu_data_release(x_handle);
}

int main(int argc, char **argv)
{
        int i, ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare x */
	starpu_variable_data_register(&x_handle, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));

	for (i = 0; i <= 1; i++)
	{
		test(STARPU_R, STARPU_R, i);
		test(STARPU_W, STARPU_R, i);
		test(STARPU_W, STARPU_RW, i);
		test(STARPU_R, STARPU_RW, i);
	}

	starpu_data_unregister(x_handle);
	starpu_shutdown();
	STARPU_RETURN(0);
}
