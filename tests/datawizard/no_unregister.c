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

#include <starpu.h>
#include <stdlib.h>
#include <unistd.h>

#include "../helper.h"

/*
 * Check that not unregistering a data is not too crashy
 */

void dummy_func(void ** buffers, void * args)
{
	(void) buffers;
	(void) args;
}

static struct starpu_codelet dummy_cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = { dummy_func },
	.cpu_funcs_name = { "dummy_func" },
	.nbuffers = 1
};

int main(void)
{
	int ret;
	int buffer[1024];
	starpu_data_handle_t handle;
	struct starpu_task *t1,*t2;

#ifdef STARPU_HAVE_VALGRIND_H
	if(RUNNING_ON_VALGRIND) return STARPU_TEST_SKIPPED;
#endif
#ifdef STARPU_SANITIZE_LEAK
	return STARPU_TEST_SKIPPED;
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)buffer, 1024*sizeof(int));

	t1 = starpu_task_create();

	t2 = starpu_task_create();
	t2->cl = &dummy_cl;
	t2->detach = 0;
	t2->handles[0] = handle;

	starpu_task_declare_deps_array(t2, 1, &t1);

	ret = starpu_task_submit(t2);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_submit(t1);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait(t2);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

	starpu_shutdown();

	return EXIT_SUCCESS;
}
