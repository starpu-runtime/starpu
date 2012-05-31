/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#include <config.h>
#include <starpu.h>

#include "../helper.h"

static void dummy_func(void ** buffers, void * args) {
}

static struct starpu_codelet dummy_cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {fummy_func, NULL},
	.nbuffers = 1
};

int main(int argc, char **argv)
{
	int i;
	int ret;
	int buffer[1024];
	starpu_data_handle_t handle;
	struct starpu_task *t1, *t2;

        ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, 0, (uintptr_t)buffer, 1024*sizeof(int));

	t1 = starpu_task_create();
	t2 = starpu_task_create();
	t1->cl = &dummy_cl;
	t2->cl = &dummy_cl;
	t1->handles[0] = handle;
	t2->handles[0] = handle;

	starpu_task_declare_deps_array(t2, 1, &t1);

	starpu_task_submit(t2);
	starpu_data_unregister_lazy(handle);

	if (starpu_data_lookup(buffer) == NULL)
	  return EXIT_FAILURE;

	starpu_task_submit(t1);

	starpu_task_wait(t2);

	if (starpu_data_lookup(buffer) != NULL)
	  return EXIT_FAILURE;

	starpu_shutdown();

	return EXIT_SUCCESS;
}
