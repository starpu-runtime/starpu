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

extern void cuda_host_increment(void *descr[], void *_args);

void cpu_increment(void *descr[], void *arg)
{
	(void)arg;
	unsigned *var = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(*var) += 2;
}

static struct starpu_codelet my_codelet =
{
	.cpu_funcs = {cpu_increment},
	.cpu_funcs_name = {"cpu_increment"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_host_increment},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = { STARPU_RW },
	.nbuffers = 1
};

int main(void)
{
        starpu_data_handle_t data_handles[2];
	int x = 12;
	int y = 12;
	int ret, ret1, ret2;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&data_handles[0], STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&data_handles[1], STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(y));

	ret1 = starpu_task_insert(&my_codelet,
				  STARPU_EXECUTE_WHERE, STARPU_CPU,
				  STARPU_RW, data_handles[0],
				  0);
	if (ret1 != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret1, "starpu_task_insert");

	ret2 = starpu_task_insert(&my_codelet,
				  STARPU_EXECUTE_WHERE, STARPU_CUDA,
				  STARPU_RW, data_handles[1],
				  0);
	if (ret2 != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret2, "starpu_task_insert");

	starpu_data_unregister(data_handles[0]);
	starpu_data_unregister(data_handles[1]);

	starpu_shutdown();

	if (ret1 != -ENODEV)
	{
		if (x != 14)
			ret = 1;
		FPRINTF(stderr, "Value x = %d (expected 14)\n", x);
	}
	if (ret2 != -ENODEV)
	{
		if (y != 13)
			ret = 1;
		FPRINTF(stderr, "Value y = %d (expected 13)\n", y);
	}

	STARPU_RETURN(ret);
}
