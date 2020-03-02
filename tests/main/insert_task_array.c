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
 * Test STARPU_DATA_ARRAY
 */

void func_cpu(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *x1 = (float *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int factor;

	starpu_codelet_unpack_args(_args, &factor);

        *x0 = *x0 * factor;
        *x1 = *x1 * (float)factor;
}

struct starpu_codelet mycodelet =
{
	.modes = { STARPU_RW, STARPU_RW },
	.cpu_funcs = {func_cpu},
	.cpu_funcs_name = {"func_cpu"},
        .nbuffers = 2
};

int main(void)
{
        int x; float f;
	int factor=12;
        int i, ret;
        starpu_data_handle_t data_handles[2];

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	x = 1;
	starpu_variable_data_register(&data_handles[0], STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	f = 2.0;
	starpu_variable_data_register(&data_handles[1], STARPU_MAIN_RAM, (uintptr_t)&f, sizeof(f));

        ret = starpu_task_insert(&mycodelet,
				 STARPU_DATA_ARRAY, data_handles, 2,
				 STARPU_VALUE, &factor, sizeof(factor),
				 STARPU_PRIORITY, 1,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

        ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

enodev:
        for(i=0 ; i<2 ; i++)
	{
		starpu_data_unregister(data_handles[i]);
        }

	starpu_shutdown();

	if (ret == -ENODEV)
	{
		fprintf(stderr, "WARNING: No one can execute this task\n");
		/* yes, we do not perform the computation but we did detect that no one
		 * could perform the kernel, so this is not an error from StarPU */
		return STARPU_TEST_SKIPPED;
	}
	else
	{
		FPRINTF(stderr, "VALUES: %d %f\n", x, f);
		ret = !(x == 12 && f == 24.0);
		return ret;
	}
}
