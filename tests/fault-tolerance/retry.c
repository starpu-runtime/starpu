/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2013,2015,2017                      CNRS
 * Copyright (C) 2017                                     Inria
 * Copyright (C) 2019                                     Université de Bordeaux
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

static int retry;
void cpu_increment(void *descr[], void *arg)
{
	(void)arg;
	unsigned *var = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *var2 = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	FPRINTF(stderr,"computing\n");
	*var2 = *var + 1;
	if (retry <= 10)
	{
		FPRINTF(stderr,"failing\n");
		retry++;
		/* Fake failure */
		starpu_task_failed(starpu_task_get_current());
	}
}

static struct starpu_codelet my_codelet =
{
	.cpu_funcs = {cpu_increment},
	.cpu_funcs_name = {"cpu_increment"},
	.modes = { STARPU_R, STARPU_W },
	.nbuffers = 2
};

int main(void)
{
	int x = 12;
	int y = 1;
        starpu_data_handle_t h_x, h_y;
	int ret, ret1;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&h_x, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&h_y, STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(y));

	retry = 0;
	ret1 = starpu_task_insert(&my_codelet,
				  STARPU_R, h_x,
				  STARPU_W, h_y,
				  0);
	if (ret1 != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret1, "starpu_task_insert");
	starpu_task_wait_for_all();

	starpu_data_unregister(h_x);
	starpu_data_unregister(h_y);

	starpu_shutdown();

	if (x != 12)
		ret = 1;
	FPRINTF(stderr, "Value x = %d (expected 12)\n", x);

	if (ret1 != -ENODEV)
	{
		if (y != 13)
			ret = 1;
		FPRINTF(stderr, "Value y = %d (expected 13)\n", y);
	}

	STARPU_RETURN(ret);
}
