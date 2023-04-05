/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../variable/increment.h"

void cpu_increment(void *descr[], void *arg)
{
	(void)arg;
	unsigned *var = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(*var) += 2;
}

int main(void)
{
        starpu_data_handle_t data_handles[2];
	int x = 12;
	int y = 12;
	int ret, ret1, ret2;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&data_handles[0], STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&data_handles[1], STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(y));

	// We change the cpu function to have a different computation
	increment_cl.cpu_funcs[0] = cpu_increment;

	ret1 = starpu_task_insert(&increment_cl,
				  STARPU_EXECUTE_WHERE, STARPU_CPU,
				  STARPU_RW, data_handles[0],
				  0);
	if (ret1 != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret1, "starpu_task_insert");

	ret2 = starpu_task_insert(&increment_cl,
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
