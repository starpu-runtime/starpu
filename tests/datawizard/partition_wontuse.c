/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <starpu.h>
#include "../helper.h"

int main(int argc, char **argv)
{
	int ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_data_filter f =
	{
	 	.filter_func = starpu_vector_filter_block,
		.nchildren = 2
	};

	int v[10];
	memset(v, 0, 10*sizeof(int));
	starpu_data_handle_t array_handle;
	starpu_vector_data_register(&array_handle, STARPU_MAIN_RAM, (uintptr_t)&v, 10, sizeof(int));

	starpu_data_partition(array_handle, &f);
	starpu_data_wont_use(array_handle);
	starpu_data_unpartition(array_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(array_handle);
	starpu_shutdown();

	return 0;
}
