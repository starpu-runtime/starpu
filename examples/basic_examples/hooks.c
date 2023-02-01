/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define NX    21

static int check_malloc = 0;
static int check_free = 0;

int malloc_hook(unsigned dst_node, void **A, size_t dim, int flags)
{
	int ret = 0;

	*A = malloc(dim);

	if (!*A)
		ret = -ENOMEM;

	check_malloc++;

	return ret;
}

int free_hook(unsigned dst_node, void *A, size_t dim, int flags)
{
	free(A);
	check_free++;

	return 0;
}

int main(void)
{
	int* vector;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc_set_hooks(malloc_hook, free_hook);
	starpu_malloc((void **)&vector, NX*sizeof(int));
	starpu_free_noflag(vector, NX*sizeof(int));

	STARPU_ASSERT(check_malloc == 1 && check_free == 1);

	starpu_shutdown();
}
