/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdio.h>

/*
 * Check the starpu_omp_init/shutdown calls.
 */

#if !defined(STARPU_OPENMP)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
int
main (void)
{
	int ret = starpu_omp_init();
	if (ret == -EINVAL) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
	starpu_omp_shutdown();
	return 0;
}
#endif
