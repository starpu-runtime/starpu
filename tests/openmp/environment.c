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
#include <stdlib.h>
#include <stdio.h>

/*
 * Check OpenMP environment variables are properly parsed.
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
	setenv("OMP_DYNAMIC","false", 1);
	setenv("OMP_NESTED","false", 1);
	setenv("OMP_SCHEDULE","auto", 1);
	setenv("OMP_STACKSIZE","2M", 1);
	setenv("OMP_WAIT_POLICY","passive", 1);
	setenv("OMP_THREAD_LIMIT","0", 1);
	setenv("OMP_MAX_ACTIVE_LEVELS","4", 1);
	setenv("OMP_CANCELLATION","false", 1);
	setenv("OMP_DEFAULT_DEVICE","0", 1);
	setenv("OMP_MAX_TASK_PRIORITY", "20", 1);
	setenv("OMP_PROC_BIND","spread, spread, close", 1);
	setenv("OMP_NUM_THREADS","4, 16, 2", 1);
	setenv("OMP_PLACES","{1,2,3,4},{5,6,7,8}", 1);
	setenv("OMP_DISPLAY_ENV","verbose", 1);
	int ret = starpu_omp_init();
	if (ret == -EINVAL) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
	starpu_omp_shutdown();
	return 0;
}
#endif
