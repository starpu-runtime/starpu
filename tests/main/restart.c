/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>

#include "../helper.h"

/*
 * Try initializing/shutting down starpu several times
 */

#ifdef STARPU_QUICK_CHECK
  #define N	2
#else
  #define N	10
#endif

static double start;
static double end;

int main(int argc, char **argv)
{
	unsigned iter;

	double init_timing = 0.0;
	double shutdown_timing = 0.0;
	int ret;

	for (iter = 0; iter < N; iter++)
	{
		start = starpu_timing_now();
		/* Initialize StarPU */
		ret = starpu_initialize(NULL, &argc, &argv);
		end = starpu_timing_now();
		if (ret == -ENODEV)
			goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
		init_timing += end - start;

		start = starpu_timing_now();
		/* Shutdown StarPU */
		starpu_shutdown();
		end = starpu_timing_now();
		shutdown_timing += end - start;
	}

	FPRINTF(stderr, "starpu_init: %2.2f seconds\n", init_timing/(N*1000000));
	FPRINTF(stderr, "starpu_shutdown: %2.2f seconds\n", shutdown_timing/(N*1000000));

	return EXIT_SUCCESS;

enodev:
	return STARPU_TEST_SKIPPED;
}
