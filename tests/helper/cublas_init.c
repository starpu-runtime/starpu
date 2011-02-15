/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>

struct timeval start;
struct timeval end;

//static float *data = NULL;

int main(int argc, char **argv)
{
	starpu_init(NULL);

	unsigned ngpus = starpu_cuda_worker_get_count();

	double init_timing;
	double shutdown_timing;

	gettimeofday(&start, NULL);
	starpu_helper_cublas_init();
	gettimeofday(&end, NULL);
	init_timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	gettimeofday(&start, NULL);
	starpu_helper_cublas_shutdown();
	gettimeofday(&end, NULL);
	shutdown_timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Total:\n");
	fprintf(stderr, "\tinit: %2.2f us\n", init_timing/(1000));
	fprintf(stderr, "\tshutdown: %2.2f us\n", shutdown_timing/(1000));

	if (ngpus != 0)
	{
		fprintf(stderr, "per-GPU (#gpu = %d):\n", ngpus);

		fprintf(stderr, "\tinit: %2.2f us\n", init_timing/(1000*ngpus));
		fprintf(stderr, "\tshutdown: %2.2f us\n", shutdown_timing/(1000*ngpus));
	}

	starpu_shutdown();

	return 0;
}
