/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
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

static void display_worker_names(enum starpu_archtype type)
{
	unsigned nworkers = starpu_worker_get_count();

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		if (starpu_worker_get_type(i) == type)
		{
			char name[256];
			starpu_worker_get_name(i, name, 256);
			fprintf(stdout, "\t\t%s\n", name);
		}
	}
}

static void display_combined_worker(unsigned workerid)
{
	int worker_size;
	int *combined_workerid;
	starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

	fprintf(stdout, "\t\t");

	int i;
	for (i = 0; i < worker_size; i++)
	{
		char name[256];

		starpu_worker_get_name(combined_workerid[i], name, 256);
		
		fprintf(stdout, "%s\t", name);
	}

	fprintf(stdout, "\n");
}

static void display_all_combined_workers(void)
{
	unsigned ncombined_workers = starpu_combined_worker_get_count();

	if (ncombined_workers == 0)
		return;

	unsigned nworkers = starpu_worker_get_count();

	fprintf(stdout, "\t%d Combined workers\n", ncombined_workers);

	unsigned i;
	for (i = 0; i < ncombined_workers; i++)
		display_combined_worker(nworkers + i);
}

int main(int argc, char **argv)
{
	starpu_init(NULL);

	unsigned ncpu = starpu_cpu_worker_get_count();
	unsigned ncuda = starpu_cuda_worker_get_count();
	unsigned nopencl = starpu_opencl_worker_get_count();

	fprintf(stdout, "StarPU has found :\n");

	fprintf(stdout, "\t%d CPU cores\n", ncpu);
	display_worker_names(STARPU_CPU_WORKER);

	fprintf(stdout, "\t%d CUDA devices\n", ncuda);
	display_worker_names(STARPU_CUDA_WORKER);

	fprintf(stdout, "\t%d OpenCL devices\n", nopencl);
	display_worker_names(STARPU_OPENCL_WORKER);

	display_all_combined_workers();

	starpu_shutdown();

	return 0;
}
