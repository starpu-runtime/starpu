/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <debug/starpu_debug_helpers.h>
#include "../helper.h"

/*
 * Trigger a ping-pong test between two CUDA GPUs
 */

static size_t vector_size = 1;

#ifdef STARPU_QUICK_CHECK
static int niter = 16;
#else
static int niter = 1000;
#endif
//static unsigned cnt;

//static unsigned finished = 0;

starpu_data_handle_t v_handle;
static unsigned *v;

static char worker_0_name[128];
static char worker_1_name[128];
static unsigned memory_node_0;
static unsigned memory_node_1;

double start;
double end;

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Create a piece of data */
	ret = starpu_malloc((void **)&v, vector_size);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
	starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)v, vector_size, 1);

	/* Find a pair of memory nodes */
	if (starpu_cuda_worker_get_count() > 1)
	{
		/* Take the two devices that come first */
		int nworkers = (int)starpu_worker_get_count();

		unsigned found_node_0 = 0;

		int w;
		for (w = 0; w < nworkers; w++)
		{
			if (starpu_worker_get_type(w) == STARPU_CUDA_WORKER)
			{
				if (!found_node_0)
				{
					memory_node_0 = starpu_worker_get_memory_node(w);
					starpu_worker_get_name(w, worker_0_name, 128);
					found_node_0 = 1;
				}
				else
				{
					memory_node_1 = starpu_worker_get_memory_node(w);
					starpu_worker_get_name(w, worker_1_name, 128);
					break;
				}
			}
		}

		fprintf(stderr, "Ping-pong will be done between %s (node %u) and %s (node %u)\n",
					worker_0_name, memory_node_0, worker_1_name, memory_node_1);
	}

	//	unsigned iter;

	/* warm up */
	//	unsigned nwarmupiter = 128;
	_starpu_benchmark_ping_pong(v_handle, memory_node_0, memory_node_1, 128);

	start = starpu_timing_now();

	_starpu_benchmark_ping_pong(v_handle, memory_node_0, memory_node_1, niter);

	end = starpu_timing_now();

	double timing = end - start;

	fprintf(stderr, "Took %f ms\n", timing/1000);
	fprintf(stderr, "Avg. transfer time : %f us\n", timing/(2*niter));

	starpu_data_unregister(v_handle);
	starpu_free_noflag(v, vector_size);
	starpu_shutdown();

	return EXIT_SUCCESS;
}
