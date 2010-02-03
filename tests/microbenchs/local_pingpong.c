/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
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

static size_t vector_size = 1;

static niter = 10000;
static unsigned cnt;

static unsigned finished = 0;

starpu_data_handle v_handle;
static unsigned *v;

static char worker_0_name[128];
static char worker_1_name[128];
static uint32_t memory_node_0;
static uint32_t memory_node_1;

struct timeval start;
struct timeval end;

int main(int argc, char **argv)
{
	starpu_init(NULL);

	/* Create a piece of data */
	starpu_malloc_pinned_if_possible((void **)&v, vector_size);
	starpu_register_vector_data(&v_handle, 0, (uintptr_t)v, vector_size, 1);

	/* Find a pair of memory nodes */
	if (starpu_get_cuda_worker_count() > 1)
	{
		/* Take the two devices that come first */
		int nworkers = (int)starpu_get_worker_count();

		unsigned found_node_0 = 0;

		int w;
		for (w = 0; w < nworkers; w++)
		{
			if (starpu_get_worker_type(w) == STARPU_CUDA_WORKER)
			{
				if (!found_node_0)
				{
					memory_node_0 = starpu_get_worker_memory_node(w);
					starpu_get_worker_name(w, worker_0_name, 128);
					found_node_0 = 1;
				}
				else {
					memory_node_1 = starpu_get_worker_memory_node(w);
					starpu_get_worker_name(w, worker_1_name, 128);
					break;
				}
			}
		}

		fprintf(stderr, "Ping-pong will be done between %s (node %d) and %s (node %d)\n",
					worker_0_name, memory_node_0, worker_1_name, memory_node_1);
	}

	unsigned iter;

	/* warm up */
	unsigned nwarmupiter = 128;
	for (iter = 0; iter < nwarmupiter; iter++)
	{
		 _starpu_prefetch_data_on_node_with_mode(v_handle, memory_node_0, 0, STARPU_RW);
		 _starpu_prefetch_data_on_node_with_mode(v_handle, memory_node_1, 0, STARPU_RW);
	}

	gettimeofday(&start, NULL);

	for (iter = 0; iter < niter; iter++)
	{
		starpu_trace_user_event(0x42);
		 _starpu_prefetch_data_on_node_with_mode(v_handle, memory_node_0, 0, STARPU_RW);
		starpu_trace_user_event(0x43);
		 _starpu_prefetch_data_on_node_with_mode(v_handle, memory_node_1, 0, STARPU_RW);
	}

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 +
					(end.tv_usec - start.tv_usec));

	fprintf(stderr, "Took %f ms\n", timing/1000);
	fprintf(stderr, "Avg. transfer time : %f us\n", timing/(2*niter));

	starpu_shutdown();

	return 0;
}
