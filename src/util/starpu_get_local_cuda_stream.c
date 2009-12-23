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

#include <starpu.h>
#include <common/config.h>

#ifdef USE_CUDA
static cudaStream_t streams[STARPU_NMAXWORKERS];
static unsigned cuda_streams_are_initalized = 0;

static void init_stream_on_worker(void *arg __attribute__((unused)))
{
	cudaError_t cures;
	cures = cudaStreamCreate(starpu_helper_get_local_stream());
	if (STARPU_UNLIKELY(cures))
		CUDA_REPORT_ERROR(cures);
}

void starpu_helper_create_per_gpu_streams(void)
{
	if (!cuda_streams_are_initalized)
	{
		starpu_execute_on_each_worker(init_stream_on_worker, NULL, CUDA);
		cuda_streams_are_initalized = 1;
	}
}

cudaStream_t *starpu_helper_get_local_stream(void)
{
	int worker = starpu_get_worker_id();

	return &streams[worker];
}
#endif
