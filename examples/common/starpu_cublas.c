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

#include "starpu_cublas.h"

#ifdef USE_CUDA
static void init_cublas_func(starpu_data_interface_t *descr, void *_args)
{
	cublasStatus cublasst = cublasInit();
	if (STARPU_UNLIKELY(cublasst))
		CUBLAS_REPORT_ERROR(cublasst);
}

static void shutdown_cublas_func(starpu_data_interface_t *descr, void *_args)
{
	cublasShutdown();
}

static starpu_codelet init_cublas_codelet = {
	.where = CUDA,
	.cuda_func = init_cublas_func,
	.model = NULL,
	.nbuffers = 0
};

static starpu_codelet shutdown_cublas_codelet = {
	.where = CUDA,
	.cuda_func = shutdown_cublas_func,
	.model = NULL,
	.nbuffers = 0
};
#endif

void init_cublas_on_all_cuda_devices(void)
{
#ifdef USE_CUDA
	unsigned worker;
	unsigned nworkers = starpu_get_worker_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		if (starpu_get_worker_type(worker) == STARPU_CUDA_WORKER)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &init_cublas_codelet;

			task->execute_on_a_specific_worker = 1;
			task->workerid = worker;

			task->synchronous = 1;
			starpu_submit_task(task);
		}
	}
#endif
}

void shutdown_cublas_on_all_cuda_devices(void)
{
#ifdef USE_CUDA
	unsigned worker;
	unsigned nworkers = starpu_get_worker_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		if (starpu_get_worker_type(worker) == STARPU_CUDA_WORKER)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &shutdown_cublas_codelet;

			task->execute_on_a_specific_worker = 1;
			task->workerid = worker;

			task->synchronous = 1;
			starpu_submit_task(task);
		}
	}
#endif
}
