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

#include <errno.h>

#include <core/workers.h>
#include <common/config.h>
#include <starpu.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

/* This method is not optimal at all, but it makes life much easier in many codes */

#ifdef USE_CUDA
struct data_interface_s;

struct malloc_pinned_codelet_struct {
	void **ptr;
	size_t dim;
};

static void malloc_pinned_codelet(struct data_interface_s *buffers __attribute__((unused)), void *arg)
{
	struct malloc_pinned_codelet_struct *s = arg;

	cuMemAllocHost((void **)(s->ptr), s->dim);
}
#endif

void starpu_malloc_pinned_if_possible(void **A, size_t dim)
{
	if (may_submit_cuda_task())
	{
#ifdef USE_CUDA
		int push_res;
	
		struct malloc_pinned_codelet_struct s = {
			.ptr = A,
			.dim = dim
		};	
	
		starpu_codelet *cl = malloc(sizeof(starpu_codelet));
			cl->cublas_func = malloc_pinned_codelet; 
			cl->where = CUBLAS;
			cl->model = NULL;
			cl->nbuffers = 0;
	
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL; 
			task->cl = cl;
			task->cl_arg = &s;

		task->synchronous = 1;
	
		push_res = starpu_submit_task(task);
		STARPU_ASSERT(push_res != -ENODEV);

		free(cl);
#endif
	}
	else {
		*A = malloc(dim);
	}
}
