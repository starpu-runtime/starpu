/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016  Inria
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

#include <stdlib.h>
#include <stdio.h>
#include <starpu.h>

#define _FSTARPU_ERROR(msg) do {fprintf(stderr, "fstarpu error: %s\n", (msg));abort();} while(0)

const int fstarpu_r = STARPU_R;
const int fstarpu_w = STARPU_W;
const int fstarpu_rw = STARPU_RW;
const int fstarpu_scratch = STARPU_SCRATCH;
const int fstarpu_redux = STARPU_REDUX;

static const int _fstarpu_data = STARPU_R | STARPU_W | STARPU_SCRATCH | STARPU_REDUX;
const void * const fstarpu_data = &_fstarpu_data;

void fstarpu_init(void)
{
	int ret = starpu_init(NULL);
	if (ret != 0)
	{
		_FSTARPU_ERROR("starpu_init failed");
	}
}

struct starpu_codelet *fstarpu_codelet_allocate(void)
{
	struct starpu_codelet *cl = malloc(sizeof(*cl));
	memset(cl, 0, sizeof(*cl));
	return cl;
}

void fstarpu_codelet_free(struct starpu_codelet *cl)
{
	memset(cl, 0, sizeof(*cl));
	free(cl);
}

void fstarpu_codelet_add_cpu_func(struct starpu_codelet *cl, void *f_ptr)
{
	int i;
	for (i = 0; i < sizeof(cl->cpu_funcs)-1; i++)
	{
		if (cl->cpu_funcs[i] == NULL)
		{
			cl->cpu_funcs[i] = f_ptr;
			return;
		}
	}
	_FSTARPU_ERROR("fstarpu: too many cpu functions in Fortran codelet");
}

void fstarpu_codelet_add_cuda_func(struct starpu_codelet *cl, void *f_ptr)
{
	int i;
	for (i = 0; i < sizeof(cl->cuda_funcs)-1; i++)
	{
		if (cl->cuda_funcs[i] == NULL)
		{
			cl->cuda_funcs[i] = f_ptr;
			return;
		}
	}
	_FSTARPU_ERROR("fstarpu: too many cuda functions in Fortran codelet");
}

void fstarpu_codelet_add_opencl_func(struct starpu_codelet *cl, void *f_ptr)
{
	int i;
	for (i = 0; i < sizeof(cl->opencl_funcs)-1; i++)
	{
		if (cl->opencl_funcs[i] == NULL)
		{
			cl->opencl_funcs[i] = f_ptr;
			return;
		}
	}
	_FSTARPU_ERROR("fstarpu: too many opencl functions in Fortran codelet");
}

void fstarpu_codelet_add_buffer(struct starpu_codelet *cl, int mode)
{
	if  (cl->nbuffers < sizeof(cl->modes)-1)
	{
		cl->modes[cl->nbuffers] = (unsigned int)mode;
		cl->nbuffers++;
	}
	else
	{
		_FSTARPU_ERROR("fstarpu: too many buffers in Fortran codelet");
	}
}

starpu_data_handle_t fstarpu_vector_data_register(void *vector, int nx, size_t elt_size, int ram)
{
	starpu_data_handle_t handle;
	starpu_vector_data_register(&handle, ram, (uintptr_t)vector, nx, elt_size);
	return handle;
}

void * fstarpu_vector_get_ptr(void *buffers[], int i)
{
	return (void *)STARPU_VECTOR_GET_PTR(buffers[i]);
}

int fstarpu_vector_get_nx(void *buffers[], int i)
{
	return STARPU_VECTOR_GET_NX(buffers[i]);
}

void fstarpu_data_unregister(starpu_data_handle_t handle)
{
	starpu_data_unregister(handle);
}

void fstarpu_insert_task(void ***_arglist)
{
	void **arglist = *_arglist;
	int i = 0;
	int current_buffer = 0;
	struct starpu_task *task = NULL;
	struct starpu_codelet *cl = arglist[i++];
	if (cl == NULL)
	{
		_FSTARPU_ERROR("task without codelet");
	}
	task = starpu_task_create();
	task->cl = cl;
	task->name = NULL;
	task->cl_arg_free = 0;
	while (arglist[i] != NULL)
	{
		if (arglist[i] == fstarpu_data)
		{
			i++;
			starpu_data_handle_t handle = arglist[i];
			if (current_buffer >= cl->nbuffers)
			{
				_FSTARPU_ERROR("too many buffers");
			}
			STARPU_TASK_SET_HANDLE(task, handle, current_buffer);
			if (!STARPU_CODELET_GET_MODE(cl, current_buffer))
			{
				_FSTARPU_ERROR("unsupported late access mode definition");
			}
			current_buffer++;
		}
		else
		{
			_FSTARPU_ERROR("unknown/unsupported argument type");
		}
		i++;
	}
	int ret = starpu_task_submit(task);
	if (ret != 0)
	{
		_FSTARPU_ERROR("starpu_task_submit failed");
	}
}
