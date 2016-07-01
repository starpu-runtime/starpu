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
#include <string.h>
#include <starpu.h>

#define _FSTARPU_ERROR(msg) do {fprintf(stderr, "fstarpu error: %s\n", (msg));abort();} while(0)

static const intptr_t fstarpu_r	= STARPU_R;
static const intptr_t fstarpu_w	= STARPU_W;
static const intptr_t fstarpu_rw	= STARPU_RW;
static const intptr_t fstarpu_scratch	= STARPU_SCRATCH;
static const intptr_t fstarpu_redux	= STARPU_REDUX;

static const intptr_t fstarpu_data = STARPU_R | STARPU_W | STARPU_SCRATCH | STARPU_REDUX;
static const intptr_t fstarpu_value = STARPU_VALUE;

extern void _starpu_pack_arguments(size_t *current_offset, size_t *arg_buffer_size_, char **arg_buffer_, void *ptr, size_t ptr_size);

intptr_t fstarpu_get_constant(char *s)
{
	if	(!strcmp(s, "FSTARPU_R"))	{ return fstarpu_r; }
	else if	(!strcmp(s, "FSTARPU_W"))	{ return fstarpu_w; }
	else if	(!strcmp(s, "FSTARPU_RW"))	{ return fstarpu_rw; }
	else if	(!strcmp(s, "FSTARPU_SCRATCH"))	{ return fstarpu_scratch; }
	else if	(!strcmp(s, "FSTARPU_REDUX"))	{ return fstarpu_redux; }

	else if (!strcmp(s, "FSTARPU_DATA"))	{ return fstarpu_data; }
	else if (!strcmp(s, "FSTARPU_VALUE"))	{ return fstarpu_value; }

	else { _FSTARPU_ERROR("unknown pointer constant"); }
}

struct starpu_conf *fstarpu_conf_allocate(void)
{
	struct starpu_conf *conf = malloc(sizeof(*conf));
	starpu_conf_init(conf);
	return conf;
}

void fstarpu_conf_free(struct starpu_conf *conf)
{
	memset(conf, 0, sizeof(*conf));
	free(conf);
}

void fstarpu_conf_set_sched_policy_name(struct starpu_conf *conf, const char *sched_policy_name)
{
	conf->sched_policy_name = sched_policy_name;
}

void fstarpu_conf_set_min_prio(struct starpu_conf *conf, int min_prio)
{
	conf->global_sched_ctx_min_priority = min_prio;
}

void fstarpu_conf_set_max_prio(struct starpu_conf *conf, int max_prio)
{
	conf->global_sched_ctx_max_priority = max_prio;
}

void fstarpu_conf_set_ncpu(struct starpu_conf *conf, int ncpu)
{
	STARPU_ASSERT(ncpu >= 0 && ncpu <= STARPU_NMAXWORKERS);
	conf->ncpus = ncpu;
}

void fstarpu_conf_set_ncuda(struct starpu_conf *conf, int ncuda)
{
	STARPU_ASSERT(ncuda >= 0 && ncuda <= STARPU_NMAXWORKERS);
	conf->ncuda = ncuda;
}

void fstarpu_conf_set_nopencl(struct starpu_conf *conf, int nopencl)
{
	STARPU_ASSERT(nopencl >= 0 && nopencl <= STARPU_NMAXWORKERS);
	conf->nopencl = nopencl;
}

void fstarpu_conf_set_nmic(struct starpu_conf *conf, int nmic)
{
	STARPU_ASSERT(nmic >= 0 && nmic <= STARPU_NMAXWORKERS);
	conf->nmic = nmic;
}

void fstarpu_conf_set_nscc(struct starpu_conf *conf, int nscc)
{
	STARPU_ASSERT(nscc >= 0 && nscc <= STARPU_NMAXWORKERS);
	conf->nscc = nscc;
}

void fstarpu_conf_set_calibrate(struct starpu_conf *conf, int calibrate)
{
	STARPU_ASSERT(calibrate == 0 || calibrate == 1);
	conf->calibrate = calibrate;
}

void fstarpu_conf_set_bus_calibrate(struct starpu_conf *conf, int bus_calibrate)
{
	STARPU_ASSERT(bus_calibrate == 0 || bus_calibrate == 1);
	conf->bus_calibrate = bus_calibrate;
}

struct starpu_codelet *fstarpu_codelet_allocate(void)
{
	struct starpu_codelet *cl = malloc(sizeof(*cl));
	starpu_codelet_init(cl);
	return cl;
}

void fstarpu_codelet_free(struct starpu_codelet *cl)
{
	memset(cl, 0, sizeof(*cl));
	free(cl);
}

void fstarpu_codelet_set_name(struct starpu_codelet *cl, const char *cl_name)
{
	cl->name = cl_name;
}

void fstarpu_codelet_add_cpu_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_cpu_funcs = sizeof(cl->cpu_funcs)/sizeof(cl->cpu_funcs[0])-1;
	size_t i;
	for (i = 0; i < max_cpu_funcs; i++)
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
	const size_t max_cuda_funcs = sizeof(cl->cuda_funcs)/sizeof(cl->cuda_funcs[0])-1;
	int i;
	for (i = 0; i < max_cuda_funcs; i++)
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
	const size_t max_opencl_funcs = sizeof(cl->opencl_funcs)/sizeof(cl->opencl_funcs[0])-1;
	int i;
	for (i = 0; i < max_opencl_funcs; i++)
	{
		if (cl->opencl_funcs[i] == NULL)
		{
			cl->opencl_funcs[i] = f_ptr;
			return;
		}
	}
	_FSTARPU_ERROR("fstarpu: too many opencl functions in Fortran codelet");
}

void fstarpu_codelet_add_mic_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_mic_funcs = sizeof(cl->mic_funcs)/sizeof(cl->mic_funcs[0])-1;
	int i;
	for (i = 0; i < max_mic_funcs; i++)
	{
		if (cl->mic_funcs[i] == NULL)
		{
			cl->mic_funcs[i] = f_ptr;
			return;
		}
	}
	_FSTARPU_ERROR("fstarpu: too many mic functions in Fortran codelet");
}

void fstarpu_codelet_add_scc_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_scc_funcs = sizeof(cl->scc_funcs)/sizeof(cl->scc_funcs[0])-1;
	int i;
	for (i = 0; i < max_scc_funcs; i++)
	{
		if (cl->scc_funcs[i] == NULL)
		{
			cl->scc_funcs[i] = f_ptr;
			return;
		}
	}
	_FSTARPU_ERROR("fstarpu: too many scc functions in Fortran codelet");
}

void fstarpu_codelet_add_buffer(struct starpu_codelet *cl, intptr_t mode)
{
	const size_t max_modes = sizeof(cl->modes)/sizeof(cl->modes[0])-1;
	if (mode !=  fstarpu_r
		&& mode != fstarpu_rw
		&& mode != fstarpu_w
		&& mode != fstarpu_scratch
		&& mode != fstarpu_redux)
	{
		_FSTARPU_ERROR("fstarpu: invalid data mode");
	}
	if  (cl->nbuffers < max_modes)
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

starpu_data_handle_t fstarpu_matrix_data_register(void *matrix, int ldy, int ny, int nx, size_t elt_size, int ram)
{
	starpu_data_handle_t handle;
	/*
	 * Fortran arrays are transposed with respect to C arrays. For convenience, we exchange
	 * the parameters as follows:
	 * C ldx = Fortran ldy
	 * C nx  = Fortran ny
	 * C ny  = Fortran nx
	 */
	starpu_matrix_data_register(&handle, ram, (uintptr_t)matrix, ldy, ny, nx, elt_size);
	return handle;
}

void * fstarpu_matrix_get_ptr(void *buffers[], int i)
{
	return (void *)STARPU_MATRIX_GET_PTR(buffers[i]);
}

int fstarpu_matrix_get_ld(void *buffers[], int i)
{
	/* Fortran ldy is C ldx */
	return STARPU_MATRIX_GET_LD(buffers[i]);
}

int fstarpu_matrix_get_ny(void *buffers[], int i)
{
	/* Fortran ny is C nx */
	return STARPU_MATRIX_GET_NX(buffers[i]);
}

int fstarpu_matrix_get_nx(void *buffers[], int i)
{
	/* Fortran nx is C ny */
	return STARPU_MATRIX_GET_NY(buffers[i]);
}

void fstarpu_data_acquire(starpu_data_handle_t handle, intptr_t mode)
{
	STARPU_ASSERT(mode == fstarpu_r || mode == fstarpu_w || mode == fstarpu_rw);
	starpu_data_acquire(handle, (int)mode);
}

void fstarpu_unpack_arg(char *cl_arg, void ***_buffer_list)
{
	void **buffer_list = *_buffer_list;
	size_t current_arg_offset = 0;
	int nargs, arg;

	/* We fill the different pointers with the appropriate arguments */
	memcpy(&nargs, cl_arg, sizeof(nargs));
	current_arg_offset += sizeof(nargs);

	for (arg = 0; arg < nargs; arg++)
	{
		void *argptr = buffer_list[arg];

		/* If not reading all cl_args */
		if(argptr == NULL)
			break;

		size_t arg_size;
		memcpy(&arg_size, cl_arg+current_arg_offset, sizeof(arg_size));
		current_arg_offset += sizeof(arg_size);

		memcpy(argptr, cl_arg+current_arg_offset, arg_size);
		current_arg_offset += arg_size;
	}
	free(cl_arg);
}

void fstarpu_insert_task(void ***_arglist)
{
	void **arglist = *_arglist;
	int i = 0;
	char *arg_buffer_ = NULL;
	size_t arg_buffer_size_ = 0;
	size_t current_offset = sizeof(int);
	int current_buffer = 0;
	int nargs = 0;
	struct starpu_task *task = NULL;
	struct starpu_codelet *cl = arglist[i++];
	if (cl == NULL)
	{
		_FSTARPU_ERROR("task without codelet");
	}
	task = starpu_task_create();
	task->cl = cl;
	task->name = NULL;
	while (arglist[i] != NULL)
	{
		const intptr_t arg_type = (intptr_t)arglist[i];
		if (arg_type == fstarpu_data
			|| arg_type == fstarpu_r
			|| arg_type == fstarpu_rw
			|| arg_type == fstarpu_w
			|| arg_type == fstarpu_scratch
			|| arg_type == fstarpu_redux)
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
		else if (arg_type == fstarpu_value)
		{
			i++;
			void *ptr = arglist[i];
			i++;
			size_t ptr_size = (size_t)(intptr_t)arglist[i];
			nargs++;
			_starpu_pack_arguments(&current_offset, &arg_buffer_size_, &arg_buffer_, ptr, ptr_size);
		}
		else
		{
			_FSTARPU_ERROR("unknown/unsupported argument type");
		}
		i++;
	}

	if (nargs)
	{
		memcpy(arg_buffer_, (int *)&nargs, sizeof(nargs));
		task->cl_arg = arg_buffer_;
		task->cl_arg_size = arg_buffer_size_;
	}
	else
	{
		free(arg_buffer_);
		arg_buffer_ = NULL;
	}

	int ret = starpu_task_submit(task);
	if (ret != 0)
	{
		_FSTARPU_ERROR("starpu_task_submit failed");
	}
}
