/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "socl.h"

static void soclCreateKernel_task(void *data)
{
	struct _cl_kernel *k = (struct _cl_kernel *)data;

	int range = starpu_worker_get_range();
	cl_int err;

	if (k->program->cl_programs[range] == NULL)
	{
		k->errcodes[range] = CL_SUCCESS;
		DEBUG_MSG("[Device %u] Kernel creation skipped: program has not been built for this device.\n", starpu_worker_get_id_check());
		return;
	}

	DEBUG_MSG("[Device %u] Creating kernel...\n", starpu_worker_get_id_check());
	k->cl_kernels[range] = clCreateKernel(k->program->cl_programs[range], k->kernel_name, &err);
	if (err != CL_SUCCESS)
	{
		k->errcodes[range] = err;
		ERROR_STOP("[Device %u] Unable to create kernel. Error %d. Aborting.\n", starpu_worker_get_id_check(), err);
		return;
	}

	/* One worker creates argument structures */
	if (__sync_bool_compare_and_swap(&k->num_args, 0, 666))
	{
		unsigned int i;
		cl_uint num_args;

		err = clGetKernelInfo(k->cl_kernels[range], CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
		if (err != CL_SUCCESS)
		{
			DEBUG_CL("clGetKernelInfo", err);
			ERROR_STOP("Unable to get kernel argument count. Aborting.\n");
		}
		k->num_args = num_args;
		DEBUG_MSG("Kernel has %u arguments\n", num_args);

		k->arg_size = (size_t*)malloc(sizeof(size_t) * num_args);
		k->arg_value = (void**)malloc(sizeof(void*) * num_args);
		k->arg_type = (enum kernel_arg_type*)malloc(sizeof(enum kernel_arg_type) * num_args);
		/* Settings default type to NULL */
		for (i=0; i<num_args; i++)
		{
			k->arg_value[i] = NULL;
			k->arg_type[i] = Null;
		}
	}
}

static void release_callback_kernel(void * e)
{
	cl_kernel kernel = (cl_kernel)e;

	//Free args
	unsigned int i;
	for (i=0; i<kernel->num_args; i++)
	{
		switch (kernel->arg_type[i])
		{
		case Null:
		case Buffer:
			break;
		case Immediate:
			free(kernel->arg_value[i]);
			break;
		}
	}
	if (kernel->arg_size != NULL)
		free(kernel->arg_size);
	if (kernel->arg_value != NULL)
		free(kernel->arg_value);
	if (kernel->arg_type != NULL)
		free(kernel->arg_type);

	//Release real kernels...
	for (i=0; i<socl_device_count; i++)
	{
		if (kernel->cl_kernels[i] != NULL)
		{
			cl_int err = clReleaseKernel(kernel->cl_kernels[i]);
			if (err != CL_SUCCESS)
				DEBUG_CL("clReleaseKernel", err);
		}
	}

	//Release perfmodel
	//FIXME: we cannot release performance models before StarPU shutdown as it
	//will use them to store kernel execution times

	//free(kernel->perfmodel);
	//free(kernel->kernel_name);

	gc_entity_unstore(&kernel->program);

	free(kernel->cl_kernels);
	free(kernel->errcodes);
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_kernel CL_API_CALL
soclCreateKernel(cl_program    program,
		 const char *    kernel_name,
		 cl_int *        errcode_ret)
{
	cl_kernel k;

	if (program == NULL)
	{
		if (errcode_ret != NULL)
			*errcode_ret = CL_INVALID_PROGRAM;
		return NULL;
	}

	//TODO: check programs (see opencl specs)

	/* Create Kernel structure */
	k = (cl_kernel)gc_entity_alloc(sizeof(struct _cl_kernel), release_callback_kernel, "kernel");
	if (k == NULL)
	{
		if (errcode_ret != NULL)
			*errcode_ret = CL_OUT_OF_HOST_MEMORY;
		return NULL;
	}

	gc_entity_store(&k->program, program);
	k->kernel_name = strdup(kernel_name);

	k->perfmodel = malloc(sizeof(struct starpu_perfmodel));
	memset(k->perfmodel, 0, sizeof(struct starpu_perfmodel));
	k->perfmodel->type = STARPU_HISTORY_BASED;
	k->perfmodel->symbol = k->kernel_name;

	k->num_args = 0;
	k->arg_value = NULL;
	k->arg_size = NULL;

	k->split_func = NULL;
	k->split_space = 0;
	k->split_data = NULL;
	k->split_perfs = NULL;
	STARPU_PTHREAD_MUTEX_INIT(&k->split_lock, NULL);

#ifdef DEBUG
	static int id = 0;
	k->id = id++;
#endif

	k->cl_kernels = (cl_kernel*)malloc(socl_device_count * sizeof(cl_kernel));
	k->errcodes = (cl_int*)malloc(socl_device_count * sizeof(cl_int));

	{
		unsigned int i;
		for (i=0; i<socl_device_count; i++)
		{
			k->cl_kernels[i] = NULL;
			k->errcodes[i] = -9999;
		}
	}

	/* Create kernel on each device */
	DEBUG_MSG("[Kernel %d] Create %u kernels (name \"%s\")\n", k->id, socl_device_count, kernel_name);
	starpu_execute_on_each_worker_ex(soclCreateKernel_task, k, STARPU_OPENCL, "SOCL_CREATE_KERNEL");

	if (errcode_ret != NULL)
	{
		unsigned int i;
		*errcode_ret = CL_SUCCESS;
		for (i=0; i<socl_device_count; i++)
		{
			switch (k->errcodes[i])
			{
#define CASE_RET(e) case e: *errcode_ret = e; return k
				CASE_RET(CL_INVALID_PROGRAM);
				CASE_RET(CL_INVALID_PROGRAM_EXECUTABLE);
				CASE_RET(CL_INVALID_KERNEL_NAME);
				CASE_RET(CL_INVALID_KERNEL_DEFINITION);
				CASE_RET(CL_INVALID_VALUE);
				CASE_RET(CL_OUT_OF_RESOURCES);
				CASE_RET(CL_OUT_OF_HOST_MEMORY);
#undef CASE_RET
			}
		}

		if (k->num_args == 666)
		{
			*errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
			return k;
		}
	}

	return k;
}
