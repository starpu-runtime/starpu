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

struct bp_data
{
	cl_program program;
	char * options;
	const cl_device_id * device_list;
	cl_uint num_devices;
};

static void soclBuildProgram_task(void *data)
{
	struct bp_data *d = (struct bp_data*)data;
	cl_device_id device;
	cl_int err;
	unsigned int i;

	int wid = starpu_worker_get_id_check();

	/* Check if the kernel has to be built for this device */
	for (i=0; i <= d->num_devices; i++)
	{
		if (i == d->num_devices)
			return;

		if (d->device_list[i]->worker_id == wid)
			break;
	}

	int range = starpu_worker_get_range();
	starpu_opencl_get_device(wid, &device);

	DEBUG_MSG("[Worker %d] Building program...\n", wid);

	cl_device_type dev_type;
	clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
	char * dev_type_str = (dev_type == CL_DEVICE_TYPE_CPU ? "CPU" :
			       dev_type == CL_DEVICE_TYPE_GPU ? "GPU" :
			       dev_type == CL_DEVICE_TYPE_ACCELERATOR ? "ACCELERATOR" : "UNKNOWN");

	char opts[4096];
	snprintf(opts, sizeof(opts), "-DSOCL_DEVICE_TYPE_%s %s",
		 dev_type_str, (d->options != NULL ? d->options : ""));

	err = clBuildProgram(d->program->cl_programs[range], 1, &device, opts, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		clGetProgramBuildInfo(d->program->cl_programs[range], device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char * buffer = malloc(len+1);
		buffer[len] = '\0';
		clGetProgramBuildInfo(d->program->cl_programs[range], device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		DEBUG_CL("clBuildProgram", err);
		ERROR_MSG("clBuildProgram: %s\n Aborting.\n", buffer);
		free(buffer);
	}

	DEBUG_MSG("[Worker %d] Done building.\n", wid);
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclBuildProgram(cl_program         program,
		 cl_uint              num_devices,
		 const cl_device_id * device_list,
		 const char *         options, 
		 void (*pfn_notify)(cl_program program, void * user_data),
		 void *               user_data)
{
	struct bp_data *data;

	program->options = options != NULL ? strdup(options) : NULL;
	program->options_size = options != NULL ? strlen(options)+1 : 0;

	data = (struct bp_data*)malloc(sizeof(struct bp_data));
	gc_entity_store(&data->program, program);
	data->options = (char*)options;

	/* If the device list is empty, we compile for every device in the context associated to the program */
	if (device_list == NULL)
	{
		num_devices = program->context->num_devices;
		device_list = program->context->devices;
	}

	data->num_devices = num_devices;
	data->device_list = device_list;

	/*FIXME: starpu_execute_on_specific_workers is synchronous.
	 * However pfn_notify is useful only because build is supposed to be asynchronous
	 */
	unsigned workers[num_devices];
	unsigned i;
	for (i=0; i<num_devices; i++)
	{
		workers[i] = device_list[i]->worker_id;
	}
	starpu_execute_on_specific_workers(soclBuildProgram_task, data, num_devices, workers, "SOCL_BUILD_PROGRAM");

	if (pfn_notify != NULL)
		pfn_notify(program, user_data);

	gc_entity_unstore(&data->program);
	free(data);

	return CL_SUCCESS;
}
