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

static void release_callback_context(void * e)
{
	cl_context context = (cl_context)e;

	/* Destruct object */
	if (context->properties != NULL)
		free(context->properties);

	//FIXME: should we free StarPU contexts?
	//starpu_sched_ctx_finished_submit(context->sched_ctx);

	free(context->devices);
}

static char * defaultScheduler = "dmda";
static char * defaultName = "default";

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_context CL_API_CALL
soclCreateContext(const cl_context_properties * properties,
		  cl_uint                       num_devices,
		  const cl_device_id *          devices,
		  void (*pfn_notify)(const char *, const void *, size_t, void *),
		  void *                        user_data,
		  cl_int *                      errcode_ret)
{
	if (pfn_notify == NULL && user_data != NULL)
	{
		if (errcode_ret != NULL)
			*errcode_ret = CL_INVALID_VALUE;
		return NULL;
	}

	//Check properties
	if (properties != NULL)
	{
		const cl_context_properties *p = properties;
		int i = 0;
		while (p[i] != 0)
		{
			switch (p[i])
			{
			case CL_CONTEXT_PLATFORM:
				i++;
				if (p[i] != ((cl_context_properties)&socl_platform))
				{
					if (errcode_ret != NULL)
						*errcode_ret = CL_INVALID_PLATFORM;
					return NULL;
				}
				break;

			case CL_CONTEXT_SCHEDULER_SOCL:
			case CL_CONTEXT_NAME_SOCL:
				i++;
				if (p[i] == 0)
				{
					if (errcode_ret != NULL)
						*errcode_ret = CL_INVALID_PROPERTY;
					return NULL;
				}
				break;
			}
			i++;
		}
	}

	cl_context ctx;
	ctx = (cl_context)gc_entity_alloc(sizeof(struct _cl_context), release_callback_context, "context");
	if (ctx == NULL)
	{
		if (errcode_ret != NULL)
			*errcode_ret = CL_OUT_OF_HOST_MEMORY;
		return NULL;
	}

	ctx->num_properties = 0;
	ctx->properties = NULL;

	char * sched = getenv("STARPU_SCHED");
	char * scheduler = sched == NULL ? defaultScheduler : sched;
	char * name = defaultName;

	// Properties
	if (properties != NULL)
	{
		//Count properties
		const cl_context_properties * p = properties;
		do
		{
			ctx->num_properties++;
			p++;
		} while (*p != 0);

		//Copy properties
		ctx->properties = malloc(sizeof(cl_context_properties) * ctx->num_properties);
		memcpy(ctx->properties, properties, sizeof(cl_context_properties) * ctx->num_properties);

		//Selected scheduler
		cl_uint i = 0;
		for (i=0; i<ctx->num_properties; i++)
		{
			if (p[i] == CL_CONTEXT_SCHEDULER_SOCL)
			{
				i++;
				scheduler = (char*)p[i];
			}
			if (p[i] == CL_CONTEXT_NAME_SOCL)
			{
				i++;
				name = (char*)p[i];
			}
		}
	}

	ctx->pfn_notify = pfn_notify;
	ctx->user_data = user_data;
	ctx->num_devices = num_devices;

#ifdef DEBUG
	static int id = 0;
	ctx->id = id++;
#endif

	ctx->devices = malloc(sizeof(cl_device_id) * num_devices);
	memcpy(ctx->devices, devices, sizeof(cl_device_id)*num_devices);

	// Create context
	int workers[num_devices];
	unsigned int i;
	for (i=0; i<num_devices; i++)
	{
		workers[i] = ctx->devices[i]->worker_id;
	}
	ctx->sched_ctx = starpu_sched_ctx_create(workers, num_devices, name, STARPU_SCHED_CTX_POLICY_NAME, scheduler, 0);

	if (errcode_ret != NULL)
		*errcode_ret = CL_SUCCESS;

	return ctx;
}
