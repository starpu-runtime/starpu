/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#ifndef __STARPU_OPENCL_H__
#define __STARPU_OPENCL_H__

#ifdef STARPU_USE_OPENCL
#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STARPU_OPENCL_REPORT_ERROR(status)                                     \
	do {                                                            \
		char *errormsg;                                         \
		switch (status) {                                       \
		case CL_SUCCESS:                                        \
			errormsg = "success";                           \
			break;                                          \
		case CL_DEVICE_NOT_FOUND:				\
			errormsg = "Device not found";                  \
			break;                                          \
		case CL_DEVICE_NOT_AVAILABLE:				\
			errormsg = "Device not available";              \
			break;                                          \
		case CL_COMPILER_NOT_AVAILABLE:				\
			errormsg = "Compiler not available";            \
			break;                                          \
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:			\
			errormsg = "Memory object allocation failure";  \
			break;                                          \
		case CL_OUT_OF_RESOURCES:				\
			errormsg = "Out of resources";                  \
			break;                                          \
		case CL_OUT_OF_HOST_MEMORY:				\
			errormsg = "Out of host memory";                \
			break;                                          \
		case CL_PROFILING_INFO_NOT_AVAILABLE:			\
			errormsg = "Profiling info not available";      \
			break;                                          \
		case CL_MEM_COPY_OVERLAP:				\
			errormsg = "Memory copy overlap";               \
			break;                                          \
		case CL_IMAGE_FORMAT_MISMATCH:				\
			errormsg = "Image format mismatch";             \
			break;                                          \
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:			\
			errormsg = "Image format not supported";        \
			break;                                          \
		case CL_BUILD_PROGRAM_FAILURE:				\
			errormsg = "Build program failure";             \
			break;                                          \
		case CL_MAP_FAILURE:				        \
			errormsg = "Map failure";                       \
			break;                                          \
		case CL_INVALID_VALUE:				        \
			errormsg = "Invalid value";                     \
			break;                                          \
		case CL_INVALID_DEVICE_TYPE:				\
			errormsg = "Invalid device type";               \
			break;                                          \
		case CL_INVALID_PLATFORM:				\
			errormsg = "Invalid platform";                  \
			break;                                          \
		case CL_INVALID_DEVICE:				        \
			errormsg = "Invalid device";                    \
			break;                                          \
		case CL_INVALID_CONTEXT:				\
			errormsg = "Invalid context";                   \
			break;                                          \
		case CL_INVALID_QUEUE_PROPERTIES:			\
			errormsg = "Invalid queue properties";          \
			break;                                          \
		case CL_INVALID_COMMAND_QUEUE:				\
			errormsg = "Invalid command queue";             \
			break;                                          \
		case CL_INVALID_HOST_PTR:				\
			errormsg = "Invalid host pointer";              \
			break;                                          \
		case CL_INVALID_MEM_OBJECT:				\
			errormsg = "Invalid memory object";             \
			break;                                          \
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:		\
			errormsg = "Invalid image format descriptor";   \
			break;                                          \
		case CL_INVALID_IMAGE_SIZE:				\
			errormsg = "Invalid image size";                \
			break;                                          \
		case CL_INVALID_SAMPLER:				\
			errormsg = "Invalid sampler";                   \
			break;                                          \
		case CL_INVALID_BINARY:				        \
			errormsg = "Invalid binary";                    \
			break;                                          \
		case CL_INVALID_BUILD_OPTIONS:				\
			errormsg = "Invalid build options";             \
			break;                                          \
		case CL_INVALID_PROGRAM:				\
			errormsg = "Invalid program";                   \
			break;                                          \
		case CL_INVALID_PROGRAM_EXECUTABLE:			\
			errormsg = "Invalid program executable";        \
			break;                                          \
		case CL_INVALID_KERNEL_NAME:				\
			errormsg = "Invalid kernel name";               \
			break;                                          \
		case CL_INVALID_KERNEL_DEFINITION:			\
			errormsg = "Invalid kernel definition";         \
			break;                                          \
		case CL_INVALID_KERNEL:				        \
			errormsg = "Invalid kernel";                    \
			break;                                          \
		case CL_INVALID_ARG_INDEX:				\
			errormsg = "Invalid argument index";            \
			break;                                          \
		case CL_INVALID_ARG_VALUE:				\
			errormsg = "Invalid argument value";            \
			break;                                          \
		case CL_INVALID_ARG_SIZE:				\
			errormsg = "Invalid argument size";             \
			break;                                          \
		case CL_INVALID_KERNEL_ARGS:				\
			errormsg = "Invalid kernel arguments";          \
			break;                                          \
		case CL_INVALID_WORK_DIMENSION:				\
			errormsg = "Invalid work dimension";            \
			break;                                          \
		case CL_INVALID_WORK_GROUP_SIZE:			\
			errormsg = "Invalid work group size";           \
			break;                                          \
		case CL_INVALID_WORK_ITEM_SIZE:				\
			errormsg = "Invalid work item size";            \
			break;                                          \
		case CL_INVALID_GLOBAL_OFFSET:				\
			errormsg = "Invalid global offset";             \
			break;                                          \
		case CL_INVALID_EVENT_WAIT_LIST:			\
			errormsg = "Invalid event wait list";           \
			break;                                          \
		case CL_INVALID_EVENT:				        \
			errormsg = "Invalid event";                     \
			break;                                          \
		case CL_INVALID_OPERATION:				\
			errormsg = "Invalid operation";                 \
			break;                                          \
		case CL_INVALID_GL_OBJECT:				\
			errormsg = "Invalid GL object";                 \
			break;                                          \
		case CL_INVALID_BUFFER_SIZE:				\
			errormsg = "Invalid buffer size";               \
			break;                                          \
		case CL_INVALID_MIP_LEVEL:				\
			errormsg = "Invalid MIP level";                 \
			break;                                          \
		default:						\
			errormsg = "unknown error";			\
			break;			                        \
		}                                                       \
		printf("oops in %s ... <%s> (%d) \n", __func__, errormsg, status); \
		assert(0);	                                        \
	} while (0)

struct starpu_opencl_codelet {
        cl_program programs[STARPU_MAXOPENCLDEVS];
};

void starpu_opencl_get_context(int devid, cl_context *context);
void starpu_opencl_get_device(int devid, cl_device_id *device);
void starpu_opencl_get_queue(int devid, cl_command_queue *queue);

int starpu_opencl_load_opencl_from_file(char *source_file_name, struct starpu_opencl_codelet *codelet);
int starpu_opencl_load_opencl_from_string(char *opencl_codelet_source, struct starpu_opencl_codelet *codelet);
int starpu_opencl_unload_opencl(struct starpu_opencl_codelet *codelet);

int starpu_opencl_load_kernel(cl_kernel *kernel, cl_command_queue *queue, struct starpu_opencl_codelet *codelet, char *kernel_name, int devid);
int starpu_opencl_release_kernel(cl_kernel kernel);


#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_OPENCL
#endif // __STARPU_OPENCL_H__

