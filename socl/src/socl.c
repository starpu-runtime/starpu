/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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

struct _cl_icd_dispatch socl_master_dispatch =
{
	soclGetPlatformIDs,
	soclGetPlatformInfo,
	soclGetDeviceIDs,
	soclGetDeviceInfo,
	soclCreateContext,
	soclCreateContextFromType,
	soclRetainContext,
	soclReleaseContext,
	soclGetContextInfo,
	soclCreateCommandQueue,
	soclRetainCommandQueue,
	soclReleaseCommandQueue,
	soclGetCommandQueueInfo,
	soclSetCommandQueueProperty,
	soclCreateBuffer,
	soclCreateImage2D,
	soclCreateImage3D,
	soclRetainMemObject,
	soclReleaseMemObject,
	soclGetSupportedImageFormats,
	soclGetMemObjectInfo,
	soclGetImageInfo,
	soclCreateSampler,
	soclRetainSampler,
	soclReleaseSampler,
	soclGetSamplerInfo,
	soclCreateProgramWithSource,
	soclCreateProgramWithBinary,
	soclRetainProgram,
	soclReleaseProgram,
	soclBuildProgram,
	soclUnloadCompiler,
	soclGetProgramInfo,
	soclGetProgramBuildInfo,
	soclCreateKernel,
	soclCreateKernelsInProgram,
	soclRetainKernel,
	soclReleaseKernel,
	soclSetKernelArg,
	soclGetKernelInfo,
	soclGetKernelWorkGroupInfo,
	soclWaitForEvents,
	soclGetEventInfo,
	soclRetainEvent,
	soclReleaseEvent,
	soclGetEventProfilingInfo,
	soclFlush,
	soclFinish,
	soclEnqueueReadBuffer,
	soclEnqueueWriteBuffer,
	soclEnqueueCopyBuffer,
	soclEnqueueReadImage,
	soclEnqueueWriteImage,
	soclEnqueueCopyImage,
	soclEnqueueCopyImageToBuffer,
	soclEnqueueCopyBufferToImage,
	soclEnqueueMapBuffer,
	soclEnqueueMapImage,
	soclEnqueueUnmapMemObject,
	soclEnqueueNDRangeKernel,
	soclEnqueueTask,
	soclEnqueueNativeKernel,
	soclEnqueueMarker,
	soclEnqueueWaitForEvents,
	soclEnqueueBarrier,
	soclGetExtensionFunctionAddress,
	(void *) NULL, //  clCreateFromGLBuffer,
	(void *) NULL, //  clCreateFromGLTexture2D,
	(void *) NULL, //  clCreateFromGLTexture3D,
	(void *) NULL, //  clCreateFromGLRenderbuffer,
	(void *) NULL, //  clGetGLObjectInfo,
	(void *) NULL, //  clGetGLTextureInfo,
	(void *) NULL, //  clEnqueueAcquireGLObjects,
	(void *) NULL, //  clEnqueueReleaseGLObjects,
	(void *) NULL, //  clGetGLContextInfoKHR,
	(void *) NULL, //
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL, //  clSetEventCallback,
	(void *) NULL, //  clCreateSubBuffer,
	(void *) NULL, //  clSetMemObjectDestructorCallback,
	(void *) NULL, //  clCreateUserEvent,
	(void *) NULL, //  clSetUserEventStatus,
	(void *) NULL, //  clEnqueueReadBufferRect,
	(void *) NULL, //  clEnqueueWriteBufferRect,
	(void *) NULL, //  clEnqueueCopyBufferRect,
	(void *) NULL, //  clCreateSubDevicesEXT,
	(void *) NULL, //  clRetainDeviceEXT,
	(void *) NULL, //  clReleaseDeviceEXT,
	(void *) NULL,
	(void *) NULL, //  clCreateSubDevices,
	(void *) NULL, //  clRetainDevice,
	(void *) NULL, //  clReleaseDevice,
	(void *) NULL, //  clCreateImage,
	(void *) NULL, //  clCreateProgramWithBuiltInKernels,
	(void *) NULL, //  clCompileProgram,
	(void *) NULL, //  clLinkProgram,
	(void *) NULL, //  clUnloadPlatformCompiler,
	(void *) NULL, //  clGetKernelArgInfo,
	(void *) NULL, //  clEnqueueFillBuffer,
	(void *) NULL, //  clEnqueueFillImage,
	(void *) NULL, //  clEnqueueMigrateMemObjects,
	soclEnqueueMarkerWithWaitList, //  clEnqueueMarkerWithWaitList,
	soclEnqueueBarrierWithWaitList, //  clEnqueueBarrierWithWaitList,
	soclGetExtensionFunctionAddressForPlatform, //  clGetExtensionFunctionAddressForPlatform,
	(void *) NULL, //  clCreateFromGLTexture,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL,
	(void *) NULL
};

struct _cl_platform_id socl_platform = {&socl_master_dispatch};

const char * __attribute__ ((aligned (16))) SOCL_PROFILE = "FULL_PROFILE";
const char * __attribute__ ((aligned (16))) SOCL_VERSION = "OpenCL 1.0 SOCL Edition (0.1.0)";
const char * __attribute__ ((aligned (16))) SOCL_PLATFORM_NAME    = "SOCL Platform";
const char * __attribute__ ((aligned (16))) SOCL_VENDOR  = "Inria";
const char * __attribute__ ((aligned (16))) SOCL_PLATFORM_EXTENSIONS = "cl_khr_icd";
const char * __attribute__ ((aligned (16))) SOCL_PLATFORM_ICD_SUFFIX_KHR ="SOCL";


/* Command queues with profiling enabled
 * This allows us to disable StarPU profiling it
 * is equal to 0
 */
int __attribute__ ((aligned (16))) profiling_queue_count = 0;

struct _cl_device_id * socl_devices = NULL;
unsigned int socl_device_count = 0;
