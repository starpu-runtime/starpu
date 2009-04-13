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

#ifndef __DRIVER_CUDA_H__
#define __DRIVER_CUDA_H__

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include <starpu.h>

#include <common/config.h>

#include <core/jobs.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>

#include <common/fxt.h>


void init_cuda(void);
void *cuda_worker(void *);

#define CUBLAS_REPORT_ERROR(status) 					\
	do {								\
		char *errormsg;						\
		switch (status) {					\
			case CUBLAS_STATUS_SUCCESS:			\
				errormsg = "success";			\
				break;					\
			case CUBLAS_STATUS_NOT_INITIALIZED:		\
				errormsg = "not initialized";		\
				break;					\
			case CUBLAS_STATUS_ALLOC_FAILED:		\
				errormsg = "alloc failed";		\
				break;					\
			case CUBLAS_STATUS_INVALID_VALUE:		\
				errormsg = "invalid value";		\
				break;					\
			case CUBLAS_STATUS_ARCH_MISMATCH:		\
				errormsg = "arch mismatch";		\
				break;					\
			case CUBLAS_STATUS_EXECUTION_FAILED:		\
				errormsg = "execution failed";		\
				break;					\
			case CUBLAS_STATUS_INTERNAL_ERROR:		\
				errormsg = "internal error";		\
				break;					\
			default:					\
				errormsg = "unknown error";		\
				break;					\
		}							\
		printf("oops  in %s ... %s \n", __func__, errormsg);	\
		assert(0);						\
	} while (0)  



#define CUDA_REPORT_ERROR(status) 					\
	do {								\
		char *errormsg;						\
		switch (status) {					\
			case CUDA_SUCCESS:				\
				errormsg = "success";			\
				break;					\
			case CUDA_ERROR_INVALID_VALUE:			\
				errormsg = "invalid value";		\
				break;					\
			case CUDA_ERROR_OUT_OF_MEMORY:			\
				errormsg = "out of memory";		\
				break;					\
			case CUDA_ERROR_NOT_INITIALIZED:		\
				errormsg = "not initialized";		\
				break;					\
			case CUDA_ERROR_DEINITIALIZED:			\
				errormsg = "deinitialized";		\
				break;					\
			case CUDA_ERROR_NO_DEVICE:			\
				errormsg = "no device";			\
				break;					\
			case CUDA_ERROR_INVALID_DEVICE:			\
				errormsg = "invalid device";		\
				break;					\
			case CUDA_ERROR_INVALID_IMAGE:			\
				errormsg = "invalid image";		\
				break;					\
			case CUDA_ERROR_INVALID_CONTEXT:		\
				errormsg = "invalid context";		\
				break;					\
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:	\
				errormsg = "context already current";	\
				break;					\
			case CUDA_ERROR_MAP_FAILED:			\
				errormsg = "map failed";		\
				break;					\
			case CUDA_ERROR_UNMAP_FAILED:			\
				errormsg = "unmap failed";		\
				break;					\
			case CUDA_ERROR_ARRAY_IS_MAPPED:		\
				errormsg = "array is mapped";		\
				break;					\
			case CUDA_ERROR_ALREADY_MAPPED:			\
				errormsg = "already mapped";		\
				break;					\
			case CUDA_ERROR_NO_BINARY_FOR_GPU:		\
				errormsg = "no binary for gpu";		\
				break;					\
			case CUDA_ERROR_ALREADY_ACQUIRED:		\
				errormsg = "already acquired";		\
				break;					\
			case CUDA_ERROR_NOT_MAPPED:			\
				errormsg = "not mapped";		\
				break;					\
			case CUDA_ERROR_INVALID_SOURCE:			\
				errormsg = "invalid source";		\
				break;					\
			case CUDA_ERROR_FILE_NOT_FOUND:			\
				errormsg = "file not found";		\
				break;					\
			case CUDA_ERROR_INVALID_HANDLE:			\
				errormsg = "invalid handle";		\
				break;					\
			case CUDA_ERROR_NOT_FOUND:			\
				errormsg = "not found";			\
				break;					\
			case CUDA_ERROR_NOT_READY:			\
				errormsg = "not ready";			\
				break;					\
			case CUDA_ERROR_LAUNCH_FAILED:			\
				errormsg = "launch failed";		\
				break;					\
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:	\
				errormsg = "launch out of resources";	\
				break;					\
			case CUDA_ERROR_LAUNCH_TIMEOUT:			\
				errormsg = "launch timeout";		\
				break;					\
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:	\
				errormsg = "launch incompatible texturing";\
				break;					\
			case CUDA_ERROR_UNKNOWN:			\
			default:					\
				errormsg = "unknown error";		\
				break;					\
		}							\
		printf("oops  in %s ... %s \n", __func__, errormsg);	\
		assert(0);						\
	} while (0)  

#endif //  __DRIVER_CUDA_H__

