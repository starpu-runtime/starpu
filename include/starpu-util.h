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

#ifndef __STARPU_UTIL_H__
#define __STARPU_UTIL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <starpu_config.h>
#include <starpu-task.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define STARPU_POISON_PTR	((void *)0xdeadbeef)

#define STARPU_MIN(a,b)	((a)<(b)?(a):(b))
#define STARPU_MAX(a,b)	((a)<(b)?(b):(a))

#ifdef STARPU_NO_ASSERT
#define STARPU_ASSERT(x)	do {} while(0);
#else
#define STARPU_ASSERT(x)	assert(x)
#endif

#define STARPU_ABORT()		abort()

#define STARPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#define STARPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))

#ifdef HAVE_SYNC_FETCH_AND_ADD
#define STARPU_ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))
#define STARPU_ATOMIC_OR(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))
#else
#error __sync_fetch_and_add is not available
#endif

#ifdef USE_CUDA

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

#endif // USE_CUDA

static inline int starpu_get_env_number(const char *str)
{
	char *strval;

	strval = getenv(str);
	if (strval) {
		/* the env variable was actually set */
		unsigned val;
		char *check;

		val = (int)strtol(strval, &check, 10);
		STARPU_ASSERT(strcmp(check, "\0") == 0);

		//fprintf(stderr, "ENV %s WAS %d\n", str, val);
		return val;
	}
	else {
		/* there is no such env variable */
		//fprintf("There was no %s ENV\n", str);
		return -1;
	}
}

/* Add an event in the execution trace if FxT is enabled */
void starpu_trace_user_event(unsigned code);

/* Some helper functions for application using CUBLAS kernels */
void starpu_helper_init_cublas(void);
void starpu_helper_shutdown_cublas(void);

/* Call func(arg) on every worker matching the "where" mask (eg. CUDA|CORE to
 * execute the function on every CPUs and every CUDA devices). This function is
 * synchronous, but the different workers may execute the function in parallel.
 * */
void starpu_execute_on_each_worker(void (*func)(void *), void *arg, uint32_t where);

/* This creates (and submits) an empty task that unlocks a tag once all its
 * dependencies are fulfilled. */
void starpu_create_sync_task(starpu_tag_t sync_tag, unsigned ndeps, starpu_tag_t *deps,
				void (*callback)(void *), void *callback_arg);

#ifdef USE_CUDA
cudaStream_t *starpu_get_local_stream(void);
#endif

#ifdef __cplusplus
}
#endif

#endif // __STARPU_UTIL_H__
