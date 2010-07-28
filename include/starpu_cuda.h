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

#ifndef __STARPU_CUDA_H__
#define __STARPU_CUDA_H__

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <starpu_config.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__CUDACC__) && defined(STARPU_HAVE_WINDOWS)
#define STARPU_CUBLAS_OOPS() do { \
		printf("oops  %s \n", errormsg); \
		*(int*)NULL = 0; \
	} while (0);
#else
#define STARPU_CUBLAS_OOPS() do { \
		printf("oops  in %s ... %s \n", __func__, errormsg); \
		assert(0);						\
	} while (0);
#endif

#define STARPU_CUBLAS_REPORT_ERROR(status) 					\
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
		STARPU_CUBLAS_OOPS();					\
	} while (0)

#define STARPU_CUDA_REPORT_ERROR(status) 				\
	do {								\
		const char *errormsg = cudaGetErrorString(status);	\
		STARPU_CUBLAS_OOPS();					\
	} while (0)

cudaStream_t *starpu_cuda_get_local_stream(void);

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_CUDA
#endif // __STARPU_CUDA_H__

