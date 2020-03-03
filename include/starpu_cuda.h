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

#ifndef __STARPU_CUDA_H__
#define __STARPU_CUDA_H__

#include <starpu_config.h>

#if defined STARPU_USE_CUDA && !defined STARPU_DONT_INCLUDE_CUDA_HEADERS
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_CUDA_Extensions CUDA Extensions
   @{
 */

/**
   Report a CUBLAS error.
*/
void starpu_cublas_report_error(const char *func, const char *file, int line, int status);

/**
   Call starpu_cublas_report_error(), passing the current function, file and line position.
*/
#define STARPU_CUBLAS_REPORT_ERROR(status) starpu_cublas_report_error(__starpu_func__, __FILE__, __LINE__, status)

/**
   Report a CUDA error.
*/
void starpu_cuda_report_error(const char *func, const char *file, int line, cudaError_t status);

/**
   Call starpu_cuda_report_error(), passing the current function, file and line position.
*/
#define STARPU_CUDA_REPORT_ERROR(status) starpu_cuda_report_error(__starpu_func__, __FILE__, __LINE__, status)

/**
   Return the current worker’s CUDA stream. StarPU provides a stream
   for every CUDA device controlled by StarPU. This function is only
   provided for convenience so that programmers can easily use
   asynchronous operations within codelets without having to create a
   stream by hand. Note that the application is not forced to use the
   stream provided by starpu_cuda_get_local_stream() and may also
   create its own streams. Synchronizing with
   <c>cudaDeviceSynchronize()</c> is allowed, but will reduce the
   likelihood of having all transfers overlapped.
*/
cudaStream_t starpu_cuda_get_local_stream(void);

/**
   Return a pointer to device properties for worker \p workerid
   (assumed to be a CUDA worker).
*/
const struct cudaDeviceProp *starpu_cuda_get_device_properties(unsigned workerid);

/**
   Copy \p ssize bytes from the pointer \p src_ptr on \p src_node
   to the pointer \p dst_ptr on \p dst_node. The function first tries to
   copy the data asynchronous (unless \p stream is <c>NULL</c>). If the
   asynchronous copy fails or if \p stream is <c>NULL</c>, it copies the
   data synchronously. The function returns <c>-EAGAIN</c> if the
   asynchronous launch was successfull. It returns 0 if the synchronous
   copy was successful, or fails otherwise.
*/
int starpu_cuda_copy_async_sync(void *src_ptr, unsigned src_node, void *dst_ptr, unsigned dst_node, size_t ssize, cudaStream_t stream, enum cudaMemcpyKind kind);

/**
   Copy \p numblocks blocks of \p blocksize bytes from the pointer \p src_ptr on
   \p src_node to the pointer \p dst_ptr on \p dst_node.

   The blocks start at addresses which are ld_src (resp. ld_dst) bytes apart in
   the source (resp. destination) interface.

   The function first tries to copy the data asynchronous (unless \p stream is
   <c>NULL</c>). If the asynchronous copy fails or if \p stream is <c>NULL</c>,
   it copies the data synchronously. The function returns <c>-EAGAIN</c> if the
   asynchronous launch was successfull. It returns 0 if the synchronous copy was
   successful, or fails otherwise.
*/
int starpu_cuda_copy2d_async_sync(void *src_ptr, unsigned src_node, void *dst_ptr, unsigned dst_node,
				  size_t blocksize,
				  size_t numblocks, size_t ld_src, size_t ld_dst,
				  cudaStream_t stream, enum cudaMemcpyKind kind);

/**
   Copy \p numblocks_1 * \p numblocks_2 blocks of \p blocksize bytes from the
   pointer \p src_ptr on \p src_node to the pointer \p dst_ptr on \p dst_node.

   The blocks are grouped by \p numblocks_1 blocks whose start addresses are
   ld1_src (resp. ld1_dst) bytes apart in the source (resp. destination)
   interface.

   The function first tries to copy the data asynchronous (unless \p stream is
   <c>NULL</c>). If the asynchronous copy fails or if \p stream is <c>NULL</c>,
   it copies the data synchronously. The function returns <c>-EAGAIN</c> if the
   asynchronous launch was successfull. It returns 0 if the synchronous copy was
   successful, or fails otherwise.
*/
int starpu_cuda_copy3d_async_sync(void *src_ptr, unsigned src_node, void *dst_ptr, unsigned dst_node,
				  size_t blocksize,
				  size_t numblocks_1, size_t ld1_src, size_t ld1_dst,
				  size_t numblocks_2, size_t ld2_src, size_t ld2_dst,
				  cudaStream_t stream, enum cudaMemcpyKind kind);

/**
   Call <c>cudaSetDevice(\p devid)</c> or <c>cudaGLSetGLDevice(\p devid)</c>,
   according to whether \p devid is among the field
   starpu_conf::cuda_opengl_interoperability.
*/
void starpu_cuda_set_device(unsigned devid);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_CUDA && !STARPU_DONT_INCLUDE_CUDA_HEADERS */

#endif /* __STARPU_CUDA_H__ */
