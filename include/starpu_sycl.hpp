/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SYCL_HPP__
#define __STARPU_SYCL_HPP__

#include <starpu_config.h>
#include <starpu_data_interfaces.h>

#ifdef STARPU_USE_SYCL
// Too much warnings from SYCL and DPCT headers
#define SYCL_DISABLE_WARNINGS _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Wall\"") \
  _Pragma("clang diagnostic ignored \"-Wshadow\"") \
  _Pragma("clang diagnostic ignored \"-Wunused\"") \
  _Pragma("clang diagnostic ignored \"-Wunused-parameter\"") \
  _Pragma("clang diagnostic ignored \"-Wsign-compare\"")
#define SYCL_RESTORE_WARNINGS _Pragma("clang diagnostic pop")

SYCL_DISABLE_WARNINGS
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
SYCL_RESTORE_WARNINGS

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_SYCL_Extensions SYCL Extensions
   @{
*/

#include <starpu_sycl.h>

/**
   Report a SYCL error.
   See \ref SYCLSupport for more details.
*/
void starpu_sycl_report_error(const char *func, const char *file, int line, dpct::err0 status);

/**
   Call starpu_sycl_report_error(), passing the current function, file and line position.
*/
#define STARPU_SYCL_REPORT_ERROR(status) starpu_sycl_report_error(__starpu_func__, __FILE__, __LINE__, status)

/**
   Return the current worker’s SYCL stream. StarPU provides a stream
   for every SYCL device controlled by StarPU. This function is only
   provided for convenience so that programmers can easily use
   asynchronous operations within codelets without having to create a
   stream by hand. Note that the application is not forced to use the
   stream provided by starpu_sycl_get_local_stream() and may also
   create its own streams. Synchronizing with
   <c>syclDeviceSynchronize()</c> is allowed, but will reduce the
   likelihood of having all transfers overlapped.
   See \ref SYCL-specificOptimizations for more details.
*/
const dpct::queue_ptr& starpu_sycl_get_local_stream(int devid = -1);

/**
   Return a pointer to device properties for worker \p workerid
   (assumed to be a SYCL worker). See \ref EnablingImplementationAccordingToCapabilities for more details.
*/
const dpct::device_info *starpu_sycl_get_device_properties(unsigned workerid);

/**
   Copy \p ssize bytes from the pointer \p src_ptr on \p src_node
   to the pointer \p dst_ptr on \p dst_node. The function first tries to
   copy the data asynchronous (unless \p stream is <c>NULL</c>). If the
   asynchronous copy fails or if \p stream is <c>NULL</c>, it copies the
   data synchronously. The function returns <c>-EAGAIN</c> if the
   asynchronous launch was successful. It returns 0 if the synchronous
   copy was successful, or fails otherwise.

   See \ref SYCLSupport for more details.
*/
int starpu_sycl_copy_async_sync(void *src_ptr, unsigned src_node, void *dst_ptr,
                                unsigned dst_node, size_t ssize,
                                dpct::queue_ptr stream,
                                dpct::memcpy_direction kind);

/**
   This is like starpu_sycl_copy_async_sync except it takes a device id and its
   kind instead of a node id.
*/
int starpu_sycl_copy_async_sync_devid(void *src_ptr, int src_dev,
                                      enum starpu_node_kind src_kind,
                                      void *dst_ptr, int dst_dev,
                                      enum starpu_node_kind dst_kind,
                                      size_t ssize, dpct::queue_ptr stream,
                                      dpct::memcpy_direction kind);

/**
   Copy \p numblocks blocks of \p blocksize bytes from the pointer \p src_ptr on
   \p src_node to the pointer \p dst_ptr on \p dst_node.

   The blocks start at addresses which are ld_src (resp. ld_dst) bytes apart in
   the source (resp. destination) interface.

   The function first tries to copy the data asynchronous (unless \p stream is
   <c>NULL</c>). If the asynchronous copy fails or if \p stream is <c>NULL</c>,
   it copies the data synchronously. The function returns <c>-EAGAIN</c> if the
   asynchronous launch was successful. It returns 0 if the synchronous copy was
   successful, or fails otherwise.

   See \ref SYCLSupport for more details.
*/
int starpu_sycl_copy2d_async_sync(void *src_ptr, unsigned src_node,
                                  void *dst_ptr, unsigned dst_node,
                                  size_t blocksize, size_t numblocks,
                                  size_t ld_src, size_t ld_dst,
                                  dpct::queue_ptr stream,
                                  dpct::memcpy_direction kind);

/**
   This is like starpu_sycl_copy2d_async_sync except it takes a device id and its
   kind instead of a node id.
*/
int starpu_sycl_copy2d_async_sync_devid(
    void *src_ptr, int src_dev, enum starpu_node_kind src_kind, void *dst_ptr,
    int dst_dev, enum starpu_node_kind dst_kind, size_t blocksize,
    size_t numblocks, size_t ld_src, size_t ld_dst, dpct::queue_ptr stream,
    dpct::memcpy_direction kind);

/**
   Copy \p numblocks_1 * \p numblocks_2 blocks of \p blocksize bytes from the
   pointer \p src_ptr on \p src_node to the pointer \p dst_ptr on \p dst_node.

   The blocks are grouped by \p numblocks_1 blocks whose start addresses are
   ld1_src (resp. ld1_dst) bytes apart in the source (resp. destination)
   interface.

   The function first tries to copy the data asynchronous (unless \p stream is
   <c>NULL</c>). If the asynchronous copy fails or if \p stream is <c>NULL</c>,
   it copies the data synchronously. The function returns <c>-EAGAIN</c> if the
   asynchronous launch was successful. It returns 0 if the synchronous copy was
   successful, or fails otherwise.

   See \ref SYCLSupport for more details.
*/
int starpu_sycl_copy3d_async_sync(void *src_ptr, unsigned src_node,
                                  void *dst_ptr, unsigned dst_node,
                                  size_t blocksize, size_t numblocks_1,
                                  size_t ld1_src, size_t ld1_dst,
                                  size_t numblocks_2, size_t ld2_src,
                                  size_t ld2_dst, dpct::queue_ptr stream,
                                  dpct::memcpy_direction kind);

/**
   This is like starpu_sycl_copy3d_async_sync except it takes a device id and
   its kind instead of a node id.
*/
int starpu_sycl_copy3d_async_sync_devid(
    void *src_ptr, int src_dev, enum starpu_node_kind src_kind, void *dst_ptr,
    int dst_dev, enum starpu_node_kind dst_kind, size_t blocksize,
    size_t numblocks_1, size_t ld1_src, size_t ld1_dst, size_t numblocks_2,
    size_t ld2_src, size_t ld2_dst, dpct::queue_ptr stream,
    dpct::memcpy_direction kind);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_SYCL */

#endif /* __STARPU_SYCL_HPP__ */
