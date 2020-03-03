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

#ifndef __STARPU_OPENCL_H__
#define __STARPU_OPENCL_H__

#include <starpu_config.h>
#ifdef STARPU_USE_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 100
#endif
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <assert.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_OpenCL_Extensions OpenCL Extensions
   @{
*/

/**
   Store the OpenCL programs as compiled for the different OpenCL
   devices.
*/
struct starpu_opencl_program
{
	/** Store each program for each OpenCL device. */
	cl_program programs[STARPU_MAXOPENCLDEVS];
};

/**
   @name Writing OpenCL kernels
   @{
*/

/**
   Return the OpenCL context of the device designated by \p devid
   in \p context.
*/
void starpu_opencl_get_context(int devid, cl_context *context);

/**
   Return the cl_device_id corresponding to \p devid in \p device.
*/
void starpu_opencl_get_device(int devid, cl_device_id *device);

/**
   Return the command queue of the device designated by \p devid
   into \p queue.
*/
void starpu_opencl_get_queue(int devid, cl_command_queue *queue);

/**
   Return the context of the current worker.
*/
void starpu_opencl_get_current_context(cl_context *context);

/**
   Return the computation kernel command queue of the current
   worker.
*/
void starpu_opencl_get_current_queue(cl_command_queue *queue);

/**
   Set the arguments of a given kernel. The list of arguments
   must be given as <c>(size_t size_of_the_argument, cl_mem *
   pointer_to_the_argument)</c>. The last argument must be 0. Return the
   number of arguments that were successfully set. In case of failure,
   return the id of the argument that could not be set and \p err is set to
   the error returned by OpenCL. Otherwise, return the number of
   arguments that were set.

   Here an example:
   \code{.c}
   int n;
   cl_int err;
   cl_kernel kernel;
   n = starpu_opencl_set_kernel_args(&err, 2, &kernel, sizeof(foo), &foo, sizeof(bar), &bar, 0);
   if (n != 2) fprintf(stderr, "Error : %d\n", err);
   \endcode
*/
int starpu_opencl_set_kernel_args(cl_int *err, cl_kernel *kernel, ...);

/** @} */

/**
   @name Compiling OpenCL kernels
   Source codes for OpenCL kernels can be stored in a file or in a
   string. StarPU provides functions to build the program executable for
   each available OpenCL device as a cl_program object. This program
   executable can then be loaded within a specific queue as explained in
   the next section. These are only helpers, Applications can also fill a
   starpu_opencl_program array by hand for more advanced use (e.g.
   different programs on the different OpenCL devices, for relocation
   purpose for instance).
   @{
*/

/**
   Store the contents of the file \p source_file_name in the buffer
   \p opencl_program_source. The file \p source_file_name can be located in the
   current directory, or in the directory specified by the environment
   variable \ref STARPU_OPENCL_PROGRAM_DIR, or
   in the directory <c>share/starpu/opencl</c> of the installation
   directory of StarPU, or in the source directory of StarPU. When the
   file is found, \p located_file_name is the full name of the file as it
   has been located on the system, \p located_dir_name the directory
   where it has been located. Otherwise, they are both set to the empty
   string.
*/
void starpu_opencl_load_program_source(const char *source_file_name, char *located_file_name, char *located_dir_name, char *opencl_program_source);

/**
   Similar to function starpu_opencl_load_program_source() but
   allocate the buffers \p located_file_name, \p located_dir_name and
   \p opencl_program_source.
*/
void starpu_opencl_load_program_source_malloc(const char *source_file_name, char **located_file_name, char **located_dir_name, char **opencl_program_source);

/**
   Compile the OpenCL kernel stored in the file \p source_file_name
   with the given options \p build_options and store the result in the
   directory <c>$STARPU_HOME/.starpu/opencl</c> with the same filename as
   \p source_file_name. The compilation is done for every OpenCL device,
   and the filename is suffixed with the vendor id and the device id of
   the OpenCL device.
*/
int starpu_opencl_compile_opencl_from_file(const char *source_file_name, const char *build_options);

/**
   Compile the OpenCL kernel in the string \p opencl_program_source
   with the given options \p build_options and store the result in the
   directory <c>$STARPU_HOME/.starpu/opencl</c> with the filename \p
   file_name. The compilation is done for every OpenCL device, and the
   filename is suffixed with the vendor id and the device id of the
   OpenCL device.
*/
int starpu_opencl_compile_opencl_from_string(const char *opencl_program_source, const char *file_name, const char *build_options);

/**
   Compile the binary OpenCL kernel identified with \p kernel_id.
   For every OpenCL device, the binary OpenCL kernel will be loaded from
   the file
   <c>$STARPU_HOME/.starpu/opencl/\<kernel_id\>.\<device_type\>.vendor_id_\<vendor_id\>_device_id_\<device_id\></c>.
*/
int starpu_opencl_load_binary_opencl(const char *kernel_id, struct starpu_opencl_program *opencl_programs);

/**
   Compile an OpenCL source code stored in a file.
*/
int starpu_opencl_load_opencl_from_file(const char *source_file_name, struct starpu_opencl_program *opencl_programs, const char *build_options);
/**
   Compile an OpenCL source code stored in a string.
 */
int starpu_opencl_load_opencl_from_string(const char *opencl_program_source, struct starpu_opencl_program *opencl_programs, const char *build_options);

/**
   Unload an OpenCL compiled code.
*/
int starpu_opencl_unload_opencl(struct starpu_opencl_program *opencl_programs);

/** @} */

/**
   @name Loading OpenCL kernels
   @{
*/

/**
   Create a kernel \p kernel for device \p devid, on its computation
   command queue returned in \p queue, using program \p opencl_programs
   and name \p kernel_name.
*/
int starpu_opencl_load_kernel(cl_kernel *kernel, cl_command_queue *queue, struct starpu_opencl_program *opencl_programs, const char *kernel_name, int devid);

/**
   Release the given \p kernel, to be called after kernel execution.
*/
int starpu_opencl_release_kernel(cl_kernel kernel);

/** @} */

/**
   @name OpenCL Statistics
   @{
*/

/**
   Collect statistics on a kernel execution.
   After termination of the kernels, the OpenCL codelet should call this
   function with the event returned by \c clEnqueueNDRangeKernel(), to
   let StarPU collect statistics about the kernel execution (used cycles,
   consumed energy).
*/
int starpu_opencl_collect_stats(cl_event event);

/** @} */

/**
   @name OpenCL Utilities
   @{
*/

/**
   Return the error message in English corresponding to \p status, an OpenCL
   error code.
*/
const char *starpu_opencl_error_string(cl_int status);

/**
   Given a valid error status, print the corresponding error message on
   \c stdout, along with the function name \p func, the filename
   \p file, the line number \p line and the message \p msg.
*/
void starpu_opencl_display_error(const char *func, const char *file, int line, const char *msg, cl_int status);

/**
   Call the function starpu_opencl_display_error() with the error
   \p status, the current function name, current file and line number,
   and a empty message.
*/
#define STARPU_OPENCL_DISPLAY_ERROR(status) starpu_opencl_display_error(__starpu_func__, __FILE__, __LINE__, NULL, status)

/**
   Call the function starpu_opencl_display_error() and abort.
*/
static __starpu_inline void starpu_opencl_report_error(const char *func, const char *file, int line, const char *msg, cl_int status)
{
	starpu_opencl_display_error(func, file, line, msg, status);
	assert(0);
}

/**
   Call the function starpu_opencl_report_error() with the error \p
   status, the current function name, current file and line number,
   and a empty message.
*/
#define STARPU_OPENCL_REPORT_ERROR(status) starpu_opencl_report_error(__starpu_func__, __FILE__, __LINE__, NULL, status)

/**
   Call the function starpu_opencl_report_error() with \p msg
   and \p status, the current function name, current file and line number.
*/
#define STARPU_OPENCL_REPORT_ERROR_WITH_MSG(msg, status) starpu_opencl_report_error(__starpu_func__, __FILE__, __LINE__, msg, status)

/**
   Allocate \p size bytes of memory, stored in \p addr. \p flags must be a valid
   combination of \c cl_mem_flags values.
*/
cl_int starpu_opencl_allocate_memory(int devid, cl_mem *addr, size_t size, cl_mem_flags flags);

/**
   Copy \p size bytes from the given \p ptr on RAM \p src_node to the
   given \p buffer on OpenCL \p dst_node. \p offset is the offset, in
   bytes, in \p buffer. if \p event is <c>NULL</c>, the copy is
   synchronous, i.e the queue is synchronised before returning. If not
   <c>NULL</c>, \p event can be used after the call to wait for this
   particular copy to complete. This function returns <c>CL_SUCCESS</c>
   if the copy was successful, or a valid OpenCL error code otherwise.
   The integer pointed to by \p ret is set to <c>-EAGAIN</c> if the
   asynchronous launch was successful, or to 0 if \p event was
   <c>NULL</c>.
*/
cl_int starpu_opencl_copy_ram_to_opencl(void *ptr, unsigned src_node, cl_mem buffer, unsigned dst_node, size_t size, size_t offset, cl_event *event, int *ret);

/**
   Copy \p size bytes asynchronously from the given \p buffer on OpenCL
   \p src_node to the given \p ptr on RAM \p dst_node. \p offset is the
   offset, in bytes, in \p buffer. if \p event is <c>NULL</c>, the copy
   is synchronous, i.e the queue is synchronised before returning. If not
   <c>NULL</c>, \p event can be used after the call to wait for this
   particular copy to complete. This function returns <c>CL_SUCCESS</c>
   if the copy was successful, or a valid OpenCL error code otherwise.
   The integer pointed to by \p ret is set to <c>-EAGAIN</c> if the
   asynchronous launch was successful, or to 0 if \p event was
   <c>NULL</c>.
*/
cl_int starpu_opencl_copy_opencl_to_ram(cl_mem buffer, unsigned src_node, void *ptr, unsigned dst_node, size_t size, size_t offset, cl_event *event, int *ret);

/**
   Copy \p size bytes asynchronously from byte offset \p src_offset of \p
   src on OpenCL \p src_node to byte offset \p dst_offset of \p dst on
   OpenCL \p dst_node. if \p event is <c>NULL</c>, the copy is
   synchronous, i.e. the queue is synchronised before returning. If not
   <c>NULL</c>, \p event can be used after the call to wait for this
   particular copy to complete. This function returns <c>CL_SUCCESS</c>
   if the copy was successful, or a valid OpenCL error code otherwise.
   The integer pointed to by \p ret is set to <c>-EAGAIN</c> if the
   asynchronous launch was successful, or to 0 if \p event was
   <c>NULL</c>.
*/
cl_int starpu_opencl_copy_opencl_to_opencl(cl_mem src, unsigned src_node, size_t src_offset, cl_mem dst, unsigned dst_node, size_t dst_offset, size_t size, cl_event *event, int *ret);

/**
   Copy \p size bytes from byte offset \p src_offset of \p src on \p
   src_node to byte offset \p dst_offset of \p dst on \p dst_node. if \p
   event is <c>NULL</c>, the copy is synchronous, i.e. the queue is
   synchronised before returning. If not <c>NULL</c>, \p event can be
   used after the call to wait for this particular copy to complete. The
   function returns <c>-EAGAIN</c> if the asynchronous launch was
   successfull. It returns 0 if the synchronous copy was successful, or
   fails otherwise.
*/
cl_int starpu_opencl_copy_async_sync(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, cl_event *event);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_OPENCL */

#endif /* __STARPU_OPENCL_H__ */
