/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_DATA_INTERFACES_H__
#define __STARPU_DATA_INTERFACES_H__

#include <starpu.h>

#ifdef STARPU_USE_CUDA
/* to use CUDA streams */
#include <cuda_runtime.h>
typedef cudaStream_t starpu_cudaStream_t;
#endif

#ifdef STARPU_USE_HIP
/* to use HIP streams */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#ifndef __cplusplus
#pragma GCC diagnostic ignored "-Wimplicit-int"
#endif
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <hip/hip_runtime.h>
#pragma GCC diagnostic pop
typedef hipStream_t starpu_hipStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Data_Interfaces Data Interfaces
   @brief Data management is done at a high-level in StarPU: rather than
   accessing a mere list of contiguous buffers, the tasks may manipulate
   data that are described by a high-level construct which we call data
   interface.

   An example of data interface is the "vector" interface which describes
   a contiguous data array on a specific memory node. This interface is a
   simple structure containing the number of elements in the array, the
   size of the elements, and the address of the array in the appropriate
   address space (this address may be invalid if there is no valid copy
   of the array in the memory node). More information on the data
   interfaces provided by StarPU are given in \ref API_Data_Interfaces.

   When a piece of data managed by StarPU is used by a task, the task
   implementation is given a pointer to an interface describing a valid
   copy of the data that is accessible from the current processing unit.

   Every worker is associated to a memory node which is a logical
   abstraction of the address space from which the processing unit gets
   its data. For instance, the memory node associated to the different
   CPU workers represents main memory (RAM), the memory node associated
   to a GPU is DRAM embedded on the device. Every memory node is
   identified by a logical index which is accessible from the
   function starpu_worker_get_memory_node(). When registering a piece of
   data to StarPU, the specified memory node indicates where the piece of
   data initially resides (we also call this memory node the home node of
   a piece of data).

   In the case of NUMA systems, functions starpu_memory_nodes_numa_devid_to_id()
   and starpu_memory_nodes_numa_id_to_devid() can be used to convert from NUMA node
   numbers as seen by the Operating System and NUMA node numbers as seen by StarPU.

   There are several ways to register a memory region so that it can be
   managed by StarPU. StarPU provides data interfaces for vectors, 2D
   matrices, 3D matrices as well as BCSR and CSR sparse matrices.

   Each data interface is provided with a set of field access functions.
   The ones using a <c>void *</c> parameter aimed to be used in codelet
   implementations (see for example the code in
   \ref VectorScalingUsingStarPUAPI).

   Applications can provide their own interface as shown in \ref DefiningANewDataInterface.

   @{
*/

/**
   Define the per-interface methods. If the
   starpu_data_copy_methods::any_to_any method is provided, it will be
   used by default if no specific method is provided. It can still be
   useful to provide more specific method in case of e.g. available
   particular CUDA, HIP or OpenCL support.

   See \ref DefiningANewDataInterface_copy for more details.
*/
struct starpu_data_copy_methods
{
	/**
	   If defined, allow the interface to declare whether it supports
	   transferring from \p src_interface on node \p src_node to \p
	   dst_interface on node \p dst_node, run from node \p handling_node.
	   If not defined, it is assumed that the interface supports all
	   transfers.
	*/
	int (*can_copy)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, unsigned handling_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node CPU node. Return 0 on success.
	*/
	int (*ram_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node CUDA node. Return 0 on success.
	*/
	int (*ram_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node HIP node. Return 0 on success.
	*/
	int (*ram_to_hip)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node OpenCL node. Return 0 on success.
	*/
	int (*ram_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node FPGA node. Return 0 on success.
	*/
	int (*ram_to_max_fpga)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CUDA node to the \p dst_interface interface on the \p
	   dst_node CPU node. Return 0 on success.
	*/
	int (*cuda_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CUDA node to the \p dst_interface interface on the \p
	   dst_node CUDA node. Return 0 on success.
	*/
	int (*cuda_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node HIP node to the \p dst_interface interface on the \p
	   dst_node CPU node. Return 0 on success.
	*/
	int (*hip_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node HIP node to the \p dst_interface interface on the \p
	   dst_node HIP node. Return 0 on success.
	*/
	int (*hip_to_hip)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node OpenCL node to the \p dst_interface interface on the
	   \p dst_node CPU node. Return 0 on success.
	*/
	int (*opencl_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node OpenCL node to the \p dst_interface interface on the
	   \p dst_node OpenCL node. Return 0 on success.
	*/
	int (*opencl_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node FPGA node to the \p dst_interface interface on the \p
	   dst_node CPU node. Return 0 on success.
	*/
	int (*max_fpga_to_ram)(void *src_interface, unsigned srd_node, void *dst_interface, unsigned dst_node);

#ifdef STARPU_USE_CUDA
	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node CUDA node, using the given stream. Must return 0 if the
	   transfer was actually completed completely synchronously, or
	   <c>-EAGAIN</c> if at least some transfers are still ongoing and
	   should be awaited for by the core.
	*/
	int (*ram_to_cuda_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, starpu_cudaStream_t stream);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CUDA node to the \p dst_interface interface on the \p
	   dst_node CPU node, using the given stream. Must return 0 if the
	   transfer was actually completed completely synchronously, or
	   <c>-EAGAIN</c> if at least some transfers are still ongoing and
	   should be awaited for by the core.
	*/
	int (*cuda_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, starpu_cudaStream_t stream);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CUDA node to the \p dst_interface interface on the \p
	   dst_node CUDA node, using the given stream. Must return 0 if the
	   transfer was actually completed completely synchronously, or
	   <c>-EAGAIN</c> if at least some transfers are still ongoing and
	   should be awaited for by the core.
	*/
	int (*cuda_to_cuda_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, starpu_cudaStream_t stream);
#else
	int (*ram_to_cuda_async)(void);
	int (*cuda_to_ram_async)(void);
	int (*cuda_to_cuda_async)(void);
#endif

#ifdef STARPU_USE_HIP
	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node HIP node, using the given stream. Must return 0 if the
	   transfer was actually completed completely synchronously, or
	   <c>-EAGAIN</c> if at least some transfers are still ongoing and
	   should be awaited for by the core.
	*/
	int (*ram_to_hip_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, starpu_hipStream_t stream);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node HIP node to the \p dst_interface interface on the \p
	   dst_node CPU node, using the given stream. Must return 0 if the
	   transfer was actually completed completely synchronously, or
	   <c>-EAGAIN</c> if at least some transfers are still ongoing and
	   should be awaited for by the core.
	*/
	int (*hip_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, starpu_hipStream_t stream);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node HIP node to the \p dst_interface interface on the \p
	   dst_node HIP node, using the given stream. Must return 0 if the
	   transfer was actually completed completely synchronously, or
	   <c>-EAGAIN</c> if at least some transfers are still ongoing and
	   should be awaited for by the core.
	*/
	int (*hip_to_hip_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, starpu_hipStream_t stream);
#else
	int (*ram_to_hip_async)(void);
	int (*hip_to_ram_async)(void);
	int (*hip_to_hip_async)(void);
#endif

#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node OpenCL node, by recording in \p event, a pointer to a
	   <c>cl_event</c>, the event of the last submitted transfer. Must
	   return 0 if the transfer was actually completed completely
	   synchronously, or <c>-EAGAIN</c> if at least some transfers are
	   still ongoing and should be awaited for by the core.
	*/
	int (*ram_to_opencl_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event);
	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node OpenCL node to the \p dst_interface interface on the
	   \p dst_node CPU node, by recording in \p event, a pointer to a
	   <c>cl_event</c>, the event of the last submitted transfer. Must
	   return 0 if the transfer was actually completed completely
	   synchronously, or <c>-EAGAIN</c> if at least some transfers are
	   still ongoing and should be awaited for by the core.
	*/
	int (*opencl_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event);
	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node OpenCL node to the \p dst_interface interface on the
	   \p dst_node OpenCL node, by recording in \p event, a pointer to a
	   <c>cl_event</c>, the event of the last submitted transfer. Must
	   return 0 if the transfer was actually completed completely
	   synchronously, or <c>-EAGAIN</c> if at least some transfers are
	   still ongoing and should be awaited for by the core.
	*/
	int (*opencl_to_opencl_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event);
#else
	int (*ram_to_opencl_async)(void);
	int (*opencl_to_ram_async)(void);
	int (*opencl_to_opencl_async)(void);
#endif

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node CPU node to the \p dst_interface interface on the \p
	   dst_node FPGA node. Must return 0 if the transfer was actually
	   completed completely synchronously, or <c>-EAGAIN</c> if at least
	   some transfers are still ongoing and should be awaited for by the
	   core.
	*/
	int (*ram_to_max_fpga_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node FPGA node to the \p dst_interface interface on the \p
	   dst_node CPU node. Must return 0 if the transfer was actually
	   completed completely synchronously, or <c>-EAGAIN</c> if at least
	   some transfers are still ongoing and should be awaited for by the
	   core.
	*/
	int (*max_fpga_to_ram_async)(void *src_interface, unsigned srd_node, void *dst_interface, unsigned dst_node);

	/**
	   Define how to copy data from the \p src_interface interface on the
	   \p src_node node to the \p dst_interface interface on the \p
	   dst_node node. This is meant to be implemented through the
	   starpu_interface_copy() helper, to which async_data should be
	   passed as such, and will be used to manage asynchronicity. This
	   must return <c>-EAGAIN</c> if any of the starpu_interface_copy()
	   calls has returned <c>-EAGAIN</c> (i.e. at least some transfer is
	   still ongoing), and return 0 otherwise.

	   This can only be implemented if the interface has ready-to-send
	   data blocks. If the interface is more involved than
	   this, i.e. it needs to collect pieces of data before
	   transferring, starpu_data_interface_ops::pack_data and
	   starpu_data_interface_ops::peek_data should be implemented instead,
	   and the core will just transfer the resulting data buffer.
	*/
	int (*any_to_any)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);
};

/**
   Identifier for all predefined StarPU data interfaces
*/
enum starpu_data_interface_id
{
	STARPU_UNKNOWN_INTERFACE_ID	= -1, /**< Unknown interface */
	STARPU_MATRIX_INTERFACE_ID	= 0,  /**< Identifier for the matrix data interface */
	STARPU_BLOCK_INTERFACE_ID	= 1,  /**< Identifier for the block data interface*/
	STARPU_VECTOR_INTERFACE_ID	= 2,  /**< Identifier for the vector data interface*/
	STARPU_CSR_INTERFACE_ID		= 3,  /**< Identifier for the CSR data interface*/
	STARPU_BCSR_INTERFACE_ID	= 4,  /**< Identifier for the BCSR data interface*/
	STARPU_VARIABLE_INTERFACE_ID	= 5,  /**< Identifier for the variable data interface*/
	STARPU_VOID_INTERFACE_ID	= 6,  /**< Identifier for the void data interface*/
	STARPU_MULTIFORMAT_INTERFACE_ID = 7,  /**< Identifier for the multiformat data interface*/
	STARPU_COO_INTERFACE_ID		= 8,  /**< Identifier for the COO data interface*/
	STARPU_TENSOR_INTERFACE_ID	= 9,  /**< Identifier for the tensor data interface*/
	STARPU_NDIM_INTERFACE_ID	= 10, /**< Identifier for the ndim array data interface*/
	STARPU_MAX_INTERFACE_ID		= 11  /**< Maximum number of data interfaces */
};

/**
   Per-interface data management methods.
*/
struct starpu_data_interface_ops
{
	/**
	   Register an existing interface into a data handle.

	   This iterates over all memory nodes to initialize all fields of the data
	   interface on each of them. Since data is not allocated yet except on the
	   home node, pointers should be left as NULL except on the \p home_node (if >= 0), for
	   which the pointers should be copied from the given \p data_interface, which
	   was filled with the application's pointers.

	   This method is mandatory.

	   See \ref DefiningANewDataInterface_registration for more details.
	*/
	void (*register_data_handle)(starpu_data_handle_t handle, int home_node, void *data_interface);

	/**
	   Unregister a data handle.

	   This iterates over all memory nodes to free any pointer in the data
	   interface on each of them.

	   At this point, free_data_on_node has been already called on each of them.
	   This just clears anything that would still be left.

	   See \ref DefiningANewDataInterface_registration for more details.
	*/
	void (*unregister_data_handle)(starpu_data_handle_t handle);

	/**
	   Allocate data for the interface on a given node. This should use
	   starpu_malloc_on_node() to perform the allocation(s), and fill the pointers
	   in the data interface. It should return the size of the allocated memory, or
	   -ENOMEM if memory could not be allocated.

	   Note that the memory node can be CPU memory, GPU memory, or even disk
	   area. The result returned by starpu_malloc_on_node() should be just
	   stored as uintptr_t without trying to interpret it since it may be a
	   GPU pointer, a disk descriptor, etc.

	   This method is mandatory to be able to support memory nodes.

	   See \ref DefiningANewDataInterface_pointers for more details.
	*/
	starpu_ssize_t (*allocate_data_on_node)(void *data_interface, unsigned node);

	/**
	   Free data of the interface on a given node.

	   This method is mandatory to be able to support memory nodes.

	   See \ref DefiningANewDataInterface_pointers for more details.
	*/
	void (*free_data_on_node)(void *data_interface, unsigned node);

	/**
	   Cache the buffers from the given node to a caching interface.

	   This method is optional, mostly useful when also making
	   starpu_data_interface_ops::unregister_data_handle check that pointers are NULL.

	   \p src_interface is an interface that already has buffers
	   allocated, but which we don't need any more. \p cached_interface
	   is a new interface into which the buffer pointers should be
	   transferred, for later reuse when allocating data of the same kind.

	   Usually we can just memcpy over the set of pointers and descriptions
	   (this is what StarPU does when this method is not implemented), but
	   if unregister_data_handle checks that pointers are NULL, we need to
	   additionally clear the pointers in \p src_interface. Also,
	   it is not useful to copy the whole interface, only the
	   pointers need to be copied (essentially the pointers that
	   starpu_data_interface_ops::reuse_data_on_node will then transfer into
	   a new handle interface), as well as the properties
	   that starpu_data_interface_ops::compare (or
	   starpu_data_interface_ops::alloc_compare if defined) needs for
	   comparing interfaces for caching compatibility.

	   When this method is not defined, StarPU will just copy the \p
	   cached_interface into \p src_interface.

	   See \ref VariableSizeDataInterface and \ref DefiningANewDataInterface_pointers for more details.
	*/
	void (*cache_data_on_node)(void *cached_interface, void *src_interface, unsigned node);

	/**
	   Reuse on the given node the buffers of the provided interface

	   This method is optional, mostly useful when also defining
	   alloc_footprint to share tiles of the same allocation size but
	   different shapes, or when the interface contains pointers which
	   are initialized at registration (e.g. nn array in the ndim interface)

	   \p cached_interface is an already-allocated buffer that we want to
	   reuse, and \p new_data_interface is an interface in which we want to
	   install that already-allocated buffer. Usually we can just memcpy over
	   the set of pointers and descriptions. But e.g. with 2D tiles the ld
	   value may not be correct, and memcpy would wrongly overwrite it in
	   new_data_interface, i.e. reusing a vertical tile allocation for a horizontal tile, or vice-versa.

	   reuse_data_on_node should thus copy over pointers, and define fields
	   that are usually set by allocate_data_on_node (e.g. ld).

	   See \ref VariableSizeDataInterface and \ref DefiningANewDataInterface_pointers for more details.
	*/
	void (*reuse_data_on_node)(void *dst_data_interface, const void *cached_interface, unsigned node);

	/**
	   Map data from a source to a destination.
	   Define function starpu_interface_map() to set this field.
	   See \ref DefiningANewDataInterface_pointers for more details.
	*/
	int (*map_data)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Unmap data from a source to a destination.
	   Define function starpu_interface_unmap() to set this field.
	   See \ref DefiningANewDataInterface_pointers for more details.
	*/
	int (*unmap_data)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Update map data from a source to a destination.
	   Define function starpu_interface_update_map() to set this field.
	   See \ref DefiningANewDataInterface_pointers for more details.
	*/
	int (*update_map)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/**
	   Initialize the interface.
	   This method is optional. It is called when initializing the
	   handler on all the memory nodes.
	*/
	void (*init)(void *data_interface);

	/**
	   Struct with pointer to functions for performing ram/cuda/opencl synchronous and asynchronous transfers.

	   This field is mandatory to be able to support memory
	   nodes, except disk nodes which can be supported by just
	   implementing starpu_data_interface_ops::pack_data and
	   starpu_data_interface_ops::unpack_data.
	*/
	const struct starpu_data_copy_methods *copy_methods;

	/**
	   @deprecated
	   Use starpu_data_interface_ops::to_pointer instead.
	   Return the current pointer (if any) for the handle on the given node.

	   This method is only required if starpu_data_interface_ops::to_pointer
	   is not implemented.
	*/
	void *(*handle_to_pointer)(starpu_data_handle_t handle, unsigned node);

	/**
	   Return the current pointer (if any) for the given interface on the given node.

	   This method is only required for starpu_data_handle_to_pointer()
	   and starpu_data_get_local_ptr(), and for disk support.
	*/
	void *(*to_pointer)(void *data_interface, unsigned node);

	/**
	   Return an estimation of the size of data, for performance models and tracing feedback.
	*/
	size_t (*get_size)(starpu_data_handle_t handle);

	/**
	   Return an estimation of the size of allocated data, for allocation
	   management.
	   If not specified, the starpu_data_interface_ops::get_size method is
	   used instead.
	*/
	size_t (*get_alloc_size)(starpu_data_handle_t handle);

	/**
	   Return the maximum size that the data may need to increase to. For
	   instance, in the case of compressed matrix tiles this is the size
	   when the block is fully dense.
	   This is currently only used for feedback tools.
	*/
	size_t (*get_max_size)(starpu_data_handle_t handle);

	/**
	  Return a 32bit footprint which characterizes the data size and layout (nx, ny, ld, elemsize, etc.), required for indexing performance models.

	  starpu_hash_crc32c_be() and alike can be used to produce this 32bit value from various types of values.
	*/
	uint32_t (*footprint)(starpu_data_handle_t handle);

	/**
	   Return a 32bit footprint which characterizes the data allocation, to be used
	   for indexing allocation cache.
	   If not specified, the starpu_data_interface_ops::footprint method is
	   used instead.
	   If specified, alloc_compare should be set to provide the strict
	   comparison, and reuse_data_on_node should be set to provide correct buffer reuse.
	*/
	uint32_t (*alloc_footprint)(starpu_data_handle_t handle);

	/**
	   Compare the data size and layout of two interfaces (nx, ny, ld, elemsize,
	   etc.), to be used for indexing performance models. It should return 1 if
	   the two interfaces size and layout match computation-wise, and 0 otherwise.
	   It does *not* compare the actual content of the interfaces.
	*/
	int (*compare)(void *data_interface_a, void *data_interface_b);

	/**
	   Compare the data allocation of two interfaces etc.), to be used for indexing
	   allocation cache. It should return
	   1 if the two interfaces are allocation-compatible, i.e. basically have the same alloc_size, and 0 otherwise.
	   If not specified, the starpu_data_interface_ops::compare method is
	   used instead.
	*/
	int (*alloc_compare)(void *data_interface_a, void *data_interface_b);

	/**
	   Dump the sizes of a handle to a file.
	   This is required for performance models
	*/
	void (*display)(starpu_data_handle_t handle, FILE *f);

	/**
	   Describe the data into a string in a brief way, such as one
	   letter to describe the type of data, and the data
	   dimensions.
	   This is required for tracing feedback.
	*/
	starpu_ssize_t (*describe)(void *data_interface, char *buf, size_t size);

	/**
	   An identifier that is unique to each interface.
	*/
	enum starpu_data_interface_id interfaceid;

	/**
	   Size of the interface data descriptor.
	*/
	size_t interface_size;

	/**
	*/
	char is_multiformat;

	/**
	   If set to non-zero, StarPU will never try to reuse an allocated
	   buffer for a  different handle. This can be notably useful for
	   application-defined interfaces which have a dynamic size, and for
	   which it thus does not make sense to reuse the buffer since will
	   probably not have the proper size.
	*/
	char dontcache;

	/**
	*/
	struct starpu_multiformat_data_interface_ops *(*get_mf_ops)(void *data_interface);

	/**
	   Pack the data handle into a contiguous buffer at the address
	   allocated with <c>starpu_malloc_flags(ptr, size, 0)</c> (and thus
	   returned in \p ptr) and set the size of the newly created buffer
	   in \p count. If \p ptr is <c>NULL</c>, the function should not
	   copy the data in the buffer but just set count to the size of the
	   buffer which would have been allocated. The special value -1
	   indicates the size is yet unknown.

	   This method (and starpu_data_interface_ops::unpack_data) is required
	   for disk support if the starpu_data_copy_methods::any_to_any method
	   is not implemented (because the in-memory data layout is too
	   complex).

	   This is also required for MPI support if there is no registered MPI data type.
	*/
	int (*pack_data)(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);

	/**
	   Read the data handle from the contiguous buffer at the address
	   \p ptr of size \p count.
	*/
	int (*peek_data)(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

	/**
	   Unpack the data handle from the contiguous buffer at the address
	   \p ptr of size \p count.
	   The memory at the address \p ptr should be freed after the data unpacking operation.
	*/
	int (*unpack_data)(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

	/**
	   Pack the interface into a contiguous buffer and set the
	   size of the newly created buffer in \p count. This function
	   is used in master slave mode for data interfaces with a
	   dynamic content.
	*/
	int (*pack_meta)(void *data_interface, void **ptr, starpu_ssize_t *count);

	/**
	   Unpack the interface from the given buffer and set the size
	   of the unpacked data in \p count. This function
	   is used in master slave mode for data interfaces with a
	   dynamic content.
	*/
	int (*unpack_meta)(void **data_interface, void *ptr, starpu_ssize_t *count);

	/**
	   Free the allocated memory by a previous call to unpack_meta()
	*/
	int (*free_meta)(void *data_interface);

	/**
	   Name of the interface
	*/
	char *name;
};

/**
   @name Basic API
   @{
*/

/**
   Register a piece of data into the handle located at the
   \p handleptr address. The \p data_interface buffer contains the initial
   description of the data in the \p home_node. The \p ops argument is a
   pointer to a structure describing the different methods used to
   manipulate this type of interface. See starpu_data_interface_ops for
   more details on this structure.
   If \p home_node is -1, StarPU will automatically allocate the memory when
   it is used for the first time in write-only mode. Once such data
   handle has been automatically allocated, it is possible to access it
   using any access mode.
   Note that StarPU supplies a set of predefined types of interface (e.g.
   vector or matrix) which can be registered by the means of helper
   functions (e.g. starpu_vector_data_register() or
   starpu_matrix_data_register()).

   See \ref DefiningANewDataInterface_registration for more details.
*/
void starpu_data_register(starpu_data_handle_t *handleptr, int home_node, void *data_interface, struct starpu_data_interface_ops *ops);

/**
   Register the given data interface operations. If the field
   starpu_data_interface_ops::field is set to
   ::STARPU_UNKNOWN_INTERFACE_ID, then a new identifier will be set by
   calling starpu_data_interface_get_next_id().
   The function is automatically called when registering a piece of
   data with starpu_data_register(). It is only necessary to call it
   beforehand for some specific cases (such as the usmaster slave mode).
*/
void starpu_data_register_ops(struct starpu_data_interface_ops *ops);

/**
   Register that a buffer for \p handle on \p node will be set. This is typically
   used by starpu_*_ptr_register helpers before setting the interface pointers for
   this node, to tell the core that that is now allocated.
   See \ref DefiningANewDataInterface_pointers for more details.
*/
void starpu_data_ptr_register(starpu_data_handle_t handle, unsigned node);

/**
   Register a new piece of data into the handle \p handledst with the
   same interface as the handle \p handlesrc.
   See \ref DataHandlesHelpers for more details.
*/
void starpu_data_register_same(starpu_data_handle_t *handledst, starpu_data_handle_t handlesrc);

/**
   Return the pointer associated with \p handle on node \p node or <c>NULL</c>
   if handle’s interface does not support this operation or data for this
   \p handle is not allocated on that \p node.
   See \ref DataPointers for more details.
*/
void *starpu_data_handle_to_pointer(starpu_data_handle_t handle, unsigned node);

/**
   Return the local pointer associated with \p handle or <c>NULL</c> if
   \p handle’s interface does not have any data allocated locally.
   See \ref DataPointers for more details.
*/
void *starpu_data_get_local_ptr(starpu_data_handle_t handle);

/**
   Return the interface associated with \p handle on \p memory_node.
   See \ref DefiningANewDataInterface_pack for more details.
*/
void *starpu_data_get_interface_on_node(starpu_data_handle_t handle, unsigned memory_node);

/**
   Return the unique identifier of the interface associated with
   the given \p handle.
   See \ref DefiningANewDataInterface_helpers for more details.
*/
enum starpu_data_interface_id starpu_data_get_interface_id(starpu_data_handle_t handle);

/**
   Execute the packing operation of the interface of the data
   registered at \p handle (see starpu_data_interface_ops). This
   packing operation must allocate a buffer large enough at \p ptr on node \p node and copy
   into the newly allocated buffer the data associated to \p handle. \p count
   will be set to the size of the allocated buffer. If \p ptr is <c>NULL</c>, the
   function should not copy the data in the buffer but just set \p count to
   the size of the buffer which would have been allocated. The special
   value -1 indicates the size is yet unknown.
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_pack_node(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);

/**
   Like starpu_data_pack_node(), but for the local memory node.
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_pack(starpu_data_handle_t handle, void **ptr, starpu_ssize_t *count);

/**
   Read in handle's \p node replicate the data located at \p ptr
   of size \p count as described by the interface of the data. The interface
   registered at \p handle must define a peeking operation (see
   starpu_data_interface_ops).
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_peek_node(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

/**
   Read in handle's local replicate the data located at \p ptr
   of size \p count as described by the interface of the data. The interface
   registered at \p handle must define a peeking operation (see
   starpu_data_interface_ops).
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_peek(starpu_data_handle_t handle, void *ptr, size_t count);

/**
   Unpack in handle the data located at \p ptr of size \p count allocated
   on node \p node as described by the interface of the data. The interface
   registered at \p handle must define an unpacking operation (see
   starpu_data_interface_ops).
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_unpack_node(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

/**
   Unpack in handle the data located at \p ptr of size \p count as
   described by the interface of the data. The interface registered at
   \p handle must define a unpacking operation (see
   starpu_data_interface_ops).
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_unpack(starpu_data_handle_t handle, void *ptr, size_t count);

/**
   Return the size of the data associated with \p handle.
   See \ref DataHandlesHelpers for more details.
*/
size_t starpu_data_get_size(starpu_data_handle_t handle);

/**
   Return the size of the allocated data associated with \p handle.
   See \ref DataHandlesHelpers for more details.
*/
size_t starpu_data_get_alloc_size(starpu_data_handle_t handle);

/**
   Return the maximum size that the \p handle data may need to increase to.
   See \ref DataHandlesHelpers for more details.
*/
starpu_ssize_t starpu_data_get_max_size(starpu_data_handle_t handle);

/**
   See \ref DataHandlesHelpers for more details.
*/
int starpu_data_get_home_node(starpu_data_handle_t handle);

/**
   Print basic information on \p handle on \p node.
   See \ref DataHandlesHelpers for more details.
 */
void starpu_data_print(starpu_data_handle_t handle, unsigned node, FILE *stream);

/**
   Return the next available id for a newly created data interface
   (\ref DefiningANewDataInterface).
*/
int starpu_data_interface_get_next_id(void);

/**
   Copy \p size bytes from byte offset \p src_offset of \p src on \p src_node
   to byte offset \p dst_offset of \p dst on \p dst_node. This is to be used in
   the starpu_data_copy_methods::any_to_any copy method, which is provided with \p async_data to
   be passed to starpu_interface_copy(). this returns <c>-EAGAIN</c> if the
   transfer is still ongoing, or 0 if the transfer is already completed.

   See \ref DefiningANewDataInterface_copy for more details.
*/
int starpu_interface_copy(uintptr_t src, size_t src_offset, unsigned src_node,
			  uintptr_t dst, size_t dst_offset, unsigned dst_node,
			  size_t size, void *async_data);

/**
   Copy \p numblocks blocks of \p blocksize bytes from byte offset \p src_offset
   of \p src on \p src_node to byte offset \p dst_offset of \p dst on \p
   dst_node.

   The blocks start at addresses which are ld_src (resp. ld_dst) bytes apart in
   the source (resp. destination) interface.

   If blocksize == ld_src == ld_dst, the transfer is optimized into a single
   starpu_interface_copy call.

   This is to be used in the starpu_data_copy_methods::any_to_any copy
   method for 2D data, which is provided with \p async_data to be passed to
   starpu_interface_copy(). this returns <c>-EAGAIN</c> if the transfer is still
   ongoing, or 0 if the transfer is already completed.

   See \ref DefiningANewDataInterface_copy for more details.
*/
int starpu_interface_copy2d(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t blocksize,
			    size_t numblocks, size_t ld_src, size_t ld_dst,
			    void *async_data);

/**
   Copy \p numblocks_1 * \p numblocks_2 blocks of \p blocksize bytes from byte
   offset \p src_offset of \p src on \p src_node to byte offset \p dst_offset of
   \p dst on \p dst_node.

   The blocks are grouped by \p numblocks_1 blocks whose start addresses are
   ld1_src (resp. ld1_dst) bytes apart in the source (resp. destination)
   interface.

   Such groups are grouped by numblocks_2 groups whose start addresses are
   ld2_src (resp. ld2_dst) bytes apart in the source (resp. destination)
   interface.

   If the blocks are contiguous, the transfers will be optimized.

   This is to be used in the starpu_data_copy_methods::any_to_any copy
   method for 3D data, which is provided with \p async_data to be passed to
   starpu_interface_copy(). this returns <c>-EAGAIN</c> if the transfer is still
   ongoing, or 0 if the transfer is already completed.

   See \ref DefiningANewDataInterface_copy for more details.
*/
int starpu_interface_copy3d(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t blocksize,
			    size_t numblocks1, size_t ld1_src, size_t ld1_dst,
			    size_t numblocks2, size_t ld2_src, size_t ld2_dst,
			    void *async_data);

/**
   Copy \p numblocks_1 * \p numblocks_2 * \p numblocks_3 blocks of \p blocksize
   bytes from byte offset \p src_offset of \p src on \p src_node to byte offset
   \p dst_offset of \p dst on \p dst_node.

   The blocks are grouped by \p numblocks_1 blocks whose start addresses are
   ld1_src (resp. ld1_dst) bytes apart in the source (resp. destination)
   interface.

   Such groups are grouped by numblocks_2 groups whose start addresses are
   ld2_src (resp. ld2_dst) bytes apart in the source (resp. destination)
   interface.

   Such groups are grouped by numblocks_3 groups whose start addresses are
   ld3_src (resp. ld3_dst) bytes apart in the source (resp. destination)
   interface.

   If the blocks are contiguous, the transfers will be optimized.

   This is to be used in the starpu_data_copy_methods::any_to_any copy
   method for 4D data, which is provided with \p async_data to be passed to
   starpu_interface_copy(). this returns <c>-EAGAIN</c> if the transfer is still
   ongoing, or 0 if the transfer is already completed.

   See \ref DefiningANewDataInterface_copy for more details.
*/
int starpu_interface_copy4d(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t blocksize,
			    size_t numblocks1, size_t ld1_src, size_t ld1_dst,
			    size_t numblocks2, size_t ld2_src, size_t ld2_dst,
			    size_t numblocks3, size_t ld3_src, size_t ld3_dst,
			    void *async_data);

/**
   Copy \p nn[1] * \p nn[2]...* \p nn[ndim-1] blocks of \p nn[0] * \p elemsize bytes from byte
   offset \p src_offset of \p src on \p src_node to byte offset \p dst_offset of
   \p dst on \p dst_node.

   The blocks are grouped by \p nn[i] blocks (i = 1, 2, ... ndim-1) whose start addresses are
   ldn_src[i] * \p elemsize (resp. ld1_dst[i] * \p elemsize) bytes apart
   in the source (resp. destination) interface.

   If the blocks are contiguous, the transfers will be optimized.

   This is to be used in the starpu_data_copy_methods::any_to_any copy
   method for Ndim data, which is provided with \p async_data to be passed to
   starpu_interface_copy(). this returns <c>-EAGAIN</c> if the transfer is still
   ongoing, or 0 if the transfer is already completed.

   See \ref DefiningANewDataInterface_copy for more details.
*/
int starpu_interface_copynd(uintptr_t src, size_t src_offset, unsigned src_node,
			    uintptr_t dst, size_t dst_offset, unsigned dst_node,
			    size_t elemsize, size_t ndim,
			    uint32_t *nn, uint32_t *ldn_src, uint32_t *ldn_dst,
			    void *async_data);

/**
   When an asynchronous implementation of the data transfer is implemented, the call
   to the underlying CUDA, OpenCL, etc. call should be surrounded
   by calls to starpu_interface_start_driver_copy_async() and
   starpu_interface_end_driver_copy_async(), so that it is recorded in offline
   execution traces, and the timing of the submission is checked. \p start must
   point to a variable whose value will be passed unchanged to
   starpu_interface_end_driver_copy_async().

   See \ref DefiningANewDataInterface_copy for more details.
*/
void starpu_interface_start_driver_copy_async(unsigned src_node, unsigned dst_node, double *start);

/**
   See starpu_interface_start_driver_copy_async().
   See \ref DefiningANewDataInterface_copy for more details.
*/
void starpu_interface_end_driver_copy_async(unsigned src_node, unsigned dst_node, double start);

/**
   Record in offline execution traces the copy of \p size bytes from
   node \p src_node to node \p dst_node.
   See \ref DefiningANewDataInterface_copy for more details.
 */
void starpu_interface_data_copy(unsigned src_node, unsigned dst_node, size_t size);

/**
   Allocate \p size bytes on node \p dst_node with the given allocation \p flags. This returns 0 if
   allocation failed, the allocation method should then return <c>-ENOMEM</c> as
   allocated size. Deallocation must be done with starpu_free_on_node_flags().

   See \ref VariableSizeDataInterface for more details.
*/
uintptr_t starpu_malloc_on_node_flags(unsigned dst_node, size_t size, int flags);

/**
   Allocate \p size bytes on node \p dst_node with the default allocation flags. This returns 0 if
   allocation failed, the allocation method should then return <c>-ENOMEM</c> as
   allocated size. Deallocation must be done with starpu_free_on_node().

   See \ref DefiningANewDataInterface_allocation for more details.
*/
uintptr_t starpu_malloc_on_node(unsigned dst_node, size_t size);

/**
   Free \p addr of \p size bytes on node \p dst_node which was previously allocated
   with starpu_malloc_on_node_flags() with the given allocation \p flags.

   See \ref VariableSizeDataInterface for more details.
*/
void starpu_free_on_node_flags(unsigned dst_node, uintptr_t addr, size_t size, int flags);

/**
   Free \p addr of \p size bytes on node \p dst_node which was previously allocated
   with starpu_malloc_on_node().

   See \ref DefiningANewDataInterface_allocation for more details.
*/
void starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size);

/**
   Define the default flags for allocations performed by starpu_malloc_on_node() and
   starpu_free_on_node(). The default is \ref STARPU_MALLOC_PINNED | \ref STARPU_MALLOC_COUNT.
	See \ref HowToLimitMemoryPerNode for more details.
*/
void starpu_malloc_on_node_set_default_flags(unsigned node, int flags);

/** @} */

/**
   @name MAP API
   @{
*/

/**
   Used to set starpu_data_interface_ops::map_data.
   See \ref DefiningANewDataInterface_pointers for more details.
*/
uintptr_t starpu_interface_map(uintptr_t src, size_t src_offset, unsigned src_node, unsigned dst_node, size_t size, int *ret);
/**
   Used to set starpu_data_interface_ops::unmap_data.
   See \ref DefiningANewDataInterface_pointers for more details.
*/
int starpu_interface_unmap(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, unsigned dst_node, size_t size);
/**
   Used to set starpu_data_interface_ops::update_map.
   See \ref DefiningANewDataInterface_pointers for more details.
*/
int starpu_interface_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size);

/** @} */

/**
   @name Accessing Matrix Data Interfaces
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_matrix_ops;

/**
   Matrix interface for dense matrices
*/
struct starpu_matrix_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */
	uintptr_t ptr;			  /**< local pointer of the matrix */
	uintptr_t dev_handle;		  /**< device handle of the matrix */
	size_t offset;			  /**< offset in the matrix */
	uint32_t nx;			  /**< number of elements on the x-axis of the matrix */
	uint32_t ny;			  /**< number of elements on the y-axis of the matrix */
	uint32_t ld;			  /**< number of elements between each row of the
					       matrix. Maybe be equal to starpu_matrix_interface::nx
					       when there is no padding.
					  */
	size_t elemsize;		  /**< size of the elements of the matrix */
	size_t allocsize;		  /**< size actually currently allocated */
};

/**
   Register the \p nx x \p  ny 2D matrix of \p elemsize-byte elements pointed
   by \p ptr and initialize \p handle to represent it. \p ld specifies the number
   of elements between rows. a value greater than \p nx adds padding, which
   can be useful for alignment purposes.

   Here an example of how to use the function.
   \code{.c}
   float *matrix;
   starpu_data_handle_t matrix_handle;
   matrix = (float*)malloc(width * height * sizeof(float));
   starpu_matrix_data_register(&matrix_handle, STARPU_MAIN_RAM, (uintptr_t)matrix, width, width, height, sizeof(float));
   \endcode

   See \ref MatrixDataInterface for more details.
*/
void starpu_matrix_data_register(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, uint32_t ld, uint32_t nx, uint32_t ny, size_t elemsize);

/**
   Similar to starpu_matrix_data_register, but additionally specifies which
   allocation size should be used instead of the initial nx*ny*elemsize.

   See \ref VariableSizeDataInterface for more details.
*/
void starpu_matrix_data_register_allocsize(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, uint32_t ld, uint32_t nx, uint32_t ny, size_t elemsize, size_t allocsize);

/**
   Register into the \p handle that to store data on node \p node it should use the
   buffer located at \p ptr, or device handle \p dev_handle and offset \p offset
   (for OpenCL, notably), with \p ld elements between rows.
*/
void starpu_matrix_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ld);

/**
   Return the number of elements on the x-axis of the matrix
   designated by \p handle.
*/
uint32_t starpu_matrix_get_nx(starpu_data_handle_t handle);

/**
   Return the number of elements on the y-axis of the matrix
   designated by \p handle.
*/
uint32_t starpu_matrix_get_ny(starpu_data_handle_t handle);

/**
   Return the number of elements between each row of the matrix
   designated by \p handle. Maybe be equal to nx when there is no padding.
*/
uint32_t starpu_matrix_get_local_ld(starpu_data_handle_t handle);

/**
   Return the local pointer associated with \p handle.
*/
uintptr_t starpu_matrix_get_local_ptr(starpu_data_handle_t handle);

/**
   Return the size of the elements registered into the matrix
   designated by \p handle.
*/
size_t starpu_matrix_get_elemsize(starpu_data_handle_t handle);

/**
   Return the allocated size of the matrix designated by \p handle.
*/
size_t starpu_matrix_get_allocsize(starpu_data_handle_t handle);

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_MATRIX_CHECK(interface)	 STARPU_ASSERT_MSG((((struct starpu_matrix_interface *)(interface))->id) == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.")
#define STARPU_MATRIX_GET_PTR(interface) (                              \
	{                                                               \
		STARPU_MATRIX_CHECK(interface);                         \
		(((struct starpu_matrix_interface *)(interface))->ptr); \
	})
#define STARPU_MATRIX_GET_DEV_HANDLE(interface) (                              \
	{                                                                      \
		STARPU_MATRIX_CHECK(interface);                                \
		(((struct starpu_matrix_interface *)(interface))->dev_handle); \
	})
#define STARPU_MATRIX_GET_OFFSET(interface) (                              \
	{                                                                  \
		STARPU_MATRIX_CHECK(interface);                            \
		(((struct starpu_matrix_interface *)(interface))->offset); \
	})
#define STARPU_MATRIX_GET_NX(interface) (                              \
	{                                                              \
		STARPU_MATRIX_CHECK(interface);                        \
		(((struct starpu_matrix_interface *)(interface))->nx); \
	})
#define STARPU_MATRIX_GET_NY(interface) (                              \
	{                                                              \
		STARPU_MATRIX_CHECK(interface);                        \
		(((struct starpu_matrix_interface *)(interface))->ny); \
	})
#define STARPU_MATRIX_GET_LD(interface) (                              \
	{                                                              \
		STARPU_MATRIX_CHECK(interface);                        \
		(((struct starpu_matrix_interface *)(interface))->ld); \
	})
#define STARPU_MATRIX_GET_ELEMSIZE(interface) (                              \
	{                                                                    \
		STARPU_MATRIX_CHECK(interface);                              \
		(((struct starpu_matrix_interface *)(interface))->elemsize); \
	})
#define STARPU_MATRIX_GET_ALLOCSIZE(interface) (                              \
	{                                                                     \
		STARPU_MATRIX_CHECK(interface);                               \
		(((struct starpu_matrix_interface *)(interface))->allocsize); \
	})
#else
/**
   Return a pointer to the matrix designated by \p interface, valid
   on CPUs and CUDA devices only. For OpenCL devices, the device handle
   and offset need to be used instead.
*/
#define STARPU_MATRIX_GET_PTR(interface)	(((struct starpu_matrix_interface *)(interface))->ptr)
/**
   Return a device handle for the matrix designated by \p interface,
   to be used with OpenCL. The offset returned by
   ::STARPU_MATRIX_GET_OFFSET has to be used in
   addition to this.
*/
#define STARPU_MATRIX_GET_DEV_HANDLE(interface) (((struct starpu_matrix_interface *)(interface))->dev_handle)
/**
   Return the offset in the matrix designated by \p interface, to be
   used with the device handle.
*/
#define STARPU_MATRIX_GET_OFFSET(interface)	(((struct starpu_matrix_interface *)(interface))->offset)
/**
   Return the number of elements on the x-axis of the matrix
   designated by \p interface.
*/
#define STARPU_MATRIX_GET_NX(interface)		(((struct starpu_matrix_interface *)(interface))->nx)
/**
   Return the number of elements on the y-axis of the matrix
   designated by \p interface.
*/
#define STARPU_MATRIX_GET_NY(interface)		(((struct starpu_matrix_interface *)(interface))->ny)
/**
   Return the number of elements between each row of the matrix
   designated by \p interface. May be equal to nx when there is no padding.
*/
#define STARPU_MATRIX_GET_LD(interface)		(((struct starpu_matrix_interface *)(interface))->ld)
/**
   Return the size of the elements registered into the matrix
   designated by \p interface.
*/
#define STARPU_MATRIX_GET_ELEMSIZE(interface)	(((struct starpu_matrix_interface *)(interface))->elemsize)
/**
   Return the allocated size of the matrix designated by \p interface.
*/
#define STARPU_MATRIX_GET_ALLOCSIZE(interface)	(((struct starpu_matrix_interface *)(interface))->allocsize)
#endif

/**
   Set the number of elements on the x-axis of the matrix
   designated by \p interface.
*/
#define STARPU_MATRIX_SET_NX(interface, newnx)                                   \
	do {                                                                     \
		STARPU_MATRIX_CHECK(interface);                                  \
		(((struct starpu_matrix_interface *)(interface))->nx) = (newnx); \
	}                                                                        \
	while (0)
/**
   Set the number of elements on the y-axis of the matrix
   designated by \p interface.
*/
#define STARPU_MATRIX_SET_NY(interface, newny)                                   \
	do {                                                                     \
		STARPU_MATRIX_CHECK(interface);                                  \
		(((struct starpu_matrix_interface *)(interface))->ny) = (newny); \
	}                                                                        \
	while (0)
/**
   Set the number of elements between each row of the matrix
   designated by \p interface. May be set to the same value as nx when there is
   no padding.
*/
#define STARPU_MATRIX_SET_LD(interface, newld)                                   \
	do {                                                                     \
		STARPU_MATRIX_CHECK(interface);                                  \
		(((struct starpu_matrix_interface *)(interface))->ld) = (newld); \
	}                                                                        \
	while (0)

/** @} */

/**
   @name Accessing COO Data Interfaces
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_coo_ops;

/**
   COO Matrices
*/
struct starpu_coo_interface
{
	enum starpu_data_interface_id id; /**< identifier of the interface */

	uint32_t *columns; /**< column array of the matrix */
	uint32_t *rows;	   /**< row array of the matrix */
	uintptr_t values;  /**< values of the matrix */
	uint32_t nx;	   /**< number of elements on the x-axis of the matrix */
	uint32_t ny;	   /**< number of elements on the y-axis of the matrix */
	uint32_t n_values; /**< number of values registered in the matrix */
	size_t elemsize;   /**< size of the elements of the matrix */
};

/**
   Register the \p nx x \p ny 2D matrix given in the COO format, using the
   \p columns, \p rows, \p values arrays, which must have \p n_values elements of
   size \p elemsize. Initialize \p handleptr.
   See \ref COODataInterface for more details.
*/
void starpu_coo_data_register(starpu_data_handle_t *handleptr, int home_node, uint32_t nx, uint32_t ny, uint32_t n_values, uint32_t *columns, uint32_t *rows, uintptr_t values, size_t elemsize);

/**
   Return a pointer to the column array of the matrix designated
   by \p interface.
*/
#define STARPU_COO_GET_COLUMNS(interface) (((struct starpu_coo_interface *)(interface))->columns)
/**
   Return a device handle for the column array of the matrix
   designated by \p interface, to be used with OpenCL. The offset
   returned by ::STARPU_COO_GET_OFFSET has to be used in addition to
   this.
*/
#define STARPU_COO_GET_COLUMNS_DEV_HANDLE(interface) (((struct starpu_coo_interface *)(interface))->columns)
/**
   Return a pointer to the rows array of the matrix designated by
   \p interface.
*/
#define STARPU_COO_GET_ROWS(interface) (((struct starpu_coo_interface *)(interface))->rows)
/**
   Return a device handle for the row array of the matrix
   designated by \p interface, to be used on OpenCL. The offset returned
   by ::STARPU_COO_GET_OFFSET has to be used in addition to this.
*/
#define STARPU_COO_GET_ROWS_DEV_HANDLE(interface) (((struct starpu_coo_interface *)(interface))->rows)
/**
   Return a pointer to the values array of the matrix designated
   by \p interface.
*/
#define STARPU_COO_GET_VALUES(interface) (((struct starpu_coo_interface *)(interface))->values)
/**
   Return a device handle for the value array of the matrix
   designated by \p interface, to be used on OpenCL. The offset returned
   by ::STARPU_COO_GET_OFFSET has to be used in addition to this.
*/
#define STARPU_COO_GET_VALUES_DEV_HANDLE(interface) (((struct starpu_coo_interface *)(interface))->values)
/**
   Return the offset in the arrays of the COO matrix designated by
   \p interface.
*/
#define STARPU_COO_GET_OFFSET 0
/**
   Return the number of elements on the x-axis of the matrix
   designated by \p interface.
*/
#define STARPU_COO_GET_NX(interface) (((struct starpu_coo_interface *)(interface))->nx)
/**
   Return the number of elements on the y-axis of the matrix
   designated by \p interface.
*/
#define STARPU_COO_GET_NY(interface) (((struct starpu_coo_interface *)(interface))->ny)
/**
   Return the number of values registered in the matrix designated
   by \p interface.
*/
#define STARPU_COO_GET_NVALUES(interface) (((struct starpu_coo_interface *)(interface))->n_values)
/**
   Return the size of the elements registered into the matrix
   designated by \p interface.
*/
#define STARPU_COO_GET_ELEMSIZE(interface) (((struct starpu_coo_interface *)(interface))->elemsize)

/** @} */

/**
   @name Block Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_block_ops;

/* TODO: rename to 3dmatrix? */
/* TODO: add allocsize support */
/**
   Block interface for 3D dense blocks
*/
struct starpu_block_interface
{
	enum starpu_data_interface_id id; /**< identifier of the interface */

	uintptr_t ptr;	      /**< local pointer of the block */
	uintptr_t dev_handle; /**< device handle of the block. */
	size_t offset;	      /**< offset in the block. */
	uint32_t nx;	      /**< number of elements on the x-axis of the block. */
	uint32_t ny;	      /**< number of elements on the y-axis of the block. */
	uint32_t nz;	      /**< number of elements on the z-axis of the block. */
	uint32_t ldy;	      /**< number of elements between two lines */
	uint32_t ldz;	      /**< number of elements between two planes */
	size_t elemsize;      /**< size of the elements of the block. */
};

/**
   Register the \p nx x \p ny x \p nz 3D matrix of \p elemsize byte elements
   pointed by \p ptr and initialize \p handle to represent it. Again, \p ldy and
   \p ldz specify the number of elements between rows and between z planes.

   Here an example of how to use the function.
   \code{.c}
   float *block;
   starpu_data_handle_t block_handle;
   block = (float*)malloc(nx*ny*nz*sizeof(float));
   starpu_block_data_register(&block_handle, STARPU_MAIN_RAM, (uintptr_t)block, nx, nx*ny, nx, ny, nz, sizeof(float));
   \endcode

   See \ref BlockDataInterface for more details.
*/
void starpu_block_data_register(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx, uint32_t ny, uint32_t nz, size_t elemsize);

/**
   Register into the \p handle that to store data on node \p node it should use the
   buffer located at \p ptr, or device handle \p dev_handle and offset \p offset
   (for OpenCL, notably), with \p ldy elements between rows and \p ldz
   elements between z planes.
*/
void starpu_block_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ldy, uint32_t ldz);

/**
   Return the number of elements on the x-axis of the block
   designated by \p handle.
 */
uint32_t starpu_block_get_nx(starpu_data_handle_t handle);

/**
   Return the number of elements on the y-axis of the block
   designated by \p handle.
 */
uint32_t starpu_block_get_ny(starpu_data_handle_t handle);

/**
   Return the number of elements on the z-axis of the block
   designated by \p handle.
 */
uint32_t starpu_block_get_nz(starpu_data_handle_t handle);

/**
   Return the number of elements between each row of the block
   designated by \p handle, in the format of the current memory node.
*/
uint32_t starpu_block_get_local_ldy(starpu_data_handle_t handle);

/**
   Return the number of elements between each z plane of the block
   designated by \p handle, in the format of the current memory node.
 */
uint32_t starpu_block_get_local_ldz(starpu_data_handle_t handle);

/**
   Return the local pointer associated with \p handle.
 */
uintptr_t starpu_block_get_local_ptr(starpu_data_handle_t handle);

/**
   Return the size of the elements of the block designated by
   \p handle.
 */
size_t starpu_block_get_elemsize(starpu_data_handle_t handle);

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_BLOCK_CHECK(interface)	STARPU_ASSERT_MSG((((struct starpu_block_interface *)(interface))->id) == STARPU_BLOCK_INTERFACE_ID, "Error. The given data is not a block.")
#define STARPU_BLOCK_GET_PTR(interface) (                              \
	{                                                              \
		STARPU_BLOCK_CHECK(interface);                         \
		(((struct starpu_block_interface *)(interface))->ptr); \
	})
#define STARPU_BLOCK_GET_DEV_HANDLE(interface) (                              \
	{                                                                     \
		STARPU_BLOCK_CHECK(interface);                                \
		(((struct starpu_block_interface *)(interface))->dev_handle); \
	})
#define STARPU_BLOCK_GET_OFFSET(interface) (                              \
	{                                                                 \
		STARPU_BLOCK_CHECK(interface);                            \
		(((struct starpu_block_interface *)(interface))->offset); \
	})
#define STARPU_BLOCK_GET_NX(interface) (                              \
	{                                                             \
		STARPU_BLOCK_CHECK(interface);                        \
		(((struct starpu_block_interface *)(interface))->nx); \
	})
#define STARPU_BLOCK_GET_NY(interface) (                              \
	{                                                             \
		STARPU_BLOCK_CHECK(interface);                        \
		(((struct starpu_block_interface *)(interface))->ny); \
	})
#define STARPU_BLOCK_GET_NZ(interface) (                              \
	{                                                             \
		STARPU_BLOCK_CHECK(interface);                        \
		(((struct starpu_block_interface *)(interface))->nz); \
	})
#define STARPU_BLOCK_GET_LDY(interface) (                              \
	{                                                              \
		STARPU_BLOCK_CHECK(interface);                         \
		(((struct starpu_block_interface *)(interface))->ldy); \
	})
#define STARPU_BLOCK_GET_LDZ(interface) (                              \
	{                                                              \
		STARPU_BLOCK_CHECK(interface);                         \
		(((struct starpu_block_interface *)(interface))->ldz); \
	})
#define STARPU_BLOCK_GET_ELEMSIZE(interface) (                              \
	{                                                                   \
		STARPU_BLOCK_CHECK(interface);                              \
		(((struct starpu_block_interface *)(interface))->elemsize); \
	})
#else
/**
   Return a pointer to the block designated by \p interface.
 */
#define STARPU_BLOCK_GET_PTR(interface)	       (((struct starpu_block_interface *)(interface))->ptr)
/**
   Return a device handle for the block designated by \p interface,
   to be used on OpenCL. The offset returned by
   ::STARPU_BLOCK_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_BLOCK_GET_DEV_HANDLE(interface) (((struct starpu_block_interface *)(interface))->dev_handle)
/**
   Return the offset in the block designated by \p interface, to be
   used with the device handle.
 */
#define STARPU_BLOCK_GET_OFFSET(interface)     (((struct starpu_block_interface *)(interface))->offset)
/**
   Return the number of elements on the x-axis of the block
   designated by \p interface.
 */
#define STARPU_BLOCK_GET_NX(interface)	       (((struct starpu_block_interface *)(interface))->nx)
/**
   Return the number of elements on the y-axis of the block
   designated by \p interface.
 */
#define STARPU_BLOCK_GET_NY(interface)	       (((struct starpu_block_interface *)(interface))->ny)
/**
Return the number of elements on the z-axis of the block
designated by \p interface.
 */
#define STARPU_BLOCK_GET_NZ(interface)	       (((struct starpu_block_interface *)(interface))->nz)
/**
   Return the number of elements between each row of the block
   designated by \p interface. May be equal to nx when there is no padding.
 */
#define STARPU_BLOCK_GET_LDY(interface)	       (((struct starpu_block_interface *)(interface))->ldy)
/**
   Return the number of elements between each z plane of the block
   designated by \p interface. May be equal to nx*ny when there is no
   padding.
 */
#define STARPU_BLOCK_GET_LDZ(interface)	       (((struct starpu_block_interface *)(interface))->ldz)
/**
   Return the size of the elements of the block designated by
   \p interface.
 */
#define STARPU_BLOCK_GET_ELEMSIZE(interface)   (((struct starpu_block_interface *)(interface))->elemsize)
#endif

/** @} */

/**
   @name Tensor Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_tensor_ops;

/* TODO: rename to 4dtensor? */
/* TODO: add allocsize support */
/**
   Tensor interface for 4D dense tensors
*/
struct starpu_tensor_interface
{
	enum starpu_data_interface_id id; /**< identifier of the interface */

	uintptr_t ptr;	      /**< local pointer of the tensor */
	uintptr_t dev_handle; /**< device handle of the tensor. */
	size_t offset;	      /**< offset in the tensor. */
	uint32_t nx;	      /**< number of elements on the x-axis of the tensor. */
	uint32_t ny;	      /**< number of elements on the y-axis of the tensor. */
	uint32_t nz;	      /**< number of elements on the z-axis of the tensor. */
	uint32_t nt;	      /**< number of elements on the t-axis of the tensor. */
	uint32_t ldy;	      /**< number of elements between two lines */
	uint32_t ldz;	      /**< number of elements between two planes */
	uint32_t ldt;	      /**< number of elements between two cubes */
	size_t elemsize;      /**< size of the elements of the tensor. */
};

/**
   Register the \p nx x \p ny x \p nz x \p nt 4D tensor of \p elemsize byte elements
   pointed by \p ptr and initialize \p handle to represent it. Again, \p ldy,
   \p ldz, and \p ldt specify the number of elements between rows, between z planes and between t cubes.

   Here an example of how to use the function.
   \code{.c}
   float *tensor;
   starpu_data_handle_t tensor_handle;
   tensor = (float*)malloc(nx*ny*nz*nt*sizeof(float));
   starpu_tensor_data_register(&tensor_handle, STARPU_MAIN_RAM, (uintptr_t)tensor, nx, nx*ny, nx*ny*nz, nx, ny, nz, nt, sizeof(float));
   \endcode

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_data_register(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t ldt, uint32_t nx, uint32_t ny, uint32_t nz, uint32_t nt, size_t elemsize);

/**
   Register into the \p handle that to store data on node \p node it should use the
   buffer located at \p ptr, or device handle \p dev_handle and offset \p offset
   (for OpenCL, notably), with \p ldy elements between rows, and \p ldz
   elements between z planes, and \p ldt elements between t cubes.
*/
void starpu_tensor_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ldy, uint32_t ldz, uint32_t ldt);

/**
   Return the number of elements on the x-axis of the tensor
   designated by \p handle.
 */
uint32_t starpu_tensor_get_nx(starpu_data_handle_t handle);

/**
   Return the number of elements on the y-axis of the tensor
   designated by \p handle.
 */
uint32_t starpu_tensor_get_ny(starpu_data_handle_t handle);

/**
   Return the number of elements on the z-axis of the tensor
   designated by \p handle.
 */
uint32_t starpu_tensor_get_nz(starpu_data_handle_t handle);

/**
   Return the number of elements on the t-axis of the tensor
   designated by \p handle.
 */
uint32_t starpu_tensor_get_nt(starpu_data_handle_t handle);

/**
   Return the number of elements between each row of the tensor
   designated by \p handle, in the format of the current memory node.
*/
uint32_t starpu_tensor_get_local_ldy(starpu_data_handle_t handle);

/**
   Return the number of elements between each z plane of the tensor
   designated by \p handle, in the format of the current memory node.
 */
uint32_t starpu_tensor_get_local_ldz(starpu_data_handle_t handle);

/**
   Return the number of elements between each t cubes of the tensor
   designated by \p handle, in the format of the current memory node.
 */
uint32_t starpu_tensor_get_local_ldt(starpu_data_handle_t handle);

/**
   Return the local pointer associated with \p handle.
 */
uintptr_t starpu_tensor_get_local_ptr(starpu_data_handle_t handle);

/**
   Return the size of the elements of the tensor designated by
   \p handle.
 */
size_t starpu_tensor_get_elemsize(starpu_data_handle_t handle);

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_TENSOR_CHECK(interface)	 STARPU_ASSERT_MSG((((struct starpu_tensor_interface *)(interface))->id) == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a tensor.")
#define STARPU_TENSOR_GET_PTR(interface) (                              \
	{                                                               \
		STARPU_TENSOR_CHECK(interface);                         \
		(((struct starpu_tensor_interface *)(interface))->ptr); \
	})
#define STARPU_TENSOR_GET_DEV_HANDLE(interface) (                              \
	{                                                                      \
		STARPU_TENSOR_CHECK(interface);                                \
		(((struct starpu_tensor_interface *)(interface))->dev_handle); \
	})
#define STARPU_TENSOR_GET_OFFSET(interface) (                              \
	{                                                                  \
		STARPU_TENSOR_CHECK(interface);                            \
		(((struct starpu_tensor_interface *)(interface))->offset); \
	})
#define STARPU_TENSOR_GET_NX(interface) (                              \
	{                                                              \
		STARPU_TENSOR_CHECK(interface);                        \
		(((struct starpu_tensor_interface *)(interface))->nx); \
	})
#define STARPU_TENSOR_GET_NY(interface) (                              \
	{                                                              \
		STARPU_TENSOR_CHECK(interface);                        \
		(((struct starpu_tensor_interface *)(interface))->ny); \
	})
#define STARPU_TENSOR_GET_NZ(interface) (                              \
	{                                                              \
		STARPU_TENSOR_CHECK(interface);                        \
		(((struct starpu_tensor_interface *)(interface))->nz); \
	})
#define STARPU_TENSOR_GET_NT(interface) (                              \
	{                                                              \
		STARPU_TENSOR_CHECK(interface);                        \
		(((struct starpu_tensor_interface *)(interface))->nt); \
	})
#define STARPU_TENSOR_GET_LDY(interface) (                              \
	{                                                               \
		STARPU_TENSOR_CHECK(interface);                         \
		(((struct starpu_tensor_interface *)(interface))->ldy); \
	})
#define STARPU_TENSOR_GET_LDZ(interface) (                              \
	{                                                               \
		STARPU_TENSOR_CHECK(interface);                         \
		(((struct starpu_tensor_interface *)(interface))->ldz); \
	})
#define STARPU_TENSOR_GET_LDT(interface) (                              \
	{                                                               \
		STARPU_TENSOR_CHECK(interface);                         \
		(((struct starpu_tensor_interface *)(interface))->ldt); \
	})
#define STARPU_TENSOR_GET_ELEMSIZE(interface) (                              \
	{                                                                    \
		STARPU_TENSOR_CHECK(interface);                              \
		(((struct starpu_tensor_interface *)(interface))->elemsize); \
	})
#else
/**
   Return a pointer to the tensor designated by \p interface.
 */
#define STARPU_TENSOR_GET_PTR(interface)	(((struct starpu_tensor_interface *)(interface))->ptr)
/**
   Return a device handle for the tensor designated by \p interface,
   to be used on OpenCL. The offset returned by
   ::STARPU_TENSOR_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_TENSOR_GET_DEV_HANDLE(interface) (((struct starpu_tensor_interface *)(interface))->dev_handle)
/**
   Return the offset in the tensor designated by \p interface, to be
   used with the device handle.
 */
#define STARPU_TENSOR_GET_OFFSET(interface)	(((struct starpu_tensor_interface *)(interface))->offset)
/**
   Return the number of elements on the x-axis of the tensor
   designated by \p interface.
 */
#define STARPU_TENSOR_GET_NX(interface)		(((struct starpu_tensor_interface *)(interface))->nx)
/**
   Return the number of elements on the y-axis of the tensor
   designated by \p interface.
 */
#define STARPU_TENSOR_GET_NY(interface)		(((struct starpu_tensor_interface *)(interface))->ny)
/**
Return the number of elements on the z-axis of the tensor
designated by \p interface.
 */
#define STARPU_TENSOR_GET_NZ(interface)		(((struct starpu_tensor_interface *)(interface))->nz)
/**
Return the number of elements on the t-axis of the tensor
designated by \p interface.
 */
#define STARPU_TENSOR_GET_NT(interface)		(((struct starpu_tensor_interface *)(interface))->nt)
/**
   Return the number of elements between each row of the tensor
   designated by \p interface. May be equal to nx when there is no padding.
 */
#define STARPU_TENSOR_GET_LDY(interface)	(((struct starpu_tensor_interface *)(interface))->ldy)
/**
   Return the number of elements between each z plane of the tensor
   designated by \p interface. May be equal to nx*ny when there is no
   padding.
 */
#define STARPU_TENSOR_GET_LDZ(interface)	(((struct starpu_tensor_interface *)(interface))->ldz)
/**
   Return the number of elements between each t cubes of the tensor
   designated by \p interface. May be equal to nx*ny*nz when there is no
   padding.
 */
#define STARPU_TENSOR_GET_LDT(interface)	(((struct starpu_tensor_interface *)(interface))->ldt)
/**
   Return the size of the elements of the tensor designated by
   \p interface.
 */
#define STARPU_TENSOR_GET_ELEMSIZE(interface)	(((struct starpu_tensor_interface *)(interface))->elemsize)
#endif

/** @} */

/**
   @name Ndim Array Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_ndim_ops;

/**
   ndim interface for ndim array
*/
struct starpu_ndim_interface
{
	enum starpu_data_interface_id id; /**< identifier of the interface */

	uintptr_t ptr;	      /**< local pointer of the ndim */
	uintptr_t dev_handle; /**< device handle of the ndim. */
	size_t offset;	      /**< offset in the ndim. */
	size_t allocsize;     /**< size actually currently allocated. */
	uint32_t *nn;	      /**< array of element number on each dimension */
	uint32_t *ldn;	      /**< array of element number between two units on each dimension */
	size_t ndim;	      /**< size of the dimension. */
	size_t elemsize;      /**< size of the elements of the ndim. */
};

/**
   Register the \p nn[0] x \p nn[1] x ... \p ndim-dimension matrix of \p elemsize byte elements
   pointed by \p ptr and initialize \p handle to represent it. Again, \p ldn,
   specifies the number of elements between two units on each dimension.

   Here an example of how to use the function.
   \code{.c}
   float *ndim_arr;
   size_t arrsize = 1;
	int i;
	for (i = 0; i < ndim; i++)
	   arrsize = arrsize * nn[i];
   starpu_data_handle_t ndim_handle;
   ndim_arr = (float*)malloc(arrsize*sizeof(float));
   starpu_ndim_data_register(&ndim_handle, STARPU_MAIN_RAM, (uintptr_t)ndim_arr, ldn, nn, ndim, sizeof(float));
   \endcode

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_data_register(starpu_data_handle_t *handleptr, int home_node, uintptr_t ptr, uint32_t *ldn, uint32_t *nn, size_t ndim, size_t elemsize);
/**
   Register into the \p handle that to store data on node \p node it should use the
   buffer located at \p ptr, or device handle \p dev_handle and offset \p offset
   (for OpenCL, notably), with \p ldn elements between two units on each dimension.
*/
void starpu_ndim_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t *ldn);

/**
   Return the number of elements on each dimension of the ndim array
   designated by \p handle.
 */
uint32_t *starpu_ndim_get_nn(starpu_data_handle_t handle);

/**
   Return the number of elements on the i-axis of the ndim array
   designated by \p handle. When i=0, it means x-axis,
   when i=1, it means y-axis, when i=2, it means z-axis, etc.
 */
uint32_t starpu_ndim_get_ni(starpu_data_handle_t handle, size_t i);

/**
   Return the number of elements between two units on each dimension of the ndim array
   designated by \p handle, in the format of the current memory node.
*/
uint32_t *starpu_ndim_get_local_ldn(starpu_data_handle_t handle);

/**
   Return the number of elements between two units i-axis dimension of the ndim array
   designated by \p handle, in the format of the current memory node.
*/
uint32_t starpu_ndim_get_local_ldi(starpu_data_handle_t handle, size_t i);

/**
   Return the local pointer associated with \p handle.
 */
uintptr_t starpu_ndim_get_local_ptr(starpu_data_handle_t handle);

/**
	Return the dimension size.
*/
size_t starpu_ndim_get_ndim(starpu_data_handle_t handle);

/**
   Return the size of the elements of the ndim array designated by
   \p handle.
 */
size_t starpu_ndim_get_elemsize(starpu_data_handle_t handle);

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_NDIM_CHECK(interface)   STARPU_ASSERT_MSG((((struct starpu_ndim_interface *)(interface))->id) == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim.")
#define STARPU_NDIM_GET_PTR(interface) (                              \
	{                                                             \
		STARPU_NDIM_CHECK(interface);                         \
		(((struct starpu_ndim_interface *)(interface))->ptr); \
	})
#define STARPU_NDIM_GET_DEV_HANDLE(interface) (                              \
	{                                                                    \
		STARPU_NDIM_CHECK(interface);                                \
		(((struct starpu_ndim_interface *)(interface))->dev_handle); \
	})
#define STARPU_NDIM_GET_OFFSET(interface) (                              \
	{                                                                \
		STARPU_NDIM_CHECK(interface);                            \
		(((struct starpu_ndim_interface *)(interface))->offset); \
	})
#define STARPU_NDIM_GET_NN(interface) (                              \
	{                                                            \
		STARPU_NDIM_CHECK(interface);                        \
		(((struct starpu_ndim_interface *)(interface))->nn); \
	})
#define STARPU_NDIM_GET_LDN(interface) (                              \
	{                                                             \
		STARPU_NDIM_CHECK(interface);                         \
		(((struct starpu_ndim_interface *)(interface))->ldn); \
	})
#define STARPU_NDIM_GET_NDIM(interface) (                              \
	{                                                              \
		STARPU_NDIM_CHECK(interface);                          \
		(((struct starpu_ndim_interface *)(interface))->ndim); \
	})
#define STARPU_NDIM_GET_ELEMSIZE(interface) (                              \
	{                                                                  \
		STARPU_NDIM_CHECK(interface);                              \
		(((struct starpu_ndim_interface *)(interface))->elemsize); \
	})
#else
/**
   Return a pointer to the ndim array designated by \p interface.
 */
#define STARPU_NDIM_GET_PTR(interface)	      (((struct starpu_ndim_interface *)(interface))->ptr)
/**
   Return a device handle for the ndim array designated by \p interface,
   to be used on OpenCL. The offset returned by
   ::STARPU_NDIM_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_NDIM_GET_DEV_HANDLE(interface) (((struct starpu_ndim_interface *)(interface))->dev_handle)
/**
   Return the offset in the ndim designated by \p interface, to be
   used with the device handle.
 */
#define STARPU_NDIM_GET_OFFSET(interface)     (((struct starpu_ndim_interface *)(interface))->offset)
/**
   Return the number of elements on each dimension of the ndim array
   designated by \p interface.
 */
#define STARPU_NDIM_GET_NN(interface)	      (((struct starpu_ndim_interface *)(interface))->nn)
/**
   Return the number of elements between each two units on each dimension of the ndim array
   designated by \p interface. May be equal to nx when there is no padding.
 */
#define STARPU_NDIM_GET_LDN(interface)	      (((struct starpu_ndim_interface *)(interface))->ldn)
/**
   Return the dimension size of the ndim array designated by
   \p interface.
 */
#define STARPU_NDIM_GET_NDIM(interface)	      (((struct starpu_ndim_interface *)(interface))->ndim)
/**
   Return the size of the elements of the ndim array designated by
   \p interface.
 */
#define STARPU_NDIM_GET_ELEMSIZE(interface)   (((struct starpu_ndim_interface *)(interface))->elemsize)
#endif

/** @} */

/**
   @name Vector Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_vector_ops;

/**
 */
struct starpu_vector_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */

	uintptr_t ptr;	      /**< local pointer of the vector */
	uintptr_t dev_handle; /**< device handle of the vector. */
	size_t offset;	      /**< offset in the vector */
	uint32_t nx;	      /**< number of elements on the x-axis of the vector */
	size_t elemsize;      /**< size of the elements of the vector */
	uint32_t slice_base;  /**< vector slice base, used by the StarPU OpenMP runtime support */
	size_t allocsize;     /**< size actually currently allocated */
};

/**
   Register the \p nx \p elemsize-byte elements pointed to by \p ptr and initialize \p handle to represent it.

   Here an example of how to use the function.
   \code{.c}
   float vector[NX];
   starpu_data_handle_t vector_handle;
   starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));
   \endcode

   See \ref VectorDataInterface for more details.
 */
void starpu_vector_data_register(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, uint32_t nx, size_t elemsize);

/**
   Similar to starpu_vector_data_register, but additionally specifies which
   allocation size should be used instead of the initial nx*elemsize.
   See \ref VariableSizeDataInterface for more details.
*/
void starpu_vector_data_register_allocsize(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, uint32_t nx, size_t elemsize, size_t allocsize);

/**
   Register into the \p handle that to store data on node \p node it should use the
   buffer located at \p ptr, or device handle \p dev_handle and offset \p offset
   (for OpenCL, notably)
*/
void starpu_vector_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset);

/**
   Return the number of elements registered into the array designated by \p handle.
 */
uint32_t starpu_vector_get_nx(starpu_data_handle_t handle);

/**
   Return the size of each element of the array designated by \p handle.
 */
size_t starpu_vector_get_elemsize(starpu_data_handle_t handle);

/**
  Return the allocated size of the array designated by \p handle.
 */
size_t starpu_vector_get_allocsize(starpu_data_handle_t handle);

/**
   Return the local pointer associated with \p handle.
 */
uintptr_t starpu_vector_get_local_ptr(starpu_data_handle_t handle);

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_VECTOR_CHECK(interface)	 STARPU_ASSERT_MSG((((struct starpu_vector_interface *)(interface))->id) == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.")
#define STARPU_VECTOR_GET_PTR(interface) (                              \
	{                                                               \
		STARPU_VECTOR_CHECK(interface);                         \
		(((struct starpu_vector_interface *)(interface))->ptr); \
	})
#define STARPU_VECTOR_GET_DEV_HANDLE(interface) (                              \
	{                                                                      \
		STARPU_VECTOR_CHECK(interface);                                \
		(((struct starpu_vector_interface *)(interface))->dev_handle); \
	})
#define STARPU_VECTOR_GET_OFFSET(interface) (                              \
	{                                                                  \
		STARPU_VECTOR_CHECK(interface);                            \
		(((struct starpu_vector_interface *)(interface))->offset); \
	})
#define STARPU_VECTOR_GET_NX(interface) (                              \
	{                                                              \
		STARPU_VECTOR_CHECK(interface);                        \
		(((struct starpu_vector_interface *)(interface))->nx); \
	})
#define STARPU_VECTOR_GET_ELEMSIZE(interface) (                              \
	{                                                                    \
		STARPU_VECTOR_CHECK(interface);                              \
		(((struct starpu_vector_interface *)(interface))->elemsize); \
	})
#define STARPU_VECTOR_GET_ALLOCSIZE(interface) (                              \
	{                                                                     \
		STARPU_VECTOR_CHECK(interface);                               \
		(((struct starpu_vector_interface *)(interface))->allocsize); \
	})
#define STARPU_VECTOR_GET_SLICE_BASE(interface) (                              \
	{                                                                      \
		STARPU_VECTOR_CHECK(interface);                                \
		(((struct starpu_vector_interface *)(interface))->slice_base); \
	})
#else
/**
   Return a pointer to the array designated by \p interface, valid on
   CPUs and CUDA only. For OpenCL, the device handle and offset need to
   be used instead.
 */
#define STARPU_VECTOR_GET_PTR(interface)	(((struct starpu_vector_interface *)(interface))->ptr)
/**
   Return a device handle for the array designated by \p interface,
   to be used with OpenCL. the offset returned by ::STARPU_VECTOR_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_VECTOR_GET_DEV_HANDLE(interface) (((struct starpu_vector_interface *)(interface))->dev_handle)
/**
   Return the offset in the array designated by \p interface, to be
   used with the device handle.
*/
#define STARPU_VECTOR_GET_OFFSET(interface)	(((struct starpu_vector_interface *)(interface))->offset)
/**
   Return the number of elements registered into the array
   designated by \p interface.
 */
#define STARPU_VECTOR_GET_NX(interface)		(((struct starpu_vector_interface *)(interface))->nx)
/**
   Return the size of each element of the array designated by
   \p interface.
 */
#define STARPU_VECTOR_GET_ELEMSIZE(interface)	(((struct starpu_vector_interface *)(interface))->elemsize)
/**
   Return the size of each element of the array designated by
   \p interface.
 */
#define STARPU_VECTOR_GET_ALLOCSIZE(interface)	(((struct starpu_vector_interface *)(interface))->allocsize)
/**
   Return the OpenMP slice base annotation of each element of the array designated by
   \p interface.
 */
#define STARPU_VECTOR_GET_SLICE_BASE(interface) (((struct starpu_vector_interface *)(interface))->slice_base)
#endif

/**
   Set the number of elements registered into the array designated by \p
   interface.
 */
#define STARPU_VECTOR_SET_NX(interface, newnx)                                   \
	do {                                                                     \
		STARPU_VECTOR_CHECK(interface);                                  \
		(((struct starpu_vector_interface *)(interface))->nx) = (newnx); \
	}                                                                        \
	while (0)

/** @} */

/**
   @name Variable Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_variable_ops;

/**
   Variable interface for a single data (not a vector, a matrix, a list,
   ...)
 */
struct starpu_variable_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */

	uintptr_t ptr;	      /**< local pointer of the variable */
	uintptr_t dev_handle; /**< device handle of the variable. */
	size_t offset;	      /**< offset in the variable */
	size_t elemsize;      /**< size of the variable */
};

/**
   Register the \p size byte element pointed to by \p ptr, which is
   typically a scalar, and initialize \p handle to represent this data item.

   Here an example of how to use the function.
   \code{.c}
   float var = 42.0;
   starpu_data_handle_t var_handle;
   starpu_variable_data_register(&var_handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));
   \endcode

   See \ref VariableDataInterface for more details.
*/
void starpu_variable_data_register(starpu_data_handle_t *handle, int home_node, uintptr_t ptr, size_t size);

/**
   Register into the \p handle that to store data on node \p node it should use the
   buffer located at \p ptr, or device handle \p dev_handle and offset \p offset
   (for OpenCL, notably)
 */
void starpu_variable_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset);

/**
   Return the size of the variable designated by \p handle.
 */
size_t starpu_variable_get_elemsize(starpu_data_handle_t handle);

/**
   Return a pointer to the variable designated by \p handle.
 */
uintptr_t starpu_variable_get_local_ptr(starpu_data_handle_t handle);

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_VARIABLE_CHECK(interface)   STARPU_ASSERT_MSG((((struct starpu_variable_interface *)(interface))->id) == STARPU_VARIABLE_INTERFACE_ID, "Error. The given data is not a variable.")
#define STARPU_VARIABLE_GET_PTR(interface) (                              \
	{                                                                 \
		STARPU_VARIABLE_CHECK(interface);                         \
		(((struct starpu_variable_interface *)(interface))->ptr); \
	})
#define STARPU_VARIABLE_GET_OFFSET(interface) (                              \
	{                                                                    \
		STARPU_VARIABLE_CHECK(interface);                            \
		(((struct starpu_variable_interface *)(interface))->offset); \
	})
#define STARPU_VARIABLE_GET_ELEMSIZE(interface) (                              \
	{                                                                      \
		STARPU_VARIABLE_CHECK(interface);                              \
		(((struct starpu_variable_interface *)(interface))->elemsize); \
	})
#define STARPU_VARIABLE_GET_DEV_HANDLE(interface) (                       \
	{                                                                 \
		STARPU_VARIABLE_CHECK(interface);                         \
		(((struct starpu_variable_interface *)(interface))->ptr); \
	})
#else
/**
   Return a pointer to the variable designated by \p interface.
 */
#define STARPU_VARIABLE_GET_PTR(interface)	  (((struct starpu_variable_interface *)(interface))->ptr)
/**
   Return the offset in the variable designated by \p interface, to
   be used with the device handle.
 */
#define STARPU_VARIABLE_GET_OFFSET(interface)	  (((struct starpu_variable_interface *)(interface))->offset)
/**
   Return the size of the variable designated by \p interface.
 */
#define STARPU_VARIABLE_GET_ELEMSIZE(interface)	  (((struct starpu_variable_interface *)(interface))->elemsize)
/**
   Return a device handle for the variable designated by
   \p interface, to be used with OpenCL. The offset returned by
   ::STARPU_VARIABLE_GET_OFFSET has to be
   used in addition to this.
 */
#define STARPU_VARIABLE_GET_DEV_HANDLE(interface) (((struct starpu_variable_interface *)(interface))->ptr)
#endif

/** @} */

/**
   @name Void Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_void_ops;

/**
   Register a void interface. There is no data really associated
   to that interface, but it may be used as a synchronization mechanism.
   It also permits to express an abstract piece of data that is managed
   by the application internally: this makes it possible to forbid the
   concurrent execution of different tasks accessing the same <c>void</c>
   data in read-write concurrently.
   See \ref DataHandlesHelpers for more details.
 */
void starpu_void_data_register(starpu_data_handle_t *handle);

/** @} */

/**
   @name CSR Data Interface
   @{
 */

extern struct starpu_data_interface_ops starpu_interface_csr_ops;

/**
   CSR interface for sparse matrices (compressed sparse row
   representation)
 */
struct starpu_csr_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */

	uint32_t nnz;	      /**< number of non-zero entries */
	uint32_t nrow;	      /**< number of rows */
	uintptr_t nzval;      /**< non-zero values */
	uint32_t *colind;     /**< position of non-zero entries on the row */
	uint32_t *rowptr;     /**< index (in nzval) of the first entry of the row */
	uint32_t *ram_colind; /**< position of non-zero entries on the row (stored in RAM) */
	uint32_t *ram_rowptr; /**< index (in nzval) of the first entry of the row (stored in RAM) */

	uint32_t firstentry; /**< k for k-based indexing (0 or 1 usually). also useful when partitioning the matrix. */

	size_t elemsize; /**< size of the elements of the matrix */
};

/**
   Register a CSR (Compressed Sparse Row Representation) sparse matrix.
   See \ref CSRDataInterface for more details.
 */
void starpu_csr_data_register(starpu_data_handle_t *handle, int home_node, uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize);

/**
   Return the number of non-zero values in the matrix designated
   by \p handle.
 */
uint32_t starpu_csr_get_nnz(starpu_data_handle_t handle);

/**
   Return the size of the row pointer array of the matrix
   designated by \p handle.
 */
uint32_t starpu_csr_get_nrow(starpu_data_handle_t handle);

/**
   Return the index at which all arrays (the column indexes, the
   row pointers...) of the matrix designated by \p handle.
 */
uint32_t starpu_csr_get_firstentry(starpu_data_handle_t handle);

/**
   Return a local pointer to the non-zero values of the matrix
   designated by \p handle.
 */
uintptr_t starpu_csr_get_local_nzval(starpu_data_handle_t handle);

/**
   Return a local pointer to the column index of the matrix
   designated by \p handle.
 */
uint32_t *starpu_csr_get_local_colind(starpu_data_handle_t handle);

/**
   Return a local pointer to the row pointer array of the matrix
   designated by \p handle.
 */
uint32_t *starpu_csr_get_local_rowptr(starpu_data_handle_t handle);

/**
   Return the size of the elements registered into the matrix
   designated by \p handle.
 */
size_t starpu_csr_get_elemsize(starpu_data_handle_t handle);

/**
   Return the number of non-zero values in the matrix designated
   by \p interface.
 */
#define STARPU_CSR_GET_NNZ(interface) (((struct starpu_csr_interface *)(interface))->nnz)
/**
   Return the size of the row pointer array of the matrix
   designated by \p interface.
 */
#define STARPU_CSR_GET_NROW(interface) (((struct starpu_csr_interface *)(interface))->nrow)
/**
   Return a pointer to the non-zero values of the matrix
   designated by \p interface.
 */
#define STARPU_CSR_GET_NZVAL(interface) (((struct starpu_csr_interface *)(interface))->nzval)
/**
   Return a device handle for the array of non-zero values in the
   matrix designated by \p interface. The offset returned by ::STARPU_CSR_GET_OFFSET
   has to used in addition to this.
 */
#define STARPU_CSR_GET_NZVAL_DEV_HANDLE(interface) (((struct starpu_csr_interface *)(interface))->nnz)
/**
   Return a pointer to the column index of the matrix designated
   by \p interface.
 */
#define STARPU_CSR_GET_COLIND(interface) (((struct starpu_csr_interface *)(interface))->colind)
/**
   Return a RAM pointer to the column index of the matrix designated
   by \p interface.
 */
#define STARPU_CSR_GET_RAM_COLIND(interface) (((struct starpu_csr_interface *)(interface))->ram_colind)
/**
   Return a device handle for the column index of the matrix
   designated by \p interface. The offset returned by ::STARPU_CSR_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_CSR_GET_COLIND_DEV_HANDLE(interface) (((struct starpu_csr_interface *)(interface))->colind)
/**
   Return a pointer to the row pointer array of the matrix
   designated by \p interface.
 */
#define STARPU_CSR_GET_ROWPTR(interface) (((struct starpu_csr_interface *)(interface))->rowptr)
/**
   Return a RAM pointer to the row pointer array of the matrix
   designated by \p interface.
 */
#define STARPU_CSR_GET_RAM_ROWPTR(interface) (((struct starpu_csr_interface *)(interface))->ram_rowptr)
/**
   Return a device handle for the row pointer array of the matrix
   designated by \p interface. The offset returned by ::STARPU_CSR_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_CSR_GET_ROWPTR_DEV_HANDLE(interface) (((struct starpu_csr_interface *)(interface))->rowptr)
/**
   Return the offset in the arrays (colind, rowptr, nzval) of the
   matrix designated by \p interface, to be used with the device handles.
 */
#define STARPU_CSR_GET_OFFSET 0
/**
   Return the index at which all arrays (the column indexes, the
   row pointers...) of the \p interface start.
 */
#define STARPU_CSR_GET_FIRSTENTRY(interface) (((struct starpu_csr_interface *)(interface))->firstentry)
/**
   Return the size of the elements registered into the matrix
   designated by \p interface.
 */
#define STARPU_CSR_GET_ELEMSIZE(interface) (((struct starpu_csr_interface *)(interface))->elemsize)

/** @} */

/**
   @name BCSR Data Interface
   @{
*/

extern struct starpu_data_interface_ops starpu_interface_bcsr_ops;

/**
   BCSR interface for sparse matrices (blocked compressed sparse
   row representation)

   Note: when a BCSR matrix is partitioned, nzval, colind, and rowptr point into
   the corresponding father arrays. The rowptr content is thus the same as the
   father's. Firstentry is used to offset this so it becomes valid for the child
   arrays.
*/
struct starpu_bcsr_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */

	uint32_t nnz;  /**< number of non-zero BLOCKS */
	uint32_t nrow; /**< number of rows (in terms of BLOCKS) */

	uintptr_t nzval;      /**< non-zero values: nnz blocks of r*c elements */
	uint32_t *colind;     /**< array of nnz elements, colind[i] is the block-column index for block i in nzval */
	uint32_t *rowptr;     /**< array of nrow+1
					   * elements, rowptr[i] is
					   * the block-index (in
					   * nzval) of the first block
					   * of row i. By convention,
					   * rowptr[nrow] is the
					   * number of blocks, this
					   * allows an easier access
					   * of the matrix's elements
					   * for the kernels. */
	uint32_t *ram_colind; /**< array of nnz elements (stored in RAM) */
	uint32_t *ram_rowptr; /**< array of nrow+1 elements (stored in RAM) */

	uint32_t firstentry; /**< k for k-based indexing (0 or 1 usually). Also useful when partitioning the matrix. */

	uint32_t r; /**< height of the blocks */
	uint32_t c; /**< width of the blocks */

	size_t elemsize; /**< size of the elements of the matrix */
};

/**
   This variant of starpu_data_register() uses the BCSR (Blocked
   Compressed Sparse Row Representation) sparse matrix interface.
   Register the sparse matrix made of \p nnz non-zero blocks of elements of
   size \p elemsize stored in \p nzval and initializes \p handle to represent it.
   Blocks have size \p r * \p c. \p nrow is the number of rows (in terms of
   blocks), \p colind is an array of nnz elements, colind[i] is the block-column index for block i in \p nzval,
   \p rowptr is an array of nrow+1 elements, rowptr[i] is the block-index (in \p nzval) of the first block of row i. By convention, rowptr[nrow] is the number of blocks, this allows an easier access of the matrix's elements for the kernels.
   \p firstentry is the index of the first entry of the given arrays
   (usually 0 or 1).

   Here an example with the following matrix:

   \code  |  0   1   0   0 | \endcode
   \code  |  2   3   0   0 | \endcode
   \code  |  4   5   8   9 | \endcode
   \code  |  6   7  10  11 | \endcode

   \code nzval  = [0, 1, 2, 3] ++ [4, 5, 6, 7] ++ [8, 9, 10, 11] \endcode
   \code colind = [0, 0, 1] \endcode
   \code rowptr = [0, 1, 3] \endcode
   \code r = c = 2 \endcode

   which translates into the following code

   \code{.c}
   int R = 2; // Size of the blocks
   int C = 2;

   int NROWS = 2;
   int NNZ_BLOCKS = 3;    // out of 4
   int NZVAL_SIZE = (R*C*NNZ_BLOCKS);

   int nzval[NZVAL_SIZE]  =
   {
     0, 1, 2, 3,    // First block
     4, 5, 6, 7,    // Second block
     8, 9, 10, 11   // Third block
   };
   uint32_t colind[NNZ_BLOCKS] =
   {
     0, // block-column index for first block in nzval
     0, // block-column index for second block in nzval
     1  // block-column index for third block in nzval
   };
   uint32_t rowptr[NROWS+1] =
   {
     0, // block-index in nzval of the first block of the first row.
     1, // block-index in nzval of the first block of the second row.
     NNZ_BLOCKS // number of blocks, to allow an easier element's access for the kernels
   };

   starpu_data_handle_t bcsr_handle;
   starpu_bcsr_data_register(&bcsr_handle,
			  STARPU_MAIN_RAM,
			  NNZ_BLOCKS,
			  NROWS,
			  (uintptr_t) nzval,
			  colind,
			  rowptr,
			  0, // firstentry
			  R,
			  C,
			  sizeof(nzval[0]));
   \endcode

   See \ref BCSRDataInterface for more details.
*/
void starpu_bcsr_data_register(starpu_data_handle_t *handle, int home_node, uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);

/**
   Return the number of non-zero elements in the matrix designated
   by \p handle.
 */
uint32_t starpu_bcsr_get_nnz(starpu_data_handle_t handle);

/**
   Return the number of rows (in terms of blocks of size r*c) in
   the matrix designated by \p handle.
 */
uint32_t starpu_bcsr_get_nrow(starpu_data_handle_t handle);

/**
   Return the index at which all arrays (the column indexes, the
   row pointers...) of the matrix desginated by \p handle.
 */
uint32_t starpu_bcsr_get_firstentry(starpu_data_handle_t handle);

/**
   Return a pointer to the non-zero values of the matrix
   designated by \p handle.
 */
uintptr_t starpu_bcsr_get_local_nzval(starpu_data_handle_t handle);

/**
   Return a pointer to the column index, which holds the positions
   of the non-zero entries in the matrix designated by \p handle.
 */
uint32_t *starpu_bcsr_get_local_colind(starpu_data_handle_t handle);

/**
   Return the row pointer array of the matrix designated by
   \p handle.
 */
uint32_t *starpu_bcsr_get_local_rowptr(starpu_data_handle_t handle);

/**
   Return the number of rows in a block.
 */
uint32_t starpu_bcsr_get_r(starpu_data_handle_t handle);

/**
   Return the number of columns in a block.
 */
uint32_t starpu_bcsr_get_c(starpu_data_handle_t handle);

/**
   Return the size of the elements in the matrix designated by
   \p handle.
 */
size_t starpu_bcsr_get_elemsize(starpu_data_handle_t handle);

/**
   Return the number of non-zero values in the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_NNZ(interface) (((struct starpu_bcsr_interface *)(interface))->nnz)
/**
   Return the number of block rows in the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_NROW(interface) (((struct starpu_bcsr_interface *)(interface))->nrow)
/**
   Return a pointer to the non-zero values of the matrix
   designated by \p interface.
 */
#define STARPU_BCSR_GET_NZVAL(interface) (((struct starpu_bcsr_interface *)(interface))->nzval)
/**
   Return a device handle for the array of non-zero values in the
   matrix designated by \p interface. The offset returned by ::STARPU_BCSR_GET_OFFSET has to be
   used in addition to this.
 */
#define STARPU_BCSR_GET_NZVAL_DEV_HANDLE(interface) (((struct starpu_bcsr_interface *)(interface))->nnz)
/**
   Return a pointer to the column index of the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_COLIND(interface) (((struct starpu_bcsr_interface *)(interface))->colind)
/**
   Return a RAM pointer to the column index of the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_RAM_COLIND(interface) (((struct starpu_bcsr_interface *)(interface))->ram_colind)
/**
   Return a device handle for the column index of the matrix
   designated by \p interface. The offset returned by ::STARPU_BCSR_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_BCSR_GET_COLIND_DEV_HANDLE(interface) (((struct starpu_bcsr_interface *)(interface))->colind)
/**
   Return a pointer to the row pointer array of the matrix
   designated by \p interface.
 */
#define STARPU_BCSR_GET_ROWPTR(interface) (((struct starpu_bcsr_interface *)(interface))->rowptr)
/**
   Return a RAM pointer to the row pointer array of the matrix
   designated by \p interface.
 */
#define STARPU_BCSR_GET_RAM_ROWPTR(interface) (((struct starpu_bcsr_interface *)(interface))->ram_rowptr)
/**
   Return a device handle for the row pointer array of the matrix
   designated by \p interface. The offset returned by ::STARPU_BCSR_GET_OFFSET has to be used in
   addition to this.
 */
#define STARPU_BCSR_GET_ROWPTR_DEV_HANDLE(interface) (((struct starpu_bcsr_interface *)(interface))->rowptr)
/**
   Return the base of the indexing (0 or 1 usually) in the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_FIRSTENTRY(interface) (((struct starpu_bcsr_interface *)(interface))->firstentry)
/**
   Return the height of blocks in the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_R(interface) (((struct starpu_bcsr_interface *)(interface))->r)
/**
   Return the width of blocks in the matrix designated
   by \p interface.
 */
#define STARPU_BCSR_GET_C(interface) (((struct starpu_bcsr_interface *)(interface))->c)
/**
   Return the size of elements in the matrix designated by \p interface.
 */
#define STARPU_BCSR_GET_ELEMSIZE(interface) (((struct starpu_bcsr_interface *)(interface))->elemsize)
/**
   Return the offset in the arrays (coling, rowptr, nzval) of the
   matrix designated by \p interface, to be used with the device handles.
 */
#define STARPU_BCSR_GET_OFFSET 0

/** @} */

/**
   @name Multiformat Data Interface
   @{
*/

/**
   Multiformat operations
 */
struct starpu_multiformat_data_interface_ops
{
	size_t cpu_elemsize;			 /**< size of each element on CPUs */
	size_t opencl_elemsize;			 /**< size of each element on OpenCL devices */
	struct starpu_codelet *cpu_to_opencl_cl; /**< pointer to a codelet which converts from CPU to OpenCL */
	struct starpu_codelet *opencl_to_cpu_cl; /**< pointer to a codelet which converts from OpenCL to CPU */
	size_t cuda_elemsize;			 /**< size of each element on CUDA devices */
	struct starpu_codelet *cpu_to_cuda_cl;	 /**< pointer to a codelet which converts from CPU to CUDA */
	struct starpu_codelet *cuda_to_cpu_cl;	 /**< pointer to a codelet which converts from CUDA to CPU */
};

struct starpu_multiformat_interface
{
	enum starpu_data_interface_id id;

	void *cpu_ptr;
	void *cuda_ptr;
	void *hip_ptr;
	void *opencl_ptr;
	uint32_t nx;
	struct starpu_multiformat_data_interface_ops *ops;
};

/**
   Register a piece of data that can be represented in different
   ways, depending upon the processing unit that manipulates it. It
   allows the programmer, for instance, to use an array of structures
   when working on a CPU, and a structure of arrays when working on a
   GPU. \p nobjects is the number of elements in the data. \p format_ops
   describes the format.
   See \ref TheMultiformatInterface for more details.
*/
void starpu_multiformat_data_register(starpu_data_handle_t *handle, int home_node, void *ptr, uint32_t nobjects, struct starpu_multiformat_data_interface_ops *format_ops);

/**
   Return the local pointer to the data with CPU format.
 */
#define STARPU_MULTIFORMAT_GET_CPU_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->cpu_ptr)
/**
   Return the local pointer to the data with CUDA format.
 */
#define STARPU_MULTIFORMAT_GET_CUDA_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->cuda_ptr)
/**
   Return the local pointer to the data with HIP format.
 */
#define STARPU_MULTIFORMAT_GET_HIP_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->hip_ptr)

/**
   Return the local pointer to the data with OpenCL format.
*/
#define STARPU_MULTIFORMAT_GET_OPENCL_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->opencl_ptr)
/**
   Return the number of elements in the data.
 */
#define STARPU_MULTIFORMAT_GET_NX(interface) (((struct starpu_multiformat_interface *)(interface))->nx)

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DATA_INTERFACES_H__ */
