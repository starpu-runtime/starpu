/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __NODE_OPS_H__
#define __NODE_OPS_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <datawizard/copy_driver.h>

#pragma GCC visibility push(hidden)

/** Request copying some data interface for handle \p handle: from interface \p
 * src_interface that exists on node \p src_node to interface \p dst_interface
 * that exists on node \p dst_node.
 *
 * If \p req is non-NULL, this can be used to start an asynchronous copy, in
 * which case -EAGAIN should be returned. Otherwise, 0 should be returned.
 *
 * _starpu_copy_interface_any_to_any can be used as a generic version, that
 * assumes that the data_interface implements the any_to_any method, and
 * copy_data_t will be used to queue the actual transfers.
 */
typedef int (*copy_interface_func_t)(starpu_data_handle_t handle, void *src_interface, unsigned src_node,
				void *dst_interface, unsigned dst_node,
				struct _starpu_data_request *req);

/** Request copying \p ssize bytes of data from \p src_ptr (plus offset \p src_offset)
 * in node \p src_node to \p dst_ptr (plus offset \p dst_offset) in node \p dst_node.
 *
 * If \p async_channel is non-NULL, this can be used to start an asynchronous copy, in
 * which case -EAGAIN should be returned. Otherwise, 0 should be returned.
 */
typedef int (*copy_data_t)(uintptr_t src_ptr, size_t src_offset, unsigned src_node,
				uintptr_t dst_ptr, size_t dst_offset, unsigned dst_node,
				size_t ssize, struct _starpu_async_channel *async_channel);

/** This is like copy_data_t, except that there are \p numblocks blocks of size
 * \p blocksize bytes to be transferred. On the source, their respective starts are \p
 * ld_src bytes apart, and on the destination their respective starts have to be
 * \p ld_dst bytes apart. (leading dimension) */
typedef int (*copy2d_data_t)(uintptr_t src_ptr, size_t src_offset, unsigned src_node,
				uintptr_t dst_ptr, size_t dst_offset, unsigned dst_node,
				size_t blocksize,
				size_t numblocks, size_t ld_src, size_t ld_dst,
				struct _starpu_async_channel *async_channel);

/** This is like copy_data_t, except that there are \p numblocks_2 metablocks to
 * be transferred. On the source, their respective starts are \p ld2_src bytes
 * apart, and on the destination their respective starts have to be \p ld2_dst
 * bytes apart.
 *
 * The metablocks are composed of \p numblocks_1 blocks of size \p blocksize
 * bytes. On the source, their respective starts are \p ld1_src bytes apart, and
 * on the destination their respective starts have to be \p ld1_dst bytes apart.
 */
typedef int (*copy3d_data_t)(uintptr_t src_ptr, size_t src_offset, unsigned src_node,
				uintptr_t dst_ptr, size_t dst_offset, unsigned dst_node,
				size_t blocksize,
				size_t numblocks_1, size_t ld1_src, size_t ld1_dst,
				size_t numblocks_2, size_t ld2_src, size_t ld2_dst,
				struct _starpu_async_channel *async_channel);

/** Map \p size bytes of data from \p src (plus offset \p src_offset) in node \p src_node
 * on node \p dst_node. If successful, return the resulting pointer, otherwise fill *ret */
typedef uintptr_t (*map_t)(uintptr_t src, size_t src_offset, unsigned src_node, unsigned dst_node, size_t size, int *ret);
/** Unmap \p size bytes of data from \p src (plus offset \p src_offset) in node \p src_node
 * on node \p dst_node. */
typedef int (*unmap_t)(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, unsigned dst_node, size_t size);
/** Update cache coherency for the mapping of \p size bytes of data from \p src (plus offset
 * \p src_offset) in node \p src_node on node \p dst_node. */
typedef int (*update_map_t)(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size);

/** Reference all the methods for copying data from this kind of device to
 * another kind of device. */
struct _starpu_node_ops
{
	/** Request copying a data interface from this type of node to another type of node.
	 * As a first start, you can just use the generic _starpu_copy_interface_any_to_any.  */
	copy_interface_func_t copy_interface_to[STARPU_MAX_RAM+1];

	/** Request copying a data interface to this type of node from another type of node.
	 * As a first start, you can just use the generic _starpu_copy_interface_any_to_any.  */
	copy_interface_func_t copy_interface_from[STARPU_MAX_RAM+1];

	/** Request copying a piece of data from this type of node to another type of node.
	 * This method is required at least for STARPU_CPU_RAM.  */
	copy_data_t copy_data_to[STARPU_MAX_RAM+1];

	/** Request copying a piece of data to this type of node from another type of node.
	 * This method is required at least for STARPU_CPU_RAM.  */
	copy_data_t copy_data_from[STARPU_MAX_RAM+1];

	/** Request copying a 2D piece of data (i.e. matrix tile with an ld)
	 * from this type of node to another type of node.
	 * This method is optional.  */
	copy2d_data_t copy2d_data_to[STARPU_MAX_RAM+1];

	/** Request copying a 2D piece of data (i.e. matrix tile with an ld)
	 * to this type of node from another type of node.
	 * This method is optional.  */
	copy2d_data_t copy2d_data_from[STARPU_MAX_RAM+1];

	/** Request copying a 3D piece of data (i.e. block piece with ldy and ldz)
	 * from this type of node to another type of node.
	 * This method is optional.  */
	copy3d_data_t copy3d_data_to[STARPU_MAX_RAM+1];

	/** Request copying a 3D piece of data (i.e. block piece with ldy and ldz)
	 * to this type of node from another type of node.
	 * This method is optional.  */
	copy3d_data_t copy3d_data_from[STARPU_MAX_RAM+1];

	/** Wait for the completion of asynchronous request \p async_channel.  */
	void (*wait_request_completion)(struct _starpu_async_channel *async_channel);
	/** Test whether asynchronous request \p async_channel has completed.  */
	unsigned (*test_request_completion)(struct _starpu_async_channel *async_channel);

	/** Return whether inter-device transfers are possible between \p node and \p handling_node.
	 * If this returns 0, copy_interface_to will always be called with
	 * CPU RAM as either source or destination. If this returns 1,
	 * copy_interface_to may be called with both source and destination in
	 * device memory.
	 *
	 * \p handling_node is the node that will initiate the transfer. This
	 * allows to prefer starting from the driver itself.
	 */
	int (*is_direct_access_supported)(unsigned node, unsigned handling_node);

	/** Allocate \p size bytes of data on node \p dst_node.
	 * \p flags can contain STARPU_MALLOC_* flags, only useful for CPU memory  */
	uintptr_t (*malloc_on_node)(unsigned dst_node, size_t size, int flags);
	/** Free data \p addr, which was a previous allocation of \p size bytes
	 * of data on node \p dst_node with flags \p flags*/
	void (*free_on_node)(unsigned dst_node, uintptr_t addr, size_t size, int flags);

	/** Map data a piece of data to this type of node from another type of node.
	 * This method is optional */
	map_t map[STARPU_MAX_RAM+1];

	/** Unmap data a piece of data to this type of node from another type of node.
	 * This method is optional */
	unmap_t unmap[STARPU_MAX_RAM+1];

	/** Update cache coherency for the mapping of a piece of data to this type of
	 * node from another type of node.
	 * This method is optional */
	update_map_t update_map[STARPU_MAX_RAM+1];

	/** Name of the type of memory, for debugging */
	char *name;
};

const char* _starpu_node_get_prefix(enum starpu_node_kind kind);

#pragma GCC visibility pop

#endif // __NODE_OPS_H__
