/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <common/uthash.h>
#ifdef STARPU_OPENMP
#include <util/openmp_runtime_support.h>
#endif

#pragma GCC visibility push(hidden)

/** Generic type representing an interface, for now it's only used before
 * execution on message-passing devices but it can be useful in other cases.
 */
union _starpu_interface
{
	/* struct starpu_void_interface void; void doesn't have any data */
	struct starpu_variable_interface variable;
	struct starpu_vector_interface vector;
	struct starpu_matrix_interface matrix;
	struct starpu_block_interface block;
	struct starpu_tensor_interface tensor;
	struct starpu_csr_interface csr;
	struct starpu_bcsr_interface bcsr;
	struct starpu_coo_interface coo;
};

/** Some data interfaces or filters use this interface internally */
extern struct starpu_data_interface_ops starpu_interface_multiformat_ops;

void _starpu_data_free_interfaces(starpu_data_handle_t handle);

extern int _starpu_data_handle_init(starpu_data_handle_t handle, struct starpu_data_interface_ops *interface_ops, unsigned int mf_node);
void _starpu_data_initialize_per_worker(starpu_data_handle_t handle);

extern struct starpu_arbiter *_starpu_global_arbiter;
extern void _starpu_data_interface_init(void);
extern int __starpu_data_check_not_busy(starpu_data_handle_t handle) STARPU_ATTRIBUTE_WARN_UNUSED_RESULT;
#define _starpu_data_check_not_busy(handle) \
	(STARPU_UNLIKELY(!handle->busy_count && \
			 (handle->busy_waiting || handle->lazy_unregister)) ? \
		__starpu_data_check_not_busy(handle) : 0)
extern void _starpu_data_interface_shutdown(void);

struct starpu_data_interface_ops *_starpu_data_interface_get_ops(unsigned interface_id) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

#define _starpu_data_is_multiformat_handle(handle) handle->ops->is_multiformat

void _starpu_data_deinitialize_submit_noplan(starpu_data_handle_t handle);

void _starpu_data_invalidate_submit_noplan(starpu_data_handle_t handle);

#pragma GCC visibility pop

#endif // __DATA_INTERFACE_H__
