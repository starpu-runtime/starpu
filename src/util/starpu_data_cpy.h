/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_DATA_CPY_H__
#define __STARPU_DATA_CPY_H__

/** @file */

#include <starpu.h>

#pragma GCC visibility push(hidden)

int _starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle,
		     int asynchronous, void (*callback_func)(void*), void *callback_arg,
		     int reduction, struct starpu_task *reduction_dep_task, int priority);

/** Achieve the Copy-on-Write on \p handle on any readonly copy, because we are writing to \p handle */
void _starpu_data_dup_ro_cow(starpu_data_handle_t handle, int priority);

#pragma GCC visibility pop

#endif // __STARPU_DATA_CPY_H__

