/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DW_WRITE_BACK_H__
#define __DW_WRITE_BACK_H__

/** @file */

#include <starpu.h>
#include <datawizard/coherency.h>

/** If a write-through mask is associated to that data handle, this propagates
 * the the current value of the data onto the different memory nodes in the
 * write_through_mask. */
void _starpu_write_through_data(starpu_data_handle_t handle, unsigned requesting_node,
					   uint32_t write_through_mask);

#endif // __DW_WRITE_BACK_H__
