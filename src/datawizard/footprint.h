/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __FOOTPRINT_H__
#define __FOOTPRINT_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <core/jobs.h>

/** Compute the footprint that characterizes the job and cache it into the job
 * structure. */
uint32_t _starpu_compute_buffers_footprint(struct starpu_perfmodel *model, struct starpu_perfmodel_arch * arch, unsigned nimpl, struct _starpu_job *j);

/** Compute the footprint that characterizes the layout of the data handle. */
uint32_t _starpu_compute_data_footprint(starpu_data_handle_t handle);

/** Compute the footprint that characterizes the allocation of the data handle. */
uint32_t _starpu_compute_data_alloc_footprint(starpu_data_handle_t handle);

#endif // __FOOTPRINT_H__
