/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __FILTERS_H__
#define __FILTERS_H__

/** @file */

#include <stdarg.h>
#include <datawizard/coherency.h>
#include <datawizard/memalloc.h>

#include <starpu.h>
#include <common/config.h>

/** submit asynchronous unpartitioning / partitioning to make target active read-only or read-write */
void _starpu_data_partition_access_submit(starpu_data_handle_t target, int write);
#endif
