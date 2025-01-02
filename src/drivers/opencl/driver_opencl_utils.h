/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_OPENCL_UTILS_H__
#define __STARPU_OPENCL_UTILS_H__

#pragma GCC visibility push(hidden)

/** @file */

char *_starpu_opencl_get_device_type_as_string(int id);

#define _STARPU_OPENCL_PLATFORM_MAX 4

#pragma GCC visibility pop

#endif /* __STARPU_OPENCL_UTILS_H__ */
