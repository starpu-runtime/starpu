/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_OPENCL_UTILS_H__
#define __STARPU_OPENCL_UTILS_H__

#include <config.h>

#ifdef STARPU_VERBOSE
#  define _STARPU_OPENCL_DEBUG(fmt, args ...) fprintf(stderr, "[starpu][%s] " fmt ,__func__ ,##args)
#else
#  define _STARPU_OPENCL_DEBUG(fmt, args ...)
#endif

#define _STARPU_OPENCL_DISP(fmt, args ...) fprintf(stderr, "[starpu][%s] " fmt ,__func__ ,##args)

#define _STARPU_OPENCL_ERROR(fmt, args ...)                                                   \
	do {                                                                          \
                fprintf(stderr, "[starpu][%s] Error: " fmt ,__func__ ,##args); \
		assert(0);                                                            \
	} while (0)

#define STARPU_OPENCL_PLATFORM_MAX 4

#endif /* __STARPU_OPENCL_UTILS_H__ */
