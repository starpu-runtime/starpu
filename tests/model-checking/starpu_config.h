/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#define STARPU_SIMGRID
#define STARPU_MAXIMPLEMENTATIONS 4
#define STARPU_NMAXBUFS 8
#define STARPU_MAXNODES 2
#define STARPU_NMAXWORKERS 16

#ifndef _MSC_VER
#include <stdint.h>
#else
#include <windows.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef UINT_PTR uintptr_t;
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef INT_PTR intptr_t;
#endif

#ifdef _MSC_VER
typedef long starpu_ssize_t;
#define __starpu_func__ __FUNCTION__
#else
#  include <sys/types.h>
typedef ssize_t starpu_ssize_t;
#define __starpu_func__ __func__
#endif

#if defined(c_plusplus) || defined(__cplusplus)
/* inline is part of C++ */
#  define __starpu_inline inline
#elif defined(_MSC_VER) || defined(__HP_cc)
#  define __starpu_inline __inline
#else
#  define __starpu_inline __inline__
#endif

