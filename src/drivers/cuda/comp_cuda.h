/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#ifndef __COMP_CUDA_H__
#define __COMP_CUDA_H__

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <cuda.h>

#define UPDIV(a,b)	(((a)+(b)-1)/((b)))

__device__ void cuda_dummy_mult(CUdeviceptr, CUdeviceptr, CUdeviceptr);

#endif // __COMP_CUDA_H__
