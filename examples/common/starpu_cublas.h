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

#ifndef __STARPU_CUBLAS_H__
#define __STARPU_CUBLAS_H__

#include <starpu.h>

#ifdef USE_CUDA
#include <cublas.h>
#endif

void init_cublas_on_all_cuda_devices(void);
void shutdown_cublas_on_all_cuda_devices(void);

#endif // __STARPU_CUBLAS_H__
