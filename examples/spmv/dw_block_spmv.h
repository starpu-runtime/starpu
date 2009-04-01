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

#ifndef __DW_BLOCK_SPMV_H__
#define __DW_BLOCK_SPMV_H__

#include <semaphore.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cublas.h>
#endif

void core_block_spmv(starpu_data_interface_t *descr, void *_args);

#ifdef USE_CUDA
void cublas_block_spmv(starpu_data_interface_t *descr, void *_args);
#endif // USE_CUDA

#endif // __DW_BLOCK_SPMV_H__
