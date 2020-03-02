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

#ifndef __SPMV_H__
#define __SPMV_H__

#include <sys/types.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <signal.h>

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_USE_CUDA
void spmv_kernel_cuda(void *descr[], void *args);
#endif

#ifdef STARPU_USE_OPENCL
void spmv_kernel_opencl(void *descr[], void *args);
void compile_spmv_opencl_kernel(void);
#endif

void spmv_kernel_cpu(void *descr[], void *arg);

#endif /* __SPMV_H__ */
