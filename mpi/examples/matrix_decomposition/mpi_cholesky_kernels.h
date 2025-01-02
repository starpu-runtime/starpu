/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __MPI_CHOLESKY_KERNELS_H__
#define __MPI_CHOLESKY_KERNELS_H__

#include <starpu.h>

void chol_cpu_codelet_update_potrf(void **, void *);
void chol_cpu_codelet_update_trsm(void **, void *);
void chol_cpu_codelet_update_syrk(void **, void *);
void chol_cpu_codelet_update_gemm(void **, void *);

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_potrf(void *descr[], void *_args);
void chol_cublas_codelet_update_trsm(void *descr[], void *_args);
void chol_cublas_codelet_update_syrk(void *descr[], void *_args);
void chol_cublas_codelet_update_gemm(void *descr[], void *_args);
#endif

#endif // __MPI_CHOLESKY_KERNELS_H__
