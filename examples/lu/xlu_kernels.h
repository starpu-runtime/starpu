/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __XLU_KERNELS_H__
#define __XLU_KERNELS_H__

#include <starpu.h>

void STARPU_LU(cpu_pivot)(void *descr[], void *_args);
void STARPU_LU(cpu_getrf_pivot)(void *descr[], void *_args);
void STARPU_LU(cpu_getrf)(void *descr[], void *_args);
void STARPU_LU(cpu_trsmll)(void *descr[], void *_args);
void STARPU_LU(cpu_trsmru)(void *descr[], void *_args);
void STARPU_LU(cpu_gemm)(void *descr[], void *_args);

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_pivot)(void *descr[], void *_args);
void STARPU_LU(cublas_getrf_pivot)(void *descr[], void *_args);
void STARPU_LU(cublas_getrf)(void *descr[], void *_args);
void STARPU_LU(cublas_trsmll)(void *descr[], void *_args);
void STARPU_LU(cublas_trsmru)(void *descr[], void *_args);
void STARPU_LU(cublas_gemm)(void *descr[], void *_args);
#endif

extern struct starpu_codelet cl_getrf;
extern struct starpu_codelet cl_getrf_pivot;
extern struct starpu_codelet cl_trsm_ll;
extern struct starpu_codelet cl_trsm_ru;
extern struct starpu_codelet cl_gemm;
extern struct starpu_codelet cl_pivot;

#endif /* __XLU_KERNELS_H__ */
