/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_EXAMPLE_CG_H__
#define __STARPU_EXAMPLE_CG_H__

#include <starpu.h>
#include <math.h>
#include <common/blas.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

#define DOUBLE

#ifdef DOUBLE
#define TYPE	double
#define GEMV	STARPU_DGEMV
#define DOT	STARPU_DDOT
#define AXPY	STARPU_DAXPY
#define SCAL	STARPU_DSCAL
#define cublasdot	cublasDdot
#define cublasscal	cublasDscal
#define cublasaxpy	cublasDaxpy
#define cublasgemv	cublasDgemv
#define cublasscal	cublasDscal
#else
#define TYPE	float
#define GEMV	STARPU_SGEMV
#define DOT	STARPU_SDOT
#define AXPY	STARPU_SAXPY
#define SCAL	STARPU_SSCAL
#define cublasdot	cublasSdot
#define cublasscal	cublasSscal
#define cublasaxpy	cublasSaxpy
#define cublasgemv	cublasSgemv
#define cublasscal	cublasSscal
#endif

#endif /* __STARPU_EXAMPLE_CG_H__ */
