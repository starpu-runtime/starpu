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

#define TYPE	float
#define EPSILON	0.000001

#define CUBLAS_GEMM cublasSgemm
#define HIPBLAS_GEMM hipblasSgemm
#define CPU_GEMM	STARPU_SGEMM
#define CPU_ASUM	STARPU_SASUM
#define CPU_IAMAX	STARPU_ISAMAX
#define STARPU_GEMM(name)	starpu_sgemm_##name

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_GEMM_STR(name)  xstr(STARPU_GEMM(name))
