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

/* Real double macros */

#define TYPE double
#define CUBLAS_TYPE TYPE

#define STARPU_LU(name)       starpu_dlu_##name

#define CUBLAS_GEMM	cublasDgemm
#define CUBLAS_TRSM	cublasDtrsm
#define CUBLAS_SCAL	cublasDscal
#define CUBLAS_GER	cublasDger
#define CUBLAS_SWAP	cublasDswap
#define CUBLAS_IAMAX	cublasIdamax

#define CPU_GEMM	STARPU_DGEMM
#define CPU_TRSM	STARPU_DTRSM
#define CPU_SCAL	STARPU_DSCAL
#define CPU_GER		STARPU_DGER
#define CPU_SWAP	STARPU_DSWAP

#define CPU_TRMM	STARPU_DTRMM
#define CPU_AXPY	STARPU_DAXPY
#define CPU_ASUM	STARPU_DASUM
#define CPU_IAMAX	STARPU_IDAMAX

#define PIVOT_THRESHHOLD	10e-10

#define CAN_EXECUTE .can_execute = can_execute,

#define ISZERO(f)	(fpclassify(f) == FP_ZERO)
