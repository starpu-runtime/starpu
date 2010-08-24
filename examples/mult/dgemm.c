/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#define TYPE	double

#define CUBLAS_GEMM cublasDgemm
#define CPU_GEMM	DGEMM
#define CPU_ASUM	DASUM
#define CPU_IAMAX	IDAMAX
#define STARPU_GEMM(name)	starpu_dgemm_##name

#include "xgemm_kernels.c"
#include "xgemm.c" 
