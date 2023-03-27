/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static cublasStatus_t st;

static void
bad_0(void)
{
	st = cublasGetError();
	if (STARPU_UNLIKELY(st != CUBLAS_STATUS_SUCCESS))
		STARPU_ABORT();
}

static void
bad_1(void)
{
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ABORT();
}

static void
bad_2(void)
{
	st = cublasGetError();
	STARPU_ASSERT(!st);
}
