/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011,2013,2015,2017                 CNRS
 * Copyright (C) 2009-2011,2014                           Université de Bordeaux
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

#include "mpi_cholesky.h"

/*
 *	Number of flops of Gemm
 */

struct starpu_perfmodel chol_model_11 =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "chol_model_11"
};

struct starpu_perfmodel chol_model_21 =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "chol_model_21"
};

struct starpu_perfmodel chol_model_22 =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "chol_model_22"
};
