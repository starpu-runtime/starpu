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

#ifndef __BLAS_MODEL_H__
#define __BLAS_MODEL_H__

#include <starpu.h>

double gemm_cost(starpu_buffer_descr *descr);

static struct starpu_perfmodel_t sgemm_model = {
	.cost_model = gemm_cost,
	.type = STARPU_HISTORY_BASED,
#ifdef ATLAS
	.symbol = "sgemm_atlas"
#elif defined(GOTO)
	.symbol = "sgemm_goto"
#else
	.symbol = "sgemm"
#endif
};

static struct starpu_perfmodel_t sgemm_model_common = {
	.cost_model = gemm_cost,
	.type = STARPU_COMMON,
};

#endif // __BLAS_MODEL_H__
