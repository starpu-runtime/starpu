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

#include "dw_cholesky_models.h"

/*
 * As a convention, in that file, descr[0] is represented by A,
 * 				  descr[1] is B ...
 */

/*
 *	Number of flops of Gemm 
 */

//#define USE_PERTURBATION	1


#ifdef USE_PERTURBATION
#define PERTURBATE(a)	((drand48()*2.0f*(AMPL) + 1.0f - (AMPL))*(a))
#else
#define PERTURBATE(a)	(a)
#endif

static double core_chol_task_11_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/1000.0f*0.894/0.79176);

#ifdef MODEL_DEBUG
	printf("core_chol_task_11_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

static double cuda_chol_task_11_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/50.0f/10.75/5.088633/0.9883);

#ifdef MODEL_DEBUG
	printf("cuda_chol_task_11_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

static double core_chol_task_21_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/7706.674/0.95/0.9965);

#ifdef MODEL_DEBUG
	printf("core_chol_task_21_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

static double cuda_chol_task_21_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/50.0f/10.75/87.29520);

#ifdef MODEL_DEBUG
	printf("cuda_chol_task_21_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

static double core_chol_task_22_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/50.0f/10.75/8.0760);

#ifdef MODEL_DEBUG
	printf("core_chol_task_22_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

static double cuda_chol_task_22_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/50.0f/10.75/76.30666);

#ifdef MODEL_DEBUG
	printf("cuda_chol_task_22_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

struct starpu_perfmodel_t chol_model_11 = {
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = core_chol_task_11_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = cuda_chol_task_11_cost }
	},
	.type = HISTORY_BASED,
	.symbol = "chol_model_11"
};

struct starpu_perfmodel_t chol_model_21 = {
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = core_chol_task_21_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = cuda_chol_task_21_cost }
	},
	.type = HISTORY_BASED,
	.symbol = "chol_model_21"
};

struct starpu_perfmodel_t chol_model_22 = {
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = core_chol_task_22_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = cuda_chol_task_22_cost }
	},
	.type = HISTORY_BASED,
	.symbol = "chol_model_22"
};
