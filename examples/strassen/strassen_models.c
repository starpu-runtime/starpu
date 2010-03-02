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

#include "strassen_models.h"

#include <starpu.h>

/*
 * As a convention, in that file, descr[0] is represented by A,
 * 				  descr[1] is B ...
 */

/*
 *	Number of flops of Gemm 
 */

//#define USE_PERTURBATION	1


#ifdef USE_PERTURBATION
#define PERTURBATE(a)	((starpu_drand48()*2.0f*(AMPL) + 1.0f - (AMPL))*(a))
#else
#define PERTURBATE(a)	(a)
#endif


static double self_add_sub_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_matrix_nx(descr[0].handle);

	double cost = (n*n)/10.0f/4.0f/7.75f;

#ifdef STARPU_MODEL_DEBUG
	printf("self add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double cuda_self_add_sub_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_matrix_nx(descr[0].handle);

	double cost = (n*n)/10.0f/4.0f;

#ifdef STARPU_MODEL_DEBUG
	printf("self add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double add_sub_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_matrix_nx(descr[0].handle);

	double cost = (1.45f*n*n)/10.0f/2.0f;

#ifdef STARPU_MODEL_DEBUG
	printf("add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double cuda_add_sub_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_matrix_nx(descr[0].handle);

	double cost = (1.45f*n*n)/10.0f/2.0f;

#ifdef STARPU_MODEL_DEBUG
	printf("add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}


static double mult_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_matrix_nx(descr[0].handle);

	double cost = (((double)(n)*n*n)/1000.0f/4.11f/0.2588);

#ifdef STARPU_MODEL_DEBUG
	printf("mult cost %e n = %d \n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double cuda_mult_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_matrix_nx(descr[0].handle);

	double cost = (((double)(n)*n*n)/1000.0f/4.11f);

#ifdef STARPU_MODEL_DEBUG
	printf("mult cost %e n = %d \n", cost, n);
#endif

	return PERTURBATE(cost);
}

struct starpu_perfmodel_t strassen_model_mult = {
	.per_arch = { 
		[STARPU_CPU_DEFAULT] = { .cost_model = mult_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = cuda_mult_cost }
	},
	.type = STARPU_HISTORY_BASED,
	.symbol = "strassen_model_mult"
};

struct starpu_perfmodel_t strassen_model_add_sub = {
	.per_arch = { 
		[STARPU_CPU_DEFAULT] = { .cost_model = add_sub_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = cuda_add_sub_cost }
	},
	.type = STARPU_HISTORY_BASED,
	.symbol = "strassen_model_add_sub"
};

struct starpu_perfmodel_t strassen_model_self_add_sub = {
	.per_arch = { 
		[STARPU_CPU_DEFAULT] = { .cost_model = self_add_sub_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = cuda_self_add_sub_cost }
	},
	.type = STARPU_HISTORY_BASED,
	.symbol = "strassen_model_self_add_sub"
};
