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

#include "lu_kernels_model.h"

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

/* 
 *
 *	Generic models
 *
 */

double task_11_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/537.5);

	return PERTURBATE(cost);
}

double task_12_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

//	double cost = ((n*n*n)/1744.695);
	double cost = ((n*n*n)/3210.80);

	//fprintf(stderr, "task 12 predicts %e\n", cost);
	return PERTURBATE(cost);
}


double task_21_cost(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

//	double cost = ((n*n*n)/1744.695);
	double cost = ((n*n*n)/3691.53);

	//fprintf(stderr, "task 12 predicts %e\n", cost);
	return PERTURBATE(cost);
}



double task_22_cost(starpu_buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = starpu_get_blas_nx(descr[2].handle);
	ny = starpu_get_blas_ny(descr[2].handle);
	nz = starpu_get_blas_ny(descr[0].handle);

	double cost = ((nx*ny*nz)/4110.0);

	return PERTURBATE(cost);
}

/* 
 *
 *	Models for CUDA
 *
 */


double task_11_cost_cuda(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/1853.7806);

//	printf("CUDA task 11 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

double task_12_cost_cuda(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/42838.5718);

//	printf("CUDA task 12 ; predict %e\n", cost);
	return PERTURBATE(cost);
}


double task_21_cost_cuda(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/49208.667);

//	printf("CUDA task 21 ; predict %e\n", cost);
	return PERTURBATE(cost);
}



double task_22_cost_cuda(starpu_buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = starpu_get_blas_nx(descr[2].handle);
	ny = starpu_get_blas_ny(descr[2].handle);
	nz = starpu_get_blas_ny(descr[0].handle);

	double cost = ((nx*ny*nz)/57523.560);

//	printf("CUDA task 22 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

/* 
 *
 *	Models for CPUs
 *
 */

double task_11_cost_core(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/537.5);

//	printf("CORE task 11 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

double task_12_cost_core(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/6668.224);

//	printf("CORE task 12 ; predict %e\n", cost);
	return PERTURBATE(cost);
}


double task_21_cost_core(starpu_buffer_descr *descr)
{
	uint32_t n;

	n = starpu_get_blas_nx(descr[0].handle);

	double cost = ((n*n*n)/6793.8423);

//	printf("CORE task 21 ; predict %e\n", cost);
	return PERTURBATE(cost);
}



double task_22_cost_core(starpu_buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = starpu_get_blas_nx(descr[2].handle);
	ny = starpu_get_blas_ny(descr[2].handle);
	nz = starpu_get_blas_ny(descr[0].handle);

	double cost = ((nx*ny*nz)/4203.0175);

//	printf("CORE task 22 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

struct starpu_perfmodel_t model_11 = {
	.cost_model = task_11_cost,
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = task_11_cost_core },
		[STARPU_CUDA_DEFAULT] = { .cost_model = task_11_cost_cuda }
	},
	.type = STARPU_HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_11_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_11_goto"
#else
	.symbol = "lu_model_11"
#endif
};

struct starpu_perfmodel_t model_12 = {
	.cost_model = task_12_cost,
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = task_12_cost_core },
		[STARPU_CUDA_DEFAULT] = { .cost_model = task_12_cost_cuda }
	},
	.type = STARPU_HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_12_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_12_goto"
#else
	.symbol = "lu_model_12"
#endif
};

struct starpu_perfmodel_t model_21 = {
	.cost_model = task_21_cost,
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = task_21_cost_core },
		[STARPU_CUDA_DEFAULT] = { .cost_model = task_21_cost_cuda }
	},
	.type = STARPU_HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_21_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_21_goto"
#else
	.symbol = "lu_model_21"
#endif
};

struct starpu_perfmodel_t model_22 = {
	.cost_model = task_22_cost,
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = task_22_cost_core },
		[STARPU_CUDA_DEFAULT] = { .cost_model = task_22_cost_cuda }
	},
	.type = STARPU_HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_22_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_22_goto"
#else
	.symbol = "lu_model_22"
#endif
};
