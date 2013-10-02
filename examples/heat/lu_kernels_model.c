/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#include "lu_kernels_model.h"

/*
 * As a convention, in that file, buffers[0] is represented by A,
 * 				  buffers[1] is B ...
 */

/*
 *	Number of flops of Gemm
 */

/* #define USE_PERTURBATION	1 */


#ifdef USE_PERTURBATION
#define PERTURBATE(a)	((starpu_drand48()*2.0f*(AMPL) + 1.0f - (AMPL))*(a))
#else
#define PERTURBATE(a)	(a)
#endif

/*
 *
 *	Generic models
 *
 */

double task_11_cost(struct starpu_task *task, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/537.5);

	return PERTURBATE(cost);
}

double task_12_cost(struct starpu_task *task, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

/*	double cost = ((n*n*n)/1744.695); */
	double cost = ((n*n*n)/3210.80);

	/* fprintf(stderr, "task 12 predicts %e\n", cost); */
	return PERTURBATE(cost);
}


double task_21_cost(struct starpu_task *task, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

/*	double cost = ((n*n*n)/1744.695); */
	double cost = ((n*n*n)/3691.53);

	/* fprintf(stderr, "task 12 predicts %e\n", cost); */
	return PERTURBATE(cost);
}



double task_22_cost(struct starpu_task *task, unsigned nimpl)
{
	uint32_t nx, ny, nz;

	nx = starpu_matrix_get_nx(task->handles[2]);
	ny = starpu_matrix_get_ny(task->handles[2]);
	nz = starpu_matrix_get_ny(task->handles[0]);

	double cost = ((nx*ny*nz)/4110.0);

	return PERTURBATE(cost);
}

/*
 *
 *	Models for CUDA
 *
 */


double task_11_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/1853.7806);

/*	printf("CUDA task 11 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}

double task_12_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/42838.5718);

/*	printf("CUDA task 12 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}


double task_21_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/49208.667);

/*	printf("CUDA task 21 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}



double task_22_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t nx, ny, nz;

	nx = starpu_matrix_get_nx(task->handles[2]);
	ny = starpu_matrix_get_ny(task->handles[2]);
	nz = starpu_matrix_get_ny(task->handles[0]);

	double cost = ((nx*ny*nz)/57523.560);

/*	printf("CUDA task 22 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}

/*
 *
 *	Models for CPUs
 *
 */

double task_11_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/537.5);

/*	printf("CPU task 11 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}

double task_12_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/6668.224);

/*	printf("CPU task 12 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}


double task_21_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/6793.8423);

/*	printf("CPU task 21 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}



double task_22_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	uint32_t nx, ny, nz;

	nx = starpu_matrix_get_nx(task->handles[2]);
	ny = starpu_matrix_get_ny(task->handles[2]);
	nz = starpu_matrix_get_ny(task->handles[0]);

	double cost = ((nx*ny*nz)/4203.0175);

/*	printf("CPU task 22 ; predict %e\n", cost); */
	return PERTURBATE(cost);
}

void initialize_lu_kernels_model(struct starpu_perfmodel* model, char * symbol,
		double (*cost_function)(struct starpu_task *, unsigned),
		double (*cpu_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned),
		double (*cuda_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned))
{
	starpu_perfmodel_init(model);
	model->type = STARPU_HISTORY_BASED;
	model->symbol = symbol;
	model->cost_function = cost_function;
	model->per_arch[STARPU_CPU_WORKER][0][0][0].cost_function = cpu_cost_function;
	if(starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) != 0)
		model->per_arch[STARPU_CUDA_WORKER][0][0][0].cost_function = cuda_cost_function;
}
