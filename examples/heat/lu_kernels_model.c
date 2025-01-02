/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
 * Copyright (C) 2011-2011  Télécom Sud Paris
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
#define PERTURB(a)	((starpu_drand48()*2.0f*(AMPL) + 1.0f - (AMPL))*(a))
#else
#define PERTURB(a)	(a)
#endif

/*
 *
 *	Generic models
 *
 */

double task_getrf_cost(struct starpu_task *task, unsigned nimpl)
{
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/537.5);

	return PERTURB(cost);
}

double task_trsm_ll_cost(struct starpu_task *task, unsigned nimpl)
{
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

/*	double cost = ((n*n*n)/1744.695); */
	double cost = ((n*n*n)/3210.80);

	/* fprintf(stderr, "task TRSM_LL predicts %e\n", cost); */
	return PERTURB(cost);
}


double task_trsm_ru_cost(struct starpu_task *task, unsigned nimpl)
{
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

/*	double cost = ((n*n*n)/1744.695); */
	double cost = ((n*n*n)/3691.53);

	/* fprintf(stderr, "task TRSM_RU predicts %e\n", cost); */
	return PERTURB(cost);
}



double task_gemm_cost(struct starpu_task *task, unsigned nimpl)
{
	(void)nimpl;
	uint32_t nx, ny, nz;

	nx = starpu_matrix_get_nx(task->handles[2]);
	ny = starpu_matrix_get_ny(task->handles[2]);
	nz = starpu_matrix_get_ny(task->handles[0]);

	double cost = ((nx*ny*nz)/4110.0);

	return PERTURB(cost);
}

/*
 *
 *	Models for CUDA
 *
 */


double task_getrf_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/1853.7806);

/*	printf("CUDA task GETRF ; predict %e\n", cost); */
	return PERTURB(cost);
}

double task_trsm_ll_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/42838.5718);

/*	printf("CUDA task TRSM_LL ; predict %e\n", cost); */
	return PERTURB(cost);
}


double task_trsm_ru_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/49208.667);

/*	printf("CUDA task TRSM_RU ; predict %e\n", cost); */
	return PERTURB(cost);
}



double task_gemm_cost_cuda(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t nx, ny, nz;

	nx = starpu_matrix_get_nx(task->handles[2]);
	ny = starpu_matrix_get_ny(task->handles[2]);
	nz = starpu_matrix_get_ny(task->handles[0]);

	double cost = ((nx*ny*nz)/57523.560);

/*	printf("CUDA task GEMM ; predict %e\n", cost); */
	return PERTURB(cost);
}

/*
 *
 *	Models for CPUs
 *
 */

double task_getrf_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/537.5);

/*	printf("CPU task GETRF ; predict %e\n", cost); */
	return PERTURB(cost);
}

double task_trsm_ll_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/6668.224);

/*	printf("CPU task TRSM_LL ; predict %e\n", cost); */
	return PERTURB(cost);
}


double task_trsm_ru_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = ((n*n*n)/6793.8423);

/*	printf("CPU task TRSM_RU ; predict %e\n", cost); */
	return PERTURB(cost);
}



double task_gemm_cost_cpu(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t nx, ny, nz;

	nx = starpu_matrix_get_nx(task->handles[2]);
	ny = starpu_matrix_get_ny(task->handles[2]);
	nz = starpu_matrix_get_ny(task->handles[0]);

	double cost = ((nx*ny*nz)/4203.0175);

/*	printf("CPU task GEMM ; predict %e\n", cost); */
	return PERTURB(cost);
}

void initialize_lu_kernels_model(struct starpu_perfmodel* model, char * symbol,
				 double (*cost_function)(struct starpu_task *, unsigned),
				 double (*cpu_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned),
				 double (*cuda_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned))
{
	(void)cost_function;
	model->symbol = symbol;
	model->type = STARPU_HISTORY_BASED;

	starpu_perfmodel_init(model);

	starpu_perfmodel_set_per_devices_cost_function(model, 0, cpu_cost_function, STARPU_CPU_WORKER, 0, 1, -1);

	if(starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) != 0)
	{
		starpu_perfmodel_set_per_devices_cost_function(model, 0, cuda_cost_function, STARPU_CUDA_WORKER, 0, 1, -1);
	}
}
