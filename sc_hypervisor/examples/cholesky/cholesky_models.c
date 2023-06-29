/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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

/*
 * Example of a cost model for BLAS operations.  This is really just an
 * example!
 */

/*
 * As a convention, in that file, buffers[0] is represented by A,
 * 				  buffers[1] is B ...
 */

/*
 *	Number of flops of Gemm
 */

#include <starpu.h>
#include <starpu_perfmodel.h>
#include "cholesky.h"

/* #define USE_PERTURBATION	1 */

#ifdef USE_PERTURBATION
#define PERTURB(a)	((starpu_drand48()*2.0f*(AMPL) + 1.0f - (AMPL))*(a))
#else
#define PERTURB(a)	(a)
#endif

double cpu_chol_task_potrf_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/1000.0f*0.894/0.79176);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cpu_chol_task_potrf_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cuda_chol_task_potrf_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/5.088633/0.9883);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cuda_chol_task_potrf_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cpu_chol_task_trsm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/7706.674/0.95/0.9965);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cpu_chol_task_trsm_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cuda_chol_task_trsm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/87.29520);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cuda_chol_task_trsm_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cpu_chol_task_syrk_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/8.0760)/2;

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cpu_chol_task_syrk_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cuda_chol_task_syrk_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/76.30666)/2;

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cuda_chol_task_syrk_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cpu_chol_task_gemm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/8.0760);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cpu_chol_task_gemm_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

double cuda_chol_task_gemm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	(void)arch;
	(void)nimpl;
	uint32_t n;

	n = starpu_matrix_get_nx(task->handles[0]);

	double cost = (((double)(n)*n*n)/50.0f/10.75/76.30666);

#ifdef STARPU_MODEL_DEBUG
	FPRINTF(stdout, "cuda_chol_task_gemm_cost n %u cost %e\n", n, cost);
#endif

	return PERTURB(cost);
}

void initialize_chol_model(struct starpu_perfmodel* model, char * symbol,
			   double (*cpu_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned),
			   double (*cuda_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned))
{
	struct starpu_perfmodel_per_arch *per_arch;

	model->symbol = symbol;
	model->type = STARPU_HISTORY_BASED;

	starpu_perfmodel_init(model);

	per_arch = starpu_perfmodel_get_model_per_devices(model, 0, STARPU_CPU_WORKER, 0, 1, -1);
	per_arch->cost_function = cpu_cost_function;
	// We could also call directly:
	// starpu_perfmodel_set_per_devices_cost_function(model, 0, cpu_cost_function, STARPU_CPU_WORKER, 0, 1, -1);

	if(starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) != 0)
	{
		per_arch = starpu_perfmodel_get_model_per_devices(model, 0, STARPU_CUDA_WORKER, 0, 1, -1);
		per_arch->cost_function = cuda_cost_function;

	}
}

unsigned g_size = 4*1024;
unsigned g_nblocks = 16;
unsigned g_nbigblocks = 8;
unsigned g_pinned = 0;
unsigned g_noprio = 0;
unsigned g_check = 0;
unsigned g_bound = 0;
unsigned g_with_ctxs = 0;
unsigned g_with_noctxs = 0;
unsigned g_chole1 = 0;
unsigned g_chole2 = 0;

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-with_ctxs") == 0)
		{
			g_with_ctxs = 1;
			break;
		}
		if (strcmp(argv[i], "-with_noctxs") == 0)
		{
			g_with_noctxs = 1;
			break;
		}

		if (strcmp(argv[i], "-chole1") == 0)
		{
			g_chole1 = 1;
			break;
		}

		if (strcmp(argv[i], "-chole2") == 0)
		{
			g_chole2 = 1;
			break;
		}

		if (strcmp(argv[i], "-size") == 0)
		{
		        char *argptr;
			g_size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0)
		{
		        char *argptr;
			g_nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nbigblocks") == 0)
		{
		        char *argptr;
			g_nbigblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-pin") == 0)
		{
			g_pinned = 1;
		}

		if (strcmp(argv[i], "-no-prio") == 0)
		{
			g_noprio = 1;
		}

		if (strcmp(argv[i], "-bound") == 0)
		{
			g_bound = 1;
		}

		if (strcmp(argv[i], "-check") == 0)
		{
			g_check = 1;
		}

		if (strcmp(argv[i], "-h") == 0)
		{
			printf("usage : %s [-pin] [-size size] [-nblocks nblocks] [-check]\n", argv[0]);
		}
	}
}
