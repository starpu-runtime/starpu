#include <core/perfmodel/perfmodel.h>
#include <task-models/task_model.h>
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

double task_11_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/537.5);

	return PERTURBATE(cost);
}

double task_12_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

//	double cost = ((n*n*n)/1744.695);
	double cost = ((n*n*n)/3210.80);

	//fprintf(stderr, "task 12 predicts %e\n", cost);
	return PERTURBATE(cost);
}


double task_21_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

//	double cost = ((n*n*n)/1744.695);
	double cost = ((n*n*n)/3691.53);

	//fprintf(stderr, "task 12 predicts %e\n", cost);
	return PERTURBATE(cost);
}



double task_22_cost(buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = get_blas_nx(descr[2].state);
	ny = get_blas_ny(descr[2].state);
	nz = get_blas_ny(descr[0].state);

	double cost = ((nx*ny*nz)/4110.0);

	return PERTURBATE(cost);
}

/* 
 *
 *	Models for CUDA
 *
 */


double task_11_cost_cuda(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/1853.7806);

//	printf("CUDA task 11 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

double task_12_cost_cuda(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/42838.5718);

//	printf("CUDA task 12 ; predict %e\n", cost);
	return PERTURBATE(cost);
}


double task_21_cost_cuda(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/49208.667);

//	printf("CUDA task 21 ; predict %e\n", cost);
	return PERTURBATE(cost);
}



double task_22_cost_cuda(buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = get_blas_nx(descr[2].state);
	ny = get_blas_ny(descr[2].state);
	nz = get_blas_ny(descr[0].state);

	double cost = ((nx*ny*nz)/57523.560);

//	printf("CUDA task 22 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

/* 
 *
 *	Models for CPUs
 *
 */

double task_11_cost_core(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/537.5);

//	printf("CORE task 11 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

double task_12_cost_core(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/6668.224);

//	printf("CORE task 12 ; predict %e\n", cost);
	return PERTURBATE(cost);
}


double task_21_cost_core(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = ((n*n*n)/6793.8423);

//	printf("CORE task 21 ; predict %e\n", cost);
	return PERTURBATE(cost);
}



double task_22_cost_core(buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = get_blas_nx(descr[2].state);
	ny = get_blas_ny(descr[2].state);
	nz = get_blas_ny(descr[0].state);

	double cost = ((nx*ny*nz)/4203.0175);

//	printf("CORE task 22 ; predict %e\n", cost);
	return PERTURBATE(cost);
}

struct perfmodel_t model_11 = {
	.cost_model = task_11_cost,
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = task_11_cost_core },
		[CUDA_DEFAULT] = { .cost_model = task_11_cost_cuda }
	},
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_11_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_11_goto"
#else
	.symbol = "lu_model_11"
#endif
};

struct perfmodel_t model_12 = {
	.cost_model = task_12_cost,
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = task_12_cost_core },
		[CUDA_DEFAULT] = { .cost_model = task_12_cost_cuda }
	},
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_12_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_12_goto"
#else
	.symbol = "lu_model_12"
#endif
};

struct perfmodel_t model_21 = {
	.cost_model = task_21_cost,
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = task_21_cost_core },
		[CUDA_DEFAULT] = { .cost_model = task_21_cost_cuda }
	},
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_21_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_21_goto"
#else
	.symbol = "lu_model_21"
#endif
};

struct perfmodel_t model_22 = {
	.cost_model = task_22_cost,
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = task_22_cost_core },
		[CUDA_DEFAULT] = { .cost_model = task_22_cost_cuda }
	},
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = "lu_model_22_atlas"
#elif defined(GOTO)
	.symbol = "lu_model_22_goto"
#else
	.symbol = "lu_model_22"
#endif
};
