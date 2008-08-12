#include <task-models/task_model.h>
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

double core_chol_task_11_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/1000.0f*0.894);

#ifdef MODEL_DEBUG
	printf("core_chol_task_11_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

double cuda_chol_task_11_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/50.0f/10.75/5.088633);

#ifdef MODEL_DEBUG
	printf("cuda_chol_task_11_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

double core_chol_task_21_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/7706.674/0.95);

#ifdef MODEL_DEBUG
	printf("core_chol_task_21_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

double cuda_chol_task_21_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/50.0f/10.75/87.29520);

#ifdef MODEL_DEBUG
	printf("cuda_chol_task_21_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

double core_chol_task_22_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/50.0f/10.75/8.0760);

#ifdef MODEL_DEBUG
	printf("core_chol_task_22_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}

double cuda_chol_task_22_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/50.0f/10.75/76.30666);

#ifdef MODEL_DEBUG
	printf("cuda_chol_task_22_cost n %d cost %e\n", n, cost);
#endif

	return PERTURBATE(cost);
}
