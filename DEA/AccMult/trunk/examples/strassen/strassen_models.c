#include <task-models/task_model.h>
#include "strassen_models.h"

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





double self_add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (n*n)/10.0f/4.0f/7.75f;

#ifdef MODEL_DEBUG
	printf("self add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

double cuda_self_add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (n*n)/10.0f/4.0f;

#ifdef MODEL_DEBUG
	printf("self add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

double add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (1.45f*n*n)/10.0f/2.0f;

#ifdef MODEL_DEBUG
	printf("add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

double cuda_add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (1.45f*n*n)/10.0f/2.0f;

#ifdef MODEL_DEBUG
	printf("add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}


double mult_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/1000.0f/4.11f/0.2588);

#ifdef MODEL_DEBUG
	printf("mult cost %e n = %d \n", cost, n);
#endif

	return PERTURBATE(cost);
}

double cuda_mult_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = (((double)(n)*n*n)/1000.0f/4.11f);

#ifdef MODEL_DEBUG
	printf("mult cost %e n = %d \n", cost, n);
#endif

	return PERTURBATE(cost);
}
