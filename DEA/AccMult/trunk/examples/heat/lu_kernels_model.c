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


double task_11_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = ((n*n*n)/50.0f/10.75);

	return PERTURBATE(cost);
}

double task_12_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = ((n*n*n)/50.0f/4.11f/8.49);

	return PERTURBATE(cost);
}


double task_21_cost(buffer_descr *descr)
{
	uint32_t n;

	n = descr[0].state->interface->blas.nx;

	double cost = ((n*n*n)/50.0f/4.11f/8.49);

	return PERTURBATE(cost);
}



double task_22_cost(buffer_descr *descr)
{
	uint32_t nx, ny, nz;

	nx = descr[2].state->interface->blas.nx;
	ny = descr[2].state->interface->blas.ny; 
	nz = descr[0].state->interface->blas.ny;

	double cost = ((nx*ny*nz)/1000.0f/4.11f);

	return PERTURBATE(cost);
}
