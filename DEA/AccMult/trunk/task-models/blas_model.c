#include <task-models/task_model.h>
#include <task-models/blas_model.h>

/*
 * As a convention, in that file, descr[0] is represented by A,
 * 				  descr[1] is B ...
 */

/*
 *	Number of flops of Gemm 
 */

double gemm_cost(buffer_descr *descr)
{
	/* C = A * B */
	uint32_t nxC, nyC, nxA;


	nxC = descr[2].state->interface->blas.nx;
	nyC = descr[2].state->interface->blas.ny;
	nxA = descr[0].state->interface->blas.nx;

//	printf("nxC %d nxC %d nxA %d\n", nxC, nyC, nxA);

	double cost = ((double)nxC)*((double)nyC)*((double)nxA/1000.0f/4.11f);

//	printf("cost %e \n", cost);

	return cost;
}
