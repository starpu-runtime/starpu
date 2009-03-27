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
#define PERTURBATE(a)	((drand48()*2.0f*(AMPL) + 1.0f - (AMPL))*(a))
#else
#define PERTURBATE(a)	(a)
#endif


static double self_add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = (n*n)/10.0f/4.0f/7.75f;

#ifdef MODEL_DEBUG
	printf("self add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double cuda_self_add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = (n*n)/10.0f/4.0f;

#ifdef MODEL_DEBUG
	printf("self add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = (1.45f*n*n)/10.0f/2.0f;

#ifdef MODEL_DEBUG
	printf("add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double cuda_add_sub_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = (1.45f*n*n)/10.0f/2.0f;

#ifdef MODEL_DEBUG
	printf("add sub cost %e n = %d\n", cost, n);
#endif

	return PERTURBATE(cost);
}


static double mult_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/1000.0f/4.11f/0.2588);

#ifdef MODEL_DEBUG
	printf("mult cost %e n = %d \n", cost, n);
#endif

	return PERTURBATE(cost);
}

static double cuda_mult_cost(buffer_descr *descr)
{
	uint32_t n;

	n = get_blas_nx(descr[0].state);

	double cost = (((double)(n)*n*n)/1000.0f/4.11f);

#ifdef MODEL_DEBUG
	printf("mult cost %e n = %d \n", cost, n);
#endif

	return PERTURBATE(cost);
}

struct perfmodel_t strassen_model_mult = {
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = mult_cost },
		[CUDA_DEFAULT] = { .cost_model = cuda_mult_cost }
	},
	.type = HISTORY_BASED,
	.symbol = "strassen_model_mult"
};

struct perfmodel_t strassen_model_add_sub = {
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = add_sub_cost },
		[CUDA_DEFAULT] = { .cost_model = cuda_add_sub_cost }
	},
	.type = HISTORY_BASED,
	.symbol = "strassen_model_add_sub"
};

struct perfmodel_t strassen_model_self_add_sub = {
	.per_arch = { 
		[CORE_DEFAULT] = { .cost_model = self_add_sub_cost },
		[CUDA_DEFAULT] = { .cost_model = cuda_self_add_sub_cost }
	},
	.type = HISTORY_BASED,
	.symbol = "strassen_model_self_add_sub"
};
