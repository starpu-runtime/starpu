/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2015-2016  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013  CNRS
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
 * This examples demonstrates how to use multiple linear regression models.

   The duration of the task test_mlr will
   be computed using the following equation:

   T = a + b * (M^2*N) + c * (N^3*K)

   where M, N, K are the parameters of the task,
   exponents are coming from cl.model->combinations[..][..] 
   and finally a, b, c are coefficients
   which mostly depend on the machine speed. 
   
   These coefficients are going to be automatically computed	
   using least square method.

 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <starpu.h>

static long sum;

/* Performance function of the task, which is in this case very simple, as the parameter values just need to be written in the array "parameters" */
static void cl_params(struct starpu_task *task, double *parameters)
{
	starpu_codelet_unpack_args(task->cl_arg,
			     	  &parameters[0],
     			     	  &parameters[1],
     			     	  &parameters[2]);
}

/* Function of the task that will be executed. In this case running dummy cycles, just to make sure task duration is significant */
void cpu_func(void *buffers[], void *cl_arg)
{
	long i;
	double m,n,k;
	starpu_codelet_unpack_args(cl_arg,
			     	  &m,
     			     	  &n,
     			     	  &k);
	
	for(i=0; i < (long) (m*m*n); i++)
		sum+=i;

	for(i=0; i < (long) (n*n*n*k); i++)
		sum+=i;
}

/* ############################################ */
/* Start of the part specific to multiple linear regression perfmodels */

/* Defining perfmodel, number of parameters and their names  */

/* Defining the equation for modeling duration of the task */
/* Refer to the explanation and equation on the top of this file
   to get more detailed explanation, here we have M^2*N and N^3*K */

static const char * parameters_names[]	= {	"M",	"N",	"K", };
static unsigned combi1 [3]		= {	2,	1,	0 };
static unsigned combi2 [3]		= {	0,	3,	1 };

static unsigned *combinations[] = { combi1, combi2 };

static struct starpu_perfmodel cl_model = {
	.type = STARPU_MULTIPLE_REGRESSION_BASED,
	.symbol = "test_mlr",
	.parameters = cl_params,
	.nparameters = 3,
	.parameters_names = parameters_names,
	.ncombinations = 2,
	.combinations = combinations,
};

static struct starpu_codelet cl = {
	.cpu_funcs = { cpu_func },
	.cpu_funcs_name = { "mlr_codelet" },
	.nbuffers = 0,
	.model = &cl_model,
};

/* End of the part specific to multiple linear regression perfmodels */
/* ############################################ */
	
int main(int argc, char **argv)
{
	/* Initialization */
	unsigned i,j;
	int ret;
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	
	sum=0;
	double m,n,k;

        /* Giving pseudo-random values to the M,N,K parameters and inserting tasks */
	for(i=0; i < 42; i++)
	{
		m = (double) ((rand() % 10)+1);
		n = (double) ((rand() % 10)+1);
		k = (double) ((rand() % 10)+1);
		
		for(j=0; j < 42; j++)
			starpu_insert_task(&cl,
				   STARPU_VALUE, &m, sizeof(double),
				   STARPU_VALUE, &n, sizeof(double),
				   STARPU_VALUE, &k, sizeof(double),
				   0);
	}
			  
	starpu_shutdown();

	return 0;
}
