/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2015  Université de Bordeaux
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

long sum;

/* Performance function of the task, which is in this case very simple, as the parameter values just need to be written in the array "parameters" */
void cl_perf_func(struct starpu_task *task, double *parameters)
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

int main(int argc, char **argv)
{
	/* Initialization */
	unsigned i,j;
	struct starpu_codelet cl;
	int ret;
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	
	/* Allocating and naming codelet, similar to any other StarPU program */
	memset(&cl, 0, sizeof(cl));	
	cl.cpu_funcs[0] = cpu_func;
	cl.cpu_funcs_name[0] = "mlr_codelet";
	cl.nbuffers = 0;
	cl.name="test_mlr";

	/* ############################################ */
	/* Start of the part specific to multiple linear regression perfmodels */
	
	/* Defining perfmodel, number of parameters and their names  */
	struct starpu_perfmodel *model = calloc(1,sizeof(struct starpu_perfmodel));
	cl.model = model;
	cl.model->type = STARPU_MULTIPLE_REGRESSION_BASED;
	cl.model->symbol = cl.name;
	cl.model->parameters = cl_perf_func;
	cl.model->nparameters = 3;
	cl.model->parameters_names = (const char **) calloc(1, cl.model->nparameters*sizeof(char *));
	cl.model->parameters_names[0] = "M";
	cl.model->parameters_names[1] = "N";
	cl.model->parameters_names[2] = "K";

	/* Defining the equation for modeling duration of the task */
	/* Refer to the explanation and equation on the top of this file
	   to get more detailed explanation */
	cl.model->ncombinations = 2;
	cl.model->combinations = (unsigned **) malloc(cl.model->ncombinations*sizeof(unsigned *));

	if (cl.model->combinations)
	{
		for (i=0; i < cl.model->ncombinations; i++)
		{
			cl.model->combinations[i] = (unsigned *) 	malloc(cl.model->nparameters*sizeof(unsigned));
		}
	}

	cl.model->combinations[0][0] = 2;
	cl.model->combinations[0][1] = 1;
	cl.model->combinations[0][2] = 0;

	cl.model->combinations[1][0] = 0;
	cl.model->combinations[1][1] = 3;
	cl.model->combinations[1][2] = 1;

	/* End of the part specific to multiple linear regression perfmodels */
	/* ############################################ */
	
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
