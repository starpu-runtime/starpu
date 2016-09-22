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
 * This examples demonstrates how to construct and submit a task to StarPU and
 * more precisely:
 *  - how to...
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <starpu.h>

int sum;

void cl_perf_func(struct starpu_task *task, double *parameters)
{
	starpu_codelet_unpack_args(task->cl_arg,
			     	  &parameters[0],
     			     	  &parameters[1],
     			     	  &parameters[2]);
}

void cpu_func(void *buffers[], void *cl_arg)
{
	double m,n,k;
	starpu_codelet_unpack_args(cl_arg,
			     	  &m,
     			     	  &n,
     			     	  &k);
	
	for(int i=0; i < (int) (m*m*n); i++)
		sum+=i;

	for(int i=0; i < (int) (n*n*n*k); i++)
		sum+=i;
}

int main(int argc, char **argv)
{
	struct starpu_codelet cl;
	starpu_init(NULL);

	memset(&cl, 0, sizeof(cl));	
	cl.cpu_funcs[0] = cpu_func;
	cl.cpu_funcs_name[0] = "mlr_codelet";
	cl.nbuffers = 0;
	cl.name="test_mlr";

	/* ############################################ */
	/* Defining perfmodel, #parameters and their names  */
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
	
	cl.model->ncombinations = 2;
	cl.model->combinations = (unsigned **) malloc(cl.model->ncombinations*sizeof(unsigned *));

	if (cl.model->combinations)
	{
		for (unsigned i = 0; i < cl.model->ncombinations; i++)
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
	/* ############################################ */
	
	sum=0;
	
	double *parameters = (double*) calloc(1,cl.model->nparameters*sizeof(double));	
	
	for(int i=0; i < 42; i++)
	{
		parameters[0] = (double) ((rand() % 10)+1);
		parameters[1] = (double) ((rand() % 10)+1);
		parameters[2] = (double) ((rand() % 10)+1);

		for(int j=0; j < 42; j++)
			starpu_insert_task(&cl,
				   STARPU_VALUE, &parameters[0], sizeof(double),
				   STARPU_VALUE, &parameters[1], sizeof(double),
				   STARPU_VALUE, &parameters[2], sizeof(double),
				   0);
	}
			  
	starpu_shutdown();

	return 0;
}
