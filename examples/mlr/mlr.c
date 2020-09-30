/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This examples demonstrates how to use multiple linear regression
   models.

   First, there is mlr_codelet__init codelet for which we know the
   parameters, but not the their exponents and relations. This tasks
   should be benchmarked and analyzed to find the model, using
   "tools/starpu_mlr_analysis" script as a template.

   For the second (codelet cl_model_final), it is assumed that the
   analysis has already been performed and that the duration of the
   codelet mlr_codelet_final will be computed using the following
   equation:

   T = a + b * (M^2*N) + c * (N^3*K)

   where M, N, K are the parameters of the task, exponents are coming
   from model->combinations[..][..]  and finally a, b, c are
   coefficients which mostly depend on the machine speed.

   These coefficients are going to be automatically computed using
   least square method.

 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <starpu.h>

static long sum;

/* Performance function of the task, which is in this case very simple, as the parameter values just need to be written in the array "parameters" */
static void cl_params(struct starpu_task *task, double *parameters)
{
	int m, n, k;
	int* vector_mn;

	vector_mn = (int*)STARPU_VECTOR_GET_PTR(task->interfaces[0]);
	m = vector_mn[0];
	n = vector_mn[1];

	starpu_codelet_unpack_args(task->cl_arg, &k);

	parameters[0] = m;
	parameters[1] = n;
	parameters[2] = k;
}

/* Function of the task that will be executed. In this case running dummy cycles, just to make sure task duration is significant */
void cpu_func(void *buffers[], void *cl_arg)
{
	long i;
	int m,n,k;
	int* vector_mn;

	vector_mn = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	m = vector_mn[0];
	n = vector_mn[1];

	starpu_codelet_unpack_args(cl_arg, &k);

	for(i=0; i < (long) (m*m*n); i++)
		sum+=i;

	for(i=0; i < (long) (n*n*n*k); i++)
		sum+=i;
}

/* ############################################ */
/* Start of the part specific to multiple linear regression perfmodels */

/* Defining perfmodel, number of parameters and their names Initially
   application developer only knows these parameters. The execution of
   this codelet will generate traces that can be analyzed using
   "tools/starpu_mlr_analysis" as a template to obtain the parameters
   combinations and exponents.
 */

static const char * parameters_names[]	= {	"M",	"N",	"K", };

static struct starpu_perfmodel cl_model_init =
{
	.type = STARPU_MULTIPLE_REGRESSION_BASED,
	.symbol = "mlr_init",
	.parameters = cl_params,
	.nparameters = 3,
	.parameters_names = parameters_names,
};

/* Defining the equation for modeling duration of the task. The
   parameters combinations and exponents are computed externally
   offline, for example using "tools/starpu_mlr_analysis" tool as a
   template.
 */

static unsigned combi1 [3]		= {	2,	1,	0 };
static unsigned combi2 [3]		= {	0,	3,	1 };

static unsigned *combinations[] = { combi1, combi2 };

static struct starpu_perfmodel cl_model_final =
{
	.type = STARPU_MULTIPLE_REGRESSION_BASED,
	.symbol = "mlr_final",
	.parameters = cl_params,
	.nparameters = 3,
	.parameters_names = parameters_names,
	.ncombinations = 2,
	.combinations = combinations,
};

/* End of the part specific to multiple linear regression perfmodels */
/* ############################################ */

static struct starpu_codelet cl_init =
{
	.cpu_funcs = { cpu_func },
	.cpu_funcs_name = { "cpu_func" },
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &cl_model_init,
};

static struct starpu_codelet cl_final =
{
	.cpu_funcs = { cpu_func },
	.cpu_funcs_name = { "cpu_func" },
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &cl_model_final,
};


int main(void)
{
	/* Initialization */
	unsigned i;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;

	sum=0;
	int* vector_mn = calloc(2, sizeof(int));
	starpu_data_handle_t vector_mn_handle;

	starpu_vector_data_register(&vector_mn_handle,
				    STARPU_MAIN_RAM,
				    (uintptr_t)vector_mn, 2,
				    sizeof(int));

	/* Giving pseudo-random values to the M,N,K parameters and inserting tasks */
	for (i = 0; i < 42; i++)
	{
		int j;
		int m,n,k;

		m = (int) ((rand() % 10)+1);
		n = (int) ((rand() % 10)+1);
		k = (int) ((rand() % 10)+1);

		/* To illustrate the usage, M and N are stored in a data handle */
		starpu_data_acquire(vector_mn_handle, STARPU_W);
		vector_mn[0] = m;
		vector_mn[1] = n;
		starpu_data_release(vector_mn_handle);

		for (j = 0; j < 1000; j++)
		{
			starpu_insert_task(&cl_init,
					   STARPU_R, vector_mn_handle,
					   STARPU_VALUE, &k, sizeof(int),
					   0);
			starpu_insert_task(&cl_final,
					   STARPU_R, vector_mn_handle,
					   STARPU_VALUE, &k, sizeof(int),
					   0);
		}
	}

	starpu_data_unregister(vector_mn_handle);
	free(vector_mn);
	starpu_shutdown();

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	starpu_perfmodel_dump_xml(stdout, &cl_model_final);
	starpu_shutdown();

	return 0;
}
