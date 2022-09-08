/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 
/**
 * void INSERT_TASK_map( const RUNTIME_option_t *options,
                      cham_uplo_t uplo, const CHAM_desc_t *A, int Am, int An,
                      cham_unary_operator_t op_fct, void *op_args );
 * compute/pzgeqrf.c 
 * Commence à chameleon_pzgeqrf puis va dans chameleon_pzgeqrf_step
 * genD == 0
**/

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <starpu.h>
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)
//~ #define TIME 0.010 /* original value */
//~ #define TIME 0.011
//~ #define TIME_CUDA_COEFFICIENT 10 /* original value */
//~ #define TIME_CUDA_COEFFICIENT 1
#define COUNT_DO_SCHEDULE /* do schedule for HFP compté ou non */
#define SEED

/* Global variables of sizes of matrices */
int A_M = 0;
int A_N = 0;

void wait_CUDA (void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
	starpu_sleep(0.011);
	//~ starpu_sleep(TIME/TIME_CUDA_COEFFICIENT); /* original value */
}

double cost_function (struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	(void) t; (void) i;
	STARPU_ASSERT(a->ndevices == 1);
	if (a->devices[0].type == STARPU_CUDA_WORKER)
	{
		//~ return TIME/TIME_CUDA_COEFFICIENT * 1000000; /* Original value */
		return 11000;
	}
	STARPU_ASSERT(0);
	return 0.0;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
	.symbol = "qr_without_dependancies",
};

/* Codelet for random task graph */
static struct starpu_codelet cl_qr_without_dependancies =
{
	.cuda_funcs = {wait_CUDA},
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
	.model = &perf_model
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-M") == 0)
		{
			char *argptr;
			A_M = strtol(argv[++i], &argptr, 10);
		}
		else if (strcmp(argv[i], "-N") == 0)
		{
			char *argptr;
			A_N = strtol(argv[++i], &argptr, 10);
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv)
{
	double start, end;
	parse_args(argc, argv);
	int ret;
	int k = 0;
	int number_forbidden_data = 0;
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	
	if (A_M > A_N)
	{
        minMNT = A_N
    } 
    else
    {
        minMNT = A_M;
    }
    
    /* pzgeqrf */
    //~ for (k = 0; k < minMNT; k++)
    //~ {
        //~ chameleon_pzgeqrf_step( genD, k, ib,
                                //~ A, T, D, &options, sequence );
    //~ }


	int i = 0;
	int j = 0;
	starpu_data_handle_t * tab_handle = malloc(number_data*sizeof(starpu_data_handle_t));
	int * forbidden_data = malloc(number_data*sizeof(int));
	if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Set of data: "); }
	for (i = 0; i < number_data; i++)
	{
		starpu_data_handle_t new_handle;
		starpu_variable_data_register(&new_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		tab_handle[i] = new_handle;
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("%p ", new_handle); }
	}
	if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("\n"); }

	start = starpu_timing_now();
	starpu_pause();
	starpu_sleep(0.001);
	for (i = 0; i < number_task; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl_random_task_graph;
				
		random_degree = random()%degree_max + 1;
		task->nbuffers = random_degree;
		for (j = 0; j < number_data; j++)
		{
			forbidden_data[j] = 0; /* 0 = not forbidden, 1 = forbidden because already in the task */
		}
		number_forbidden_data = 0;
		for (j = 0; j < random_degree; j++)
		{
			/* A task can't have two times the same data. So I put 1 in forbidden_data in the corresponding space. Each time our random number pass over a forbidden data, I add 1 to random_data. Also the modulo for random_data is on the number of remaining data. */
			random_data = random()%(number_data - number_forbidden_data);
			for (k = 0; k <= random_data; k++)
			{
				if (forbidden_data[k] == 1)
				{
					random_data++;
				}
			}
			task->handles[j] = tab_handle[random_data];
			forbidden_data[random_data] = 1;
			number_forbidden_data++;
			task->modes[j] = STARPU_R; /* Acces mode of each data set here because the codelet won't work if the number of data is variable */
		}
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Created task %p, with %d data:", task, random_degree);
		for (j = 0; j < random_degree; j++) { printf(" %p", task->handles[j]); } printf("\n"); }
				
		/* submit the task to StarPU */
		ret = starpu_task_submit(task);
		if (ret == -ENODEV) { goto enodev; }
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	starpu_resume();
	//~ starpu_task_wait_for_all();
	
					//~ if (starpu_get_env_number_default("COUNT_DO_SCHEDULE", 0) == 0)
				//~ {
					//~ starpu_do_schedule();
					//~ printf("la1.5 random set of tasks \n");
					//~ start = starpu_timing_now();	
					//~ printf("la1.6 random set of tasks \n");				
					//~ starpu_resume();
					//~ printf("la1.7 random set of tasks \n");
					//~ starpu_task_wait_for_all();
					//~ printf("la1.8 random set of tasks \n");
					//~ end = starpu_timing_now();
				//~ }
				//~ else
				//~ {
					//~ start = starpu_timing_now();
					//~ starpu_do_schedule();		
					//~ starpu_resume();
					//~ starpu_task_wait_for_all();
					//~ end = starpu_timing_now();
				//~ }
	//~ for (i = 0; i < number_data; i++)
	//~ {
		//~ starpu_data_unregister(tab_handle[i]);
	//~ }
	end = starpu_timing_now();
	double timing = end - start;
	double temp_number_task = number_task;
	double flops = 960*temp_number_task*960*960*4;
	//~ printf("flops : %f, time : %f\n", flops, timing);
	PRINTF("# Nbtasks\tms\tGFlops\n");
	PRINTF("%d\t%.0f\t%.1f\n", number_task, timing/1000.0, flops/timing/1000);
	
	for (i = 0; i < number_data; i++)
	{
		starpu_data_unregister(tab_handle[i]);
	}
	
	starpu_shutdown();
	
	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
