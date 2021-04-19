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

/*
 * Random set of task. Parameters to enter:
 * Number of tasks: -ntasks x
 * Number of different data: -ndata x
 * Degree max of a task: -degreemax x
 * For now the number of degree of a task is a random number between 1 and degreemax
 */

#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

//~ #include <limits.h>
//~ #include <time.h>
//~ #include <string.h>
//~ #include <unistd.h>
//~ #include <math.h>
//~ #include <stdbool.h>
//~ #include <sys/types.h>
//~ #include <starpu.h>
//~ #include <starpu_fxt.h>
//~ #include <common/blas.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)
#define TIME 0.010
#define TIME_CUDA_COEFFICIENT 10
//~ #define TIME_OPENCL_COEFFICIENT 5
//~ #define SECONDS_SCALE_COEFFICIENT_TIMING_NOW 1000000
//~ #define NB_FLOAT 400000
int number_task = 0;
int number_data = 0;
int degree_max = 0;

/* When the task is done, task->callback_func(task->callback_arg) is called. Any
 * callback function must have the prototype void (*)(void *).
 * NB: Callback are NOT allowed to perform potentially blocking operations */
//~ void callback_func(void *callback_arg)
//~ {
        //~ FPRINTF(stdout, "Callback function got argument %p\n", callback_arg);
//~ }

/* Every implementation of a codelet must have this prototype, the first
 * argument (buffers) describes the buffers/streams that are managed by the
 * DSM; the second arguments references read-only data that is passed as an
 * argument of the codelet (task->cl_arg). Here, "buffers" is unused as there
 * are no data input/output managed by the DSM (cl.nbuffers = 0) */
struct params
{
	int i;
	float f;
};

void wait_CUDA(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
	starpu_sleep(TIME/TIME_CUDA_COEFFICIENT);
}

double cost_function(struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	(void) t; (void) i;
	STARPU_ASSERT(a->ndevices == 1);
	if (a->devices[0].type == STARPU_CUDA_WORKER)
	{
		return TIME/TIME_CUDA_COEFFICIENT * 1000000;
	}
	STARPU_ASSERT(0);
	return 0.0;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
};

/* Codelet for random task graph */
static struct starpu_codelet cl_random_task_graph =
{
	//~ .type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	//~ .max_parallelism = INT_MAX,
	//~ .cpu_funcs = {cpu_mult},
	//~ .cpu_funcs_name = {"cpu_gemm"},
//~ #ifdef STARPU_USE_CUDA
	.cuda_funcs = {wait_CUDA},
//~ #elif defined(STARPU_SIMGRID)
	//~ .cuda_funcs = {(void*)1},
//~ #endif
	//~ .cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
	//~ .nbuffers = 1,
	//~ .modes = {STARPU_R, STARPU_R, STARPU_R},
	//~ .modes = {STARPU_R},
	.model = &perf_model
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-ntasks") == 0)
		{
			char *argptr;
			number_task = strtol(argv[++i], &argptr, 10);
		}
		else if (strcmp(argv[i], "-ndata") == 0)
		{
			char *argptr;
			number_data = strtol(argv[++i], &argptr, 10);
		}
		else if (strcmp(argv[i], "-degreemax") == 0)
		{
			char *argptr;
			degree_max = strtol(argv[++i], &argptr, 10);
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			exit(EXIT_FAILURE);
		}
	}
	if (degree_max > number_data)
	{
		fprintf(stderr,"Too few data for the maximum degree of a task\n");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv)
{
	double start, end;
	parse_args(argc, argv);
	printf("Main of examples/random_task_graph/random_task_graph.c, %d tasks, %d different data, degree max of a task of %d\n", number_task, number_data, degree_max);
	int random_data = 0;
	int ret;
	int k = 0;
	int number_forbidden_data = 0;
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	
	int value=42;	
	int i = 0;
	int j = 0;
	int random_degree = 0;
	//~ starpu_data_handle_t new_handle;
	//~ starpu_variable_data_register(&new_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
	starpu_data_handle_t * tab_handle = malloc(number_data*sizeof(starpu_data_handle_t));
	int * forbidden_data = malloc(number_data*sizeof(int));
	printf("Set of data: ");
	for (i = 0; i < number_data; i++)
	{
		starpu_data_handle_t new_handle;
		starpu_variable_data_register(&new_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		tab_handle[i] = new_handle;
		printf("%p ", new_handle);
	}
	printf("\n");

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
			printf("random_data = %d\n", random_data);
			for (k = 0; k <= random_data; k++)
			{
				if (forbidden_data[k] == 1)
				{
					random_data++;
				}
			}
			printf("Adding %p\n", tab_handle[random_data]);
			task->handles[j] = tab_handle[random_data];
			forbidden_data[random_data] = 1;
			number_forbidden_data++;
			task->modes[j] = STARPU_R; /* Acces mode of each data set here because the codelet won't work if the number of data is variable */
		}
		printf("Created task %p, with %d data:", task, random_degree);
		for (j = 0; j < random_degree; j++) { printf(" %p", task->handles[j]); } printf("\n");
		
		//~ task->cl->nbuffers = 1;
		//~ task->handles[0] = new_handle;
		
		/* submit the task to StarPU */
		ret = starpu_task_submit(task);
		if (ret == -ENODEV) { goto enodev; }
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	starpu_resume();
	starpu_task_wait_for_all();
	
	for (i = 0; i < number_data; i++)
	{
		//~ starpu_data_unregister(new_handle);
		starpu_data_unregister(tab_handle[i]);
	}
	
	end = starpu_timing_now();
	double timing = end - start;
	double flops = number_task;
	PRINTF("# Nb task \tGFlops\n");
	PRINTF("  %d		%f		\t%.10f\n", number_task, timing, flops/timing);
	
	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
