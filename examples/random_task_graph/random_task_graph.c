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
 * This examples demonstrates how to construct and submit a task to StarPU and
 * more precisely:
 *  - how to allocate a new task structure (starpu_task_create)
 *  - how to describe a multi-versionned computational kernel (ie. a codelet) 
 *  - how to pass an argument to the codelet (task->cl_arg)
 *  - how to declare a callback function that is called once the task has been
 *    executed
 *  - how to specify if starpu_task_submit is a blocking or non-blocking
 *    operation (task->synchronous)
 */

#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
//~ #include <limits.h>
//~ #include <string.h>
//~ #include <unistd.h>
//~ #include <math.h>
//~ #include <stdbool.h>
//~ #include <sys/types.h>
//~ #include <starpu.h>
//~ #include <starpu_fxt.h>

//~ #include <common/blas.h>
//~ #include <cuda.h>
//~ #include <starpu_cublas_v2.h>
//~ static const TYPE p1 = 1.0;
//~ static const TYPE m1 = -1.0;
//~ static const TYPE v0 = 0.0;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)
#define TIME 0.010
#define TIME_CUDA_COEFFICIENT 10
//~ #define TIME_OPENCL_COEFFICIENT 5
//~ #define SECONDS_SCALE_COEFFICIENT_TIMING_NOW 1000000
//~ #define NB_FLOAT 400000
int number_task = 0;

/* When the task is done, task->callback_func(task->callback_arg) is called. Any
 * callback function must have the prototype void (*)(void *).
 * NB: Callback are NOT allowed to perform potentially blocking operations */
void callback_func(void *callback_arg)
{
        FPRINTF(stdout, "Callback function got argument %p\n", callback_arg);
}

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
	.cuda_funcs = {(void*)1},
//~ #endif
	//~ .cuda_flags = {STARPU_CUDA_ASYNC},
	//~ .nbuffers = STARPU_VARIABLE_NBUFFERS,
	.nbuffers = 1,
	.modes = {STARPU_R, STARPU_R, STARPU_R},
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
	printf("Main of examples/random_task_graph/random_task_graph.c\n");
	//~ struct starpu_codelet cl;
	//~ cl.nbuffers = STARPU_VARIABLE_NBUFFERS;
	int value=42;
	//~ struct params params = {1, 2.0f};
	int ret;
	int i = 0;
	starpu_data_handle_t handle1;
	starpu_variable_data_register(&handle1, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
	//~ starpu_data_unregister(handle);
	
	parse_args(argc, argv);
	
	PRINTF("# nb task \tGFlops\n");

	/* initialize StarPU : passing a NULL argument means that we use
 	* default configuration for the scheduling policies and the number of
	* processors/accelerators */
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	start = starpu_timing_now();

	/* create a new task that is non-blocking by default : the task is not
	 * submitted to the scheduler until the starpu_task_submit function is
	 * called */
	starpu_pause(); /* To get all tasks at once */
	for (i = 0; i < number_task; i++)
	{
		printf("Create task\n");
		struct starpu_task *task = starpu_task_create();

		//~ starpu_codelet_init(&cl_random_task_graph);
		/* this codelet may only be executed on a CPU, and its cpu
		* implementation is function "cpu_func" */
		//~ printf("After init codelet\n");
		//~ cl.cpu_funcs[0] = cpu_func;
		//~ cl.cpu_funcs_name[0] = "cpu_func";
		/* the codelet does not manipulate any data that is managed
		* by our DSM */
		//~ cl.name="random_task_graph";

		/* the task uses codelet "cl" */
		//~ task->cl = &cl_random_task_graph;
		task->cl = &cl_random_task_graph;
		
		/* It is possible to pass buffers that are not managed by the DSM to the
		 * kernels: the second argument of the "cpu_func" function is a pointer to a
		 * buffer that contains information for the codelet (cl_arg stands for
		 * codelet argument). In the case of accelerators, it is possible that
		 * the codelet is given a pointer to a copy of that buffer: this buffer
		 * is read-only so that any modification is not passed to other copies
		 * of the buffer.  For this reason, a buffer passed as a codelet
		 * argument (cl_arg) is NOT a valid synchronization medium! */
		//~ task->cl_arg = &params;
		//~ task->cl_arg_size = sizeof(params);

		/* once the task has been executed, callback_func(0x42)
		 * will be called on a CPU */
		//~ task->callback_func = callback_func;
		//~ task->callback_arg = (void*) (uintptr_t) 0x42;
		
		/* starpu_task_submit will be a blocking call */
		//~ task->synchronous = 1;
		//~ task->cl->nbuffers = 1;
		task->handles[0] = handle1;
		printf("%p\n", task->handles[0]);
		
		/* submit the task to StarPU */
		printf("Before submit\n");
		ret = starpu_task_submit(task);
		printf("After submit\n");
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	starpu_resume(); /* Because I paused above */
	starpu_task_wait_for_all();
	starpu_data_unregister(handle1);
	
	end = starpu_timing_now();
	double timing = end - start;
	double flops = number_task;
	PRINTF("  %d		%f		\t%.10f\n", number_task, timing, flops/timing);
	
	/* terminate StarPU: statistics and other debug outputs are not
	 * guaranteed to be generated unless this function is called. Once it
	 * is called, it is not possible to submit tasks anymore, and the user
	 * is responsible for making sure all tasks have already been executed:
	 * calling starpu_shutdown() before the termination of all the tasks
	 * results in an undefined behaviour */
	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
