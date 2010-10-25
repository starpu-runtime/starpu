/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <starpu_cuda.h>
#include <stdlib.h>

#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_CURAND)
#error CURAND is required to run that example on CUDA devices
#endif

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <curand.h>
#endif

#define NSHOT_PER_TASK	(1024*1024)

/* default value */
static unsigned ntasks = 128;

/*
 *	Initialization of the Random Number Generators (RNG)
 */

#ifdef STARPU_USE_CUDA
/* RNG for the CURAND library */
static curandGenerator_t curandgens[STARPU_NMAXWORKERS];
#endif 

/* state for the erand48 function */
static unsigned short xsubi[3*STARPU_NMAXWORKERS];

/* Function to initialize the random number generator in the current worker */
static void init_rng(void *arg __attribute__((unused)))
{
#ifdef STARPU_USE_CUDA
	curandStatus_t res;
#endif

	int workerid = starpu_worker_get_id();

	switch (starpu_worker_get_type(workerid)) {
		case STARPU_CPU_WORKER:
			/* create a seed */
			xsubi[0 + 3*workerid] = (unsigned short)workerid;
			xsubi[1 + 3*workerid] = (unsigned short)workerid;
			xsubi[2 + 3*workerid] = (unsigned short)workerid;
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_WORKER:

			/* Create a RNG */
			res = curandCreateGenerator(&curandgens[workerid],
						CURAND_RNG_PSEUDO_DEFAULT);
			STARPU_ASSERT(res == CURAND_STATUS_SUCCESS);

			/* Seed it with worker's id */
			res = curandSetPseudoRandomGeneratorSeed(curandgens[workerid],
							(unsigned long long)workerid);
			STARPU_ASSERT(res == CURAND_STATUS_SUCCESS);
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-ntasks") == 0) {
			char *argptr;
			ntasks = strtol(argv[++i], &argptr, 10);
		}
	}
}

/*
 *	Monte-carlo kernel
 */

static void pi_func_cpu(void *descr[], void *cl_arg __attribute__ ((unused)))
{
	int workerid = starpu_worker_get_id();

	unsigned short *worker_xsub;
	worker_xsub = &xsubi[3*workerid];

	unsigned long local_cnt = 0;

	/* Fill the scratchpad with random numbers */
	int i;
	for (i = 0; i < NSHOT_PER_TASK; i++)
	{
		float x = (float)(2.0*erand48(worker_xsub) - 1.0);
		float y = (float)(2.0*erand48(worker_xsub) - 1.0);

		float dist = x*x + y*y;
		if (dist < 1.0)
			local_cnt++;
	}

	/* Put the contribution of that task into the counter */
	unsigned long *cnt = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);
	*cnt = *cnt + local_cnt;
}

extern void pi_redux_cuda_kernel(float *x, float *y, unsigned n, unsigned long *shot_cnt);

static void pi_func_cuda(void *descr[], void *cl_arg __attribute__ ((unused)))
{
	cudaError_t cures;
	curandStatus_t res;	

	int workerid = starpu_worker_get_id();

	/* CURAND is a bit silly: it assumes that any error is fatal. Calling
	 * cudaGetLastError resets the last error value. */
	cures = cudaGetLastError();
//	if (cures)
//		STARPU_CUDA_REPORT_ERROR(cures);

	/* Fill the scratchpad with random numbers. Note that both x and y
	 * arrays are in stored the same vector. */
	float *scratchpad_xy = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	res = curandGenerateUniform(curandgens[workerid], scratchpad_xy, 2*NSHOT_PER_TASK);
	STARPU_ASSERT(res == CURAND_STATUS_SUCCESS);

	float *x = &scratchpad_xy[0];
	float *y = &scratchpad_xy[NSHOT_PER_TASK];

	unsigned long *shot_cnt = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);
	pi_redux_cuda_kernel(x, y, NSHOT_PER_TASK, shot_cnt);
}

static struct starpu_codelet_t pi_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = pi_func_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = pi_func_cuda,
#endif
	.nbuffers = 2,
	.model = NULL
};

/*
 *	Codelets to implement reduction
 */

static void init_cpu_func(void *descr[], void *cl_arg)
{
        unsigned long *val = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
        *val = 0;
}

#ifdef STARPU_USE_CUDA
static void init_cuda_func(void *descr[], void *cl_arg)
{
        unsigned long *val = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
        cudaMemset(val, 0, sizeof(unsigned long));
        cudaThreadSynchronize();
}
#endif

static struct starpu_codelet_t init_codelet = {
        .where = STARPU_CPU|STARPU_CUDA,
        .cpu_func = init_cpu_func,
#ifdef STARPU_USE_CUDA
        .cuda_func = init_cuda_func,
#endif
        .nbuffers = 1
};


void redux_cpu_func(void *descr[], void *cl_arg)
{
	unsigned long *a = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned long *b = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*a = *a + *b;
};

static struct starpu_codelet_t redux_codelet = {
	.where = STARPU_CPU,
	.cpu_func = redux_cpu_func,
	.nbuffers = 2
};

/*
 *	Main program
 */

int main(int argc, char **argv)
{
	unsigned i;

	parse_args(argc, argv);

	starpu_init(NULL);

	/* Launch a Random Number Generator (RNG) on each worker */
	starpu_execute_on_each_worker(init_rng, NULL, STARPU_CPU|STARPU_CUDA);

	/* Create a scratchpad data */
	starpu_data_handle xy_scratchpad_handle;
	starpu_vector_data_register(&xy_scratchpad_handle, -1, (uintptr_t)NULL,
		2*NSHOT_PER_TASK, sizeof(float));

	/* Create a variable that will be used to count the number of shots
	 * that actually hit the unit circle when shooting randomly in
	 * [-1,1]^2. */
	unsigned long shot_cnt = 0;
	starpu_data_handle shot_cnt_handle;
	starpu_variable_data_register(&shot_cnt_handle, 0,
			(uintptr_t)&shot_cnt, sizeof(shot_cnt));

	starpu_data_set_reduction_methods(shot_cnt_handle,
					&redux_codelet, &init_codelet);
	starpu_data_start_reduction_mode(shot_cnt_handle);

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &pi_cl;

		task->buffers[0].handle = xy_scratchpad_handle;
		task->buffers[0].mode   = STARPU_SCRATCH;
		task->buffers[1].handle = shot_cnt_handle;
		task->buffers[1].mode   = STARPU_REDUX;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_task_wait_for_all();
	starpu_data_end_reduction_mode(shot_cnt_handle);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	starpu_data_unregister(shot_cnt_handle);

	/* Total surface : Pi * r^ 2 = Pi*1^2, total square surface : 2^2 = 4, probability to impact the disk: pi/4 */
	unsigned long total = ntasks*NSHOT_PER_TASK;	
	fprintf(stderr, "Pi approximation : %f (%ld / %ld)\n",
			((float)shot_cnt*4.0)/total, shot_cnt, total);
	fprintf(stderr, "Total time : %f ms\n", timing/1000.0);
	fprintf(stderr, "Speed : %f GShot/s\n", total/(1e3*timing));

	starpu_shutdown();

	return 0;
}
