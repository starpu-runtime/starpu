/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This computes Pi by using drawing random coordinates (thanks to the sobol
 * generator) and check whether they fall within one quarter of a circle.  The
 * proportion gives an approximation of Pi. For each task, we draw a number of
 * coordinates, and we gather the number of successful draws.
 *
 * This version uses reduction to optimize gathering the number of successful
 * draws.
 */

#include <starpu.h>
#include <stdlib.h>
#include <math.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PI	3.14159265358979323846

#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CURAND)
#warning CURAND is required to run that example on CUDA devices
#endif

#ifdef STARPU_HAVE_CURAND
#include <cuda.h>
#include <curand.h>
#endif

static unsigned long long nshot_per_task = 16*1024*1024ULL;

/* default value */
static unsigned long ntasks = 1024;
static unsigned long ntasks_warmup = 0;

static unsigned use_redux = 1;
static unsigned do_warmup = 0;

/*
 *	Initialization of the Random Number Generators (RNG)
 */

#ifdef STARPU_HAVE_CURAND
/* RNG for the CURAND library */
static curandGenerator_t curandgens[STARPU_NMAXWORKERS];
#endif

/* state for the erand48 function : note the huge padding to avoid false-sharing */
#define PADDING	1024
static unsigned short xsubi[STARPU_NMAXWORKERS*PADDING];
static starpu_drand48_data randbuffer[STARPU_NMAXWORKERS*PADDING];

/* Function to initialize the random number generator in the current worker */
static void init_rng(void *arg)
{
	(void)arg;
#ifdef STARPU_HAVE_CURAND
	curandStatus_t res;
#endif

	int workerid = starpu_worker_get_id_check();

	switch (starpu_worker_get_type(workerid))
	{
		case STARPU_CPU_WORKER:
		case STARPU_MIC_WORKER:
			/* create a seed */
			starpu_srand48_r((long int)workerid, &randbuffer[PADDING*workerid]);

			xsubi[0 + PADDING*workerid] = (unsigned short)workerid;
			xsubi[1 + PADDING*workerid] = (unsigned short)workerid;
			xsubi[2 + PADDING*workerid] = (unsigned short)workerid;
			break;
#ifdef STARPU_HAVE_CURAND
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

/* The amount of work does not depend on the data size at all :) */
static size_t size_base(struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	return nshot_per_task;
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-ntasks") == 0)
		{
			char *argptr;
			ntasks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nshot") == 0)
		{
			char *argptr;
			nshot_per_task = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-noredux") == 0)
		{
			use_redux = 0;
		}

		if (strcmp(argv[i], "-warmup") == 0)
		{
			do_warmup = 1;
			ntasks_warmup = 8; /* arbitrary number of warmup tasks */
		}

		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			fprintf(stderr, "Usage: %s [-ntasks n] [-noredux] [-warmup] [-h]\n", argv[0]);
			exit(-1);
		}
	}
}

/*
 *	Monte-carlo kernel
 */

void pi_func_cpu(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	int workerid = starpu_worker_get_id_check();

	unsigned short *worker_xsub;
	worker_xsub = &xsubi[PADDING*workerid];

	starpu_drand48_data *buffer;
	buffer = &randbuffer[PADDING*workerid];

	unsigned long local_cnt = 0;

	/* Fill the scratchpad with random numbers */
	unsigned i;
	for (i = 0; i < nshot_per_task; i++)
	{
		double randx, randy;

		starpu_erand48_r(worker_xsub, buffer, &randx);
		starpu_erand48_r(worker_xsub, buffer, &randy);

		double x = (2.0*randx - 1.0);
		double y = (2.0*randy - 1.0);

		double dist = x*x + y*y;
		if (dist < 1.0)
			local_cnt++;
	}

	/* Put the contribution of that task into the counter */
	unsigned long *cnt = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);
	*cnt = *cnt + local_cnt;
}

extern void pi_redux_cuda_kernel(float *x, float *y, unsigned n, unsigned long *shot_cnt);

#ifdef STARPU_HAVE_CURAND
static void pi_func_cuda(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	curandStatus_t res;

	int workerid = starpu_worker_get_id_check();

	/* CURAND is a bit silly: it assumes that any error is fatal. Calling
	 * cudaGetLastError resets the last error value. */
	(void) cudaGetLastError();

	/* Fill the scratchpad with random numbers. Note that both x and y
	 * arrays are in stored the same vector. */
	float *scratchpad_xy = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	res = curandGenerateUniform(curandgens[workerid], scratchpad_xy, 2*nshot_per_task);
	STARPU_ASSERT(res == CURAND_STATUS_SUCCESS);

	float *x = &scratchpad_xy[0];
	float *y = &scratchpad_xy[nshot_per_task];

	unsigned long *shot_cnt = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);
	pi_redux_cuda_kernel(x, y, nshot_per_task, shot_cnt);
}
#endif

static struct starpu_perfmodel pi_model =
{
	.type = STARPU_HISTORY_BASED,
	.size_base = size_base,
	.symbol = "monte_carlo_pi_scratch"
};

static struct starpu_codelet pi_cl =
{
	.cpu_funcs = {pi_func_cpu},
	.cpu_funcs_name = {"pi_func_cpu"},
#ifdef STARPU_HAVE_CURAND
	.cuda_funcs = {pi_func_cuda},
#endif
	.nbuffers = 2,
	.modes    = {STARPU_SCRATCH, STARPU_RW},
	.model = &pi_model
};

static struct starpu_perfmodel pi_model_redux =
{
	.type = STARPU_HISTORY_BASED,
	.size_base = size_base,
	.symbol = "monte_carlo_pi_scratch_redux"
};

static struct starpu_codelet pi_cl_redux =
{
	.cpu_funcs = {pi_func_cpu},
	.cpu_funcs_name = {"pi_func_cpu"},
#ifdef STARPU_HAVE_CURAND
	.cuda_funcs = {pi_func_cuda},
#endif
	.nbuffers = 2,
	.modes    = {STARPU_SCRATCH, STARPU_REDUX},
	.model = &pi_model_redux
};

/*
 *	Codelets to implement reduction
 */

void init_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
        unsigned long *val = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
        *val = 0;
}

#ifdef STARPU_HAVE_CURAND
static void init_cuda_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
        unsigned long *val = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
        cudaMemsetAsync(val, 0, sizeof(unsigned long), starpu_cuda_get_local_stream());
}
#endif

static struct starpu_codelet init_codelet =
{
        .cpu_funcs = {init_cpu_func},
        .cpu_funcs_name = {"init_cpu_func"},
#ifdef STARPU_HAVE_CURAND
        .cuda_funcs = {init_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = {STARPU_W},
        .nbuffers = 1
};

#ifdef STARPU_HAVE_CURAND
/* Dummy implementation of the addition of two unsigned longs in CUDA */
static void redux_cuda_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned long *d_a = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned long *d_b = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);

	unsigned long h_a, h_b;

	cudaMemcpyAsync(&h_a, d_a, sizeof(h_a), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaMemcpyAsync(&h_b, d_b, sizeof(h_b), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	h_a += h_b;

	cudaMemcpyAsync(d_a, &h_a, sizeof(h_a), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
}
#endif

void redux_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned long *a = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned long *b = (unsigned long *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*a = *a + *b;
}

static struct starpu_codelet redux_codelet =
{
	.cpu_funcs = {redux_cpu_func},
	.cpu_funcs_name = {"redux_cpu_func"},
#ifdef STARPU_HAVE_CURAND
	.cuda_funcs = {redux_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = {STARPU_RW, STARPU_R},
	.nbuffers = 2
};

/*
 *	Main program
 */

int main(int argc, char **argv)
{
	unsigned i;
	int ret;

	/* Not supported yet */
	if (starpu_get_env_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return 77;

	parse_args(argc, argv);

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Launch a Random Number Generator (RNG) on each worker */
	starpu_execute_on_each_worker(init_rng, NULL, STARPU_CPU|STARPU_CUDA);

	/* Create a scratchpad data */
	starpu_data_handle_t xy_scratchpad_handle;
	starpu_vector_data_register(&xy_scratchpad_handle, -1, (uintptr_t)NULL,
		2*nshot_per_task, sizeof(float));

	/* Create a variable that will be used to count the number of shots
	 * that actually hit the unit circle when shooting randomly in
	 * [-1,1]^2. */
	unsigned long shot_cnt = 0;
	starpu_data_handle_t shot_cnt_handle;
	starpu_variable_data_register(&shot_cnt_handle, STARPU_MAIN_RAM,
			(uintptr_t)&shot_cnt, sizeof(shot_cnt));

	starpu_data_set_reduction_methods(shot_cnt_handle,
					&redux_codelet, &init_codelet);

	double start;
	double end;

	for (i = 0; i < ntasks_warmup; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = use_redux?&pi_cl_redux:&pi_cl;

		task->handles[0] = xy_scratchpad_handle;
		task->handles[1] = shot_cnt_handle;

		ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}


	start = starpu_timing_now();

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = use_redux?&pi_cl_redux:&pi_cl;

		task->handles[0] = xy_scratchpad_handle;
		task->handles[1] = shot_cnt_handle;

		ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_data_unregister(shot_cnt_handle);
	starpu_data_unregister(xy_scratchpad_handle);

	end = starpu_timing_now();
	double timing = end - start;
	/* Total surface : Pi * r^ 2 = Pi*1^2, total square surface : 2^2 = 4,
	 * probability to impact the disk: pi/4 */
	unsigned long total = (ntasks + ntasks_warmup)*nshot_per_task;
	double pi_approx = ((double)shot_cnt*4.0)/total;

	FPRINTF(stderr, "Reductions? %s\n", use_redux?"yes":"no");
	FPRINTF(stderr, "Pi approximation : %f (%lu / %lu)\n", pi_approx, shot_cnt, total);
	FPRINTF(stderr, "Error %e \n", pi_approx - PI);
	FPRINTF(stderr, "Total time : %f ms\n", timing/1000.0);
	FPRINTF(stderr, "Speed : %f GShot/s\n", total/(1e3*timing));

	starpu_shutdown();

	if (fabs(pi_approx - PI) > 1.0)
		return 1;

	return 0;
}
