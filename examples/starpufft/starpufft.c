/*
 * StarPU
 * Copyright (C) INRIA 2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR in PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include <starpu.h>

#include <starpu_config.h>
#include "starpufft.h"

#ifdef HAVE_FFTW
#include <fftw3.h>
#endif
#ifdef USE_CUDA
#include <cufft.h>
#endif

enum type {
	R2C,
	C2R,
	C2C
};

static unsigned task_per_worker[STARPU_NMAXWORKERS];
static unsigned samples_per_worker[STARPU_NMAXWORKERS];
static struct timeval start, middle, end;

/*
 *
 *	The actual kernels
 *
 */

/* we don't reinitialize the FFT plan for every kernel, so we "cache" it */
struct starpufftf_plan {
	int *n;
	int *n1;
	int *n2;
	int dim;
	enum type type;
	int sign;

	starpufftf_complex *roots;

	/* Synchronization for termination */
	unsigned todo;
	pthread_mutex_t mutex;
	pthread_cond_t cond;

	struct {
#ifdef USE_CUDA
		cufftHandle plan_cuda;
		int initialized;
#endif
#ifdef HAVE_FFTW
		fftwf_plan plan_cpu;
		void *in;
		void *out;
#endif
	} plans[STARPU_NMAXWORKERS];

#ifdef HAVE_FFTW
	fftwf_plan plan_gather;
#endif

	void *split_in, *split_out;
	void *output;
};

#ifdef USE_CUDA
static void
dft_1d_kernel_gpu(starpu_data_interface_t *descr, void *_plan)
{
	starpufftf_plan plan = _plan;
	cufftResult cures;

	starpufftf_complex *in = (starpufftf_complex *)descr[0].vector.ptr;
	starpufftf_complex *out = (starpufftf_complex *)descr[1].vector.ptr;

	int workerid = starpu_get_worker_id();

	if (!plan->plans[workerid].initialized) {
		cufftResult cures;
		cures = cufftPlan1d(&plan->plans[workerid].plan_cuda, plan->n2[0], CUFFT_C2C, 1);
		plan->plans[workerid].initialized = 1;
		STARPU_ASSERT(cures == CUFFT_SUCCESS);
	}

	/* May be in-place */
	cures = cufftExecC2C(plan->plans[workerid].plan_cuda, in, out, plan->sign == -1 ? CUFFT_FORWARD : CUFFT_INVERSE);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);
}

static void
dft_r2c_1d_kernel_gpu(starpu_data_interface_t *descr, void *_plan)
{
	starpufftf_plan plan = _plan;
	cufftResult cures;

	float *in = (float *)descr[0].vector.ptr;
	starpufftf_complex *out = (starpufftf_complex *)descr[1].vector.ptr;

	int workerid = starpu_get_worker_id();

	/* May be in-place */
	cures = cufftExecR2C(plan->plans[workerid].plan_cuda, in, (cufftComplex*) out);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);
}

static void
dft_c2r_1d_kernel_gpu(starpu_data_interface_t *descr, void *_plan)
{
	starpufftf_plan plan = _plan;
	cufftResult cures;

	starpufftf_complex *in = (starpufftf_complex *)descr[0].vector.ptr;
	float *out = (float *)descr[1].vector.ptr;

	int workerid = starpu_get_worker_id();

	/* May be in-place */
	cures = cufftExecC2R(plan->plans[workerid].plan_cuda, (cufftComplex*) in, out);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);
}
#endif

#ifdef HAVE_FFTW
static void
dft_1d_kernel_cpu(starpu_data_interface_t *descr, void *_plan)
{
	starpufftf_plan plan = _plan;

	starpufftf_complex *in = (starpufftf_complex *)descr[0].vector.ptr;
	starpufftf_complex *out = (starpufftf_complex *)descr[1].vector.ptr;

	int workerid = starpu_get_worker_id();
	
	memcpy(plan->plans[workerid].in, in, plan->n2[0]*sizeof(starpufftf_complex));
	fftwf_execute(plan->plans[workerid].plan_cpu);
	memcpy(out, plan->plans[workerid].out, plan->n2[0]*sizeof(starpufftf_complex));
}
#endif

struct starpu_perfmodel_t dft_1d_model = {
	.type = HISTORY_BASED,
	.symbol = "dft_1d"
};

struct starpu_perfmodel_t dft_r2c_1d_model = {
	.type = HISTORY_BASED,
	.symbol = "dft_r2c_1d"
};

struct starpu_perfmodel_t dft_c2r_1d_model = {
	.type = HISTORY_BASED,
	.symbol = "dft_c2r_1d"
};

static starpu_codelet dft_1d_codelet = {
	.where =
#ifdef USE_CUDA
		CUBLAS|
#endif
#ifdef HAVE_FFTW
		CORE|
#endif
		0,
#ifdef USE_CUDA
	.cublas_func = dft_1d_kernel_gpu,
#endif
#ifdef HAVE_FFTW
	.core_func = dft_1d_kernel_cpu,
#endif
	.model = &dft_1d_model,
	.nbuffers = 2
};

static starpu_codelet dft_r2c_1d_codelet = {
	.where =
#ifdef USE_CUDA
		CUBLAS|
#endif
#ifdef HAVE_FFTW
		CORE|
#endif
		0,
#ifdef USE_CUDA
	.cublas_func = dft_r2c_1d_kernel_gpu,
#endif
#ifdef HAVE_FFTW
	.core_func = dft_1d_kernel_cpu,
#endif
	.model = &dft_r2c_1d_model,
	.nbuffers = 2
};

static starpu_codelet dft_c2r_1d_codelet = {
	.where =
#ifdef USE_CUDA
		CUBLAS|
#endif
#ifdef HAVE_FFTW
		CORE|
#endif
		0,
#ifdef USE_CUDA
	.cublas_func = dft_c2r_1d_kernel_gpu,
#endif
#ifdef HAVE_FFTW
	.core_func = dft_1d_kernel_cpu,
#endif
	.model = &dft_c2r_1d_model,
	.nbuffers = 2
};

void
callback(void *_plan)
{
	starpufftf_plan plan = _plan;

	int workerid = starpu_get_worker_id();

	/* do some accounting */
	task_per_worker[workerid]++;
	samples_per_worker[workerid] += plan->n2[0];

	if (STARPU_ATOMIC_ADD(&plan->todo, -1) == 0)
	{
		pthread_mutex_lock(&plan->mutex);
		pthread_cond_signal(&plan->cond);
		pthread_mutex_unlock(&plan->mutex);
	}
}

starpufftf_plan
starpufftf_plan_dft_1d(int n, int sign, unsigned flags)
{
	int workerid;
	int n1 = 128;
	int n2 = n / n1;
	int i;

	if (n%n1) {
		fprintf(stderr,"can't cope with non-power-of-2 n\n");
		STARPU_ASSERT(0);
	}
	/* TODO: flags? Automatically set FFTW_MEASURE on calibration? */
	STARPU_ASSERT(flags == 0);

	starpufftf_plan plan = malloc(sizeof(*plan));
	memset(plan, 0, sizeof(*plan));

	plan->dim = 1;
	plan->n = malloc(plan->dim * sizeof(*plan->n));
	plan->n[0] = n;
	plan->n1 = malloc(plan->dim * sizeof(*plan->n1));
	plan->n1[0] = n1;
	plan->n2 = malloc(plan->dim * sizeof(*plan->n2));
	plan->n2[0] = n2;
	plan->type = C2C;
	plan->sign = sign;

	/* Compute n1 n-roots of unity for twiddling */
	starpufftf_complex exp = (sign * 2. * 4.*atan(1.)) * _Complex_I / (starpufftf_complex) n, curroot = 1;
	plan->roots = malloc(n * sizeof(*plan->roots));
	for (i = 0; i < n; i++)
		plan->roots[i] = cexp(exp*i);

	pthread_mutex_init(&plan->mutex, NULL);
	pthread_cond_init(&plan->cond, NULL);

	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			plan->plans[workerid].in = fftwf_malloc(n2 * sizeof(fftwf_complex));
			plan->plans[workerid].out = fftwf_malloc(n2 * sizeof(fftwf_complex));
			plan->plans[workerid].plan_cpu = fftwf_plan_dft_1d(n2, plan->plans[workerid].in, plan->plans[workerid].out, sign, FFTW_ESTIMATE);
			STARPU_ASSERT(plan->plans[workerid].plan_cpu);
#endif
			break;
		case STARPU_CUDA_WORKER:
#ifdef USE_CUDA
			plan->plans[workerid].initialized = 0;
#endif
			break;
		default:
			STARPU_ASSERT(0);
			break;
		}
	}

	plan->split_in = starpufftf_malloc(n * sizeof(starpufftf_complex));
	plan->split_out = starpufftf_malloc(n * sizeof(starpufftf_complex));
	plan->output = starpufftf_malloc(n * sizeof(starpufftf_complex));

#ifdef HAVE_FFTW
	plan->plan_gather = fftwf_plan_many_dft(1, &n1, n2,
			/* input */ plan->split_out, NULL, plan->n2[0], 1,
			/* output */ plan->output, NULL, plan->n2[0], 1,
			sign, FFTW_ESTIMATE);
	STARPU_ASSERT(plan->plan_gather);
#else
#warning libstarpufft can not work correctly without libfftw3
#endif

	return plan;
}

void
starpufftf_execute(starpufftf_plan plan, void *_in, void *_out)
{
	memset(task_per_worker, 0, sizeof(task_per_worker));
	memset(samples_per_worker, 0, sizeof(task_per_worker));
	gettimeofday(&start, NULL);

	switch (plan->dim) {
		case 1: {

#define WORK1D(in_type, out_type, codelet) \
	do { \
	} while(0)

			switch (plan->type) {
			case C2C: {
				starpufftf_complex *in = _in;
				starpufftf_complex *out = _out;
				starpufftf_complex *split_in = plan->split_in;
				starpufftf_complex *split_out = plan->split_out;
				int n1 = plan->n1[0], n2 = plan->n2[0], n = plan->n[0];
				starpu_data_handle in_handle[n1];
				starpu_data_handle out_handle[n1];
				struct starpu_task *tasks[n1];
				struct starpu_task *task;
				int i,j;

				plan->todo = n1;

				for (i = 0; i < n1; i++)
					for (j = 0; j < n2; j++)
						split_in[i*n2 + j] = in[j*n1 + i];

				for (i=0; i < n1; i++) {
					/* Register data */
					starpu_register_vector_data(&in_handle[i], 0, (uintptr_t) &split_in[i*n2], n2, sizeof(*split_in));
					starpu_register_vector_data(&out_handle[i], 0, (uintptr_t) &split_out[i*n2], n2, sizeof(*split_out));

					/* We'll need it on the CPU only anyway */
					starpu_data_set_wb_mask(out_handle[i], 1<<0);

					/* Create task */
					tasks[i] = task = starpu_task_create();
					task->cl = &dft_1d_codelet;
					task->buffers[0].handle = in_handle[i];
					task->buffers[1].handle = out_handle[i];
					task->cl_arg = plan;
					task->callback_func = callback;
					task->callback_arg = plan;
					starpu_submit_task(task);
				}
				/* Wait for tasks */
				pthread_mutex_lock(&plan->mutex);
				while (plan->todo != 0)
					pthread_cond_wait(&plan->cond, &plan->mutex);
				pthread_mutex_unlock(&plan->mutex);

				/* Unregister data */
				for (i = 0; i < n1; i++) {
					/* Make sure output is here? */
					starpu_sync_data_with_mem(out_handle[i]);
					starpu_delete_data(in_handle[i]);
					starpu_delete_data(out_handle[i]);
				}

				gettimeofday(&middle, NULL);

				/* Twiddle values */
				for (i = 0; i < n1; i++)
					for (j = 0; j < n2; j++)
						split_out[i*n2+j] *= plan->roots[i*j];
#ifdef HAVE_FFTW
				/* Perform n2 n1-ffts */
				fftwf_execute(plan->plan_gather);
				memcpy(out, plan->output, n * sizeof(*out));
#endif
				break;
			}
			default:
				STARPU_ASSERT(0);
				break;
			}
			break;
		}
		default:
			STARPU_ASSERT(0);
			break;
	}

	gettimeofday(&end, NULL);
}

void
starpufftf_destroy_plan(starpufftf_plan plan)
{
	int workerid;

	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			fftwf_free(plan->plans[workerid].in);
			fftwf_free(plan->plans[workerid].out);
			fftwf_destroy_plan(plan->plans[workerid].plan_cpu);
#endif
			break;
		case STARPU_CUDA_WORKER:
			break;
		default:
			STARPU_ASSERT(0);
			break;
		}
	}
	free(plan);
}

void *
starpufftf_malloc(size_t n)
{
#ifdef HAVE_FFTW
	return fftwf_malloc(n);
#else
	return malloc(n);
#endif
}

void
starpufftf_free(void *p)
{
#ifdef HAVE_FFTW
	fftwf_free(p);
#else
	free(p);
#endif
}

void
starpufftf_showstats(FILE *out)
{
	int worker;
	unsigned total;

	double paratiming = (double)((middle.tv_sec - start.tv_sec)*1000000 + (middle.tv_usec - start.tv_usec));
	double gathertiming = (double)((end.tv_sec - middle.tv_sec)*1000000 + (end.tv_usec - middle.tv_usec));
	double timing = paratiming + gathertiming;
	fprintf(out, "Fully parallel computation took %2.2f ms\n", paratiming/1000);
	fprintf(out, "Gather computation took %2.2f ms\n", gathertiming/1000);
	fprintf(out, "Total %2.2f ms\n", timing/1000);

	for (worker = 0, total = 0; worker < STARPU_NMAXWORKERS; worker++)
		total += task_per_worker[worker];

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (task_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);

			unsigned long bytes = sizeof(float)*samples_per_worker[worker];

			fprintf(stderr, "\t%s -> %2.2f MB\t%2.2f\tMB/s\t%u %2.2f %%\n", name, (1.0*bytes)/(1024*1024), bytes/paratiming, task_per_worker[worker], (100.0*task_per_worker[worker])/total);
		}
	}
}
