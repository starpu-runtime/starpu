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

#include "starpufft.h"

#define DIV_1D 64
#define DIV_2D 8

#define _FFTW_FLAGS FFTW_ESTIMATE

enum steps {
	START, SHUFFLED_1, FFT_1, SHUFFLED_2, FFT_2, SHUFFLED_3, END
};
#define STEP_TAG(plan, step, n) (((starpu_tag_t) plan->number << 60) | ((starpu_tag_t)(step) << 56) | (starpu_tag_t) n)

// TODO: Z2Z, D2Z, Z2D
enum type {
	R2C,
	C2R,
	C2C
};

static unsigned task_per_worker[STARPU_NMAXWORKERS];
static unsigned samples_per_worker[STARPU_NMAXWORKERS];
static struct timeval start, submit_tasks, do_tasks, tasks_done, gather, end;

/*
 *
 *	The actual kernels
 *
 */

struct STARPUFFT(plan) {
	int number;	/* uniquely identifies the plan, for starpu tags */

	int *n;
	int *n1;
	int *n2;
	int totsize;
	int totsize1;
	int totsize2;
	int dim;
	enum type type;
	int sign;

	STARPUFFT(complex) *roots[2];
	starpu_data_handle roots_handle[2];

	/* Synchronization for termination */
	unsigned todo;
	pthread_mutex_t mutex;
	pthread_cond_t cond;

	struct {
#ifdef USE_CUDA
		cufftHandle plan_cuda;
		_cufftComplex *gpu_in;
		STARPUFFT(complex) *local_in;
		int initialized;
#endif
#ifdef HAVE_FFTW
		_fftw_plan plan_cpu;
		_fftw_complex *in;
		_fftw_complex *out;
#endif
	} plans[STARPU_NMAXWORKERS];

#ifdef HAVE_FFTW
	_fftw_plan plan_gather;
#endif

	STARPUFFT(complex) *in;
	STARPUFFT(complex) *split_in, *split_out;
	STARPUFFT(complex) *output;

	starpu_data_handle *in_handle;
	starpu_data_handle *out_handle;
	struct starpu_task **tasks;
	struct STARPUFFT(args) *args;
};

struct STARPUFFT(args) {
	struct STARPUFFT(plan) *plan;
	int i, j, *iv;
};

#ifdef USE_CUDA

extern void STARPUFFT(cuda_1d_twiddle_host)(_cuComplex *out, _cuComplex *roots, unsigned n, unsigned i);

static void
STARPUFFT(dft_1d_kernel_gpu)(starpu_data_interface_t *descr, void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int j;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];
	cufftResult cures;

	_cufftComplex *in = (_cufftComplex *)descr[0].vector.ptr;
	_cufftComplex *out = (_cufftComplex *)descr[1].vector.ptr;
	_cufftComplex *roots = (_cufftComplex *)descr[2].vector.ptr;

	int workerid = starpu_get_worker_id();

	if (!plan->plans[workerid].initialized) {
		cures = cufftPlan1d(&plan->plans[workerid].plan_cuda, n2, _CUFFT_C2C, 1);
		STARPU_ASSERT(cures == CUFFT_SUCCESS);
		cures = cudaMalloc((void**) &plan->plans[workerid].gpu_in, n2 * sizeof(_cufftComplex));
		plan->plans[workerid].local_in = malloc(n2 * sizeof(_cufftComplex));
		STARPU_ASSERT(cures == CUDA_SUCCESS);
		plan->plans[workerid].initialized = 1;
	}

	for (j = 0; j < n2; j++)
		plan->plans[workerid].local_in[j] = plan->in[i + j*n1];
	cudaMemcpy(plan->plans[workerid].gpu_in, plan->plans[workerid].local_in, n2 * sizeof(_cufftComplex), cudaMemcpyHostToDevice);
	in = plan->plans[workerid].gpu_in;

	cures = _cufftExecC2C(plan->plans[workerid].plan_cuda, in, out, plan->sign == -1 ? CUFFT_FORWARD : CUFFT_INVERSE);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);

	STARPUFFT(cuda_1d_twiddle_host)(out, roots, n2, i);
}

extern void STARPUFFT(cuda_2d_twiddle_host)(_cuComplex *out, _cuComplex *roots0, _cuComplex *roots1, unsigned n2, unsigned m2, unsigned i, unsigned j);

static void
STARPUFFT(dft_2d_kernel_gpu)(starpu_data_interface_t *descr, void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int j = args->j;
	int k, l;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];
	int m1 = plan->n1[1];
	int m2 = plan->n2[1];
	int m = plan->n[1];
	cufftResult cures;

	_cufftComplex *in = (_cufftComplex *)descr[0].vector.ptr;
	_cufftComplex *out = (_cufftComplex *)descr[1].vector.ptr;
	_cufftComplex *roots0 = (_cufftComplex *)descr[2].vector.ptr;
	_cufftComplex *roots1 = (_cufftComplex *)descr[3].vector.ptr;

	int workerid = starpu_get_worker_id();

	if (!plan->plans[workerid].initialized) {
		cures = cufftPlan2d(&plan->plans[workerid].plan_cuda, n2, m2, _CUFFT_C2C);
		STARPU_ASSERT(cures == CUFFT_SUCCESS);
		cures = cudaMalloc((void**) &plan->plans[workerid].gpu_in, n2 * m2 * sizeof(_cufftComplex));
		plan->plans[workerid].local_in = malloc(n2 * m2 * sizeof(_cufftComplex));
		STARPU_ASSERT(cures == CUDA_SUCCESS);
		plan->plans[workerid].initialized = 1;
	}

	for (k = 0; k < n2; k++)
		for (l = 0; l < m2; l++)
			plan->plans[workerid].local_in[k*m2+l] = plan->in[i*m+j+k*m*n1+l*m1];
	cudaMemcpy(plan->plans[workerid].gpu_in, plan->plans[workerid].local_in, n2 * m2 * sizeof(_cufftComplex), cudaMemcpyHostToDevice);
	in = plan->plans[workerid].gpu_in;

	cures = _cufftExecC2C(plan->plans[workerid].plan_cuda, in, out, plan->sign == -1 ? CUFFT_FORWARD : CUFFT_INVERSE);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);

	STARPUFFT(cuda_2d_twiddle_host)(out, roots0, roots1, n2, m2, i, j);
}
#endif

#ifdef HAVE_FFTW
static void
STARPUFFT(dft_1d_kernel_cpu)(starpu_data_interface_t *descr, void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int j;
	int workerid = starpu_get_worker_id();
	int n1 = plan->n1[0], n2 = plan->n2[0];

	STARPUFFT(complex) *out = (STARPUFFT(complex) *)descr[1].vector.ptr;

	_fftw_complex *worker_out = (STARPUFFT(complex) *)plan->plans[workerid].out;

	for (j = 0; j < n2; j++)
		plan->plans[workerid].in[j] = plan->in[i + j*n1];
	_FFTW(execute)(plan->plans[workerid].plan_cpu);

	for (j = 0; j < n2; j++)
		out[j] = worker_out[j] * plan->roots[0][i*j];
}

static void
STARPUFFT(dft_2d_kernel_cpu)(starpu_data_interface_t *descr, void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int j = args->j;
	int k, l;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];
	int m1 = plan->n1[1];
	int m2 = plan->n2[1];
	int m = plan->n[1];
	int workerid = starpu_get_worker_id();

	STARPUFFT(complex) *out = (STARPUFFT(complex) *)descr[1].vector.ptr;

	_fftw_complex *worker_out = (STARPUFFT(complex) *)plan->plans[workerid].out;

	for (k = 0; k < n2; k++)
		for (l = 0; l < m2; l++)
			plan->plans[workerid].in[k*m2+l] = plan->in[i*m+j+k*m*n1+l*m1];
	_FFTW(execute)(plan->plans[workerid].plan_cpu);
	for (k = 0; k < n2; k++)
		for (l = 0; l < m2; l++)
			out[k*m2 + l] = worker_out[k*m2 + l] * plan->roots[0][i*k] * plan->roots[1][j*l];
}
#endif

struct starpu_perfmodel_t STARPUFFT(dft_1d_model) = {
	.type = HISTORY_BASED,
	.symbol = TYPE"dft_1d"
};

struct starpu_perfmodel_t STARPUFFT(dft_2d_model) = {
	.type = HISTORY_BASED,
	.symbol = TYPE"dft_2d"
};

static starpu_codelet STARPUFFT(dft_1d_codelet) = {
	.where =
#ifdef USE_CUDA
		CUBLAS|
#endif
#ifdef HAVE_FFTW
		CORE|
#endif
		0,
#ifdef USE_CUDA
	.cublas_func = STARPUFFT(dft_1d_kernel_gpu),
#endif
#ifdef HAVE_FFTW
	.core_func = STARPUFFT(dft_1d_kernel_cpu),
#endif
	.model = &STARPUFFT(dft_1d_model),
	.nbuffers = 3
};

static starpu_codelet STARPUFFT(dft_2d_codelet) = {
	.where =
#ifdef USE_CUDA
		CUBLAS|
#endif
#ifdef HAVE_FFTW
		CORE|
#endif
		0,
#ifdef USE_CUDA
	.cublas_func = STARPUFFT(dft_2d_kernel_gpu),
#endif
#ifdef HAVE_FFTW
	.core_func = STARPUFFT(dft_2d_kernel_cpu),
#endif
	.model = &STARPUFFT(dft_2d_model),
	.nbuffers = 4
};

void
STARPUFFT(callback)(void *_plan)
{
	STARPUFFT(plan) plan = _plan;

	int workerid = starpu_get_worker_id();

	/* do some accounting */
	task_per_worker[workerid]++;
	samples_per_worker[workerid] += plan->totsize2;

	if (STARPU_ATOMIC_ADD(&plan->todo, -1) == 0)
	{
		pthread_mutex_lock(&plan->mutex);
		pthread_cond_signal(&plan->cond);
		pthread_mutex_unlock(&plan->mutex);
	}
}

static void
check_dims(STARPUFFT(plan) plan)
{
	int dim;
	for (dim = 0; dim < plan->dim; dim++)
		if (plan->n[dim] & (plan->n[dim]-1)) {
			fprintf(stderr,"can't cope with non-power-of-2\n");
			STARPU_ASSERT(0);
		}
}

static void
compute_roots(STARPUFFT(plan) plan)
{
	int dim, k;

	/* Compute the n-roots and m-roots of unity for twiddling */
	for (dim = 0; dim < plan->dim; dim++) {
		STARPUFFT(complex) exp = (plan->sign * 2. * 4.*atan(1.)) * _Complex_I / (STARPUFFT(complex)) plan->n[dim];
		plan->roots[dim] = malloc(plan->n[dim] * sizeof(**plan->roots));
		for (k = 0; k < plan->n[dim]; k++)
			plan->roots[dim][k] = cexp(exp*k);
		starpu_register_vector_data(&plan->roots_handle[dim], 0, (uintptr_t) plan->roots[dim], plan->n[dim], sizeof(**plan->roots));
	}
}

STARPUFFT(plan)
STARPUFFT(plan_dft_1d)(int n, int sign, unsigned flags)
{
	int workerid;
	int n1 = DIV_1D;
	int n2 = n / n1;
	int i;
	struct starpu_task *task;

#ifdef USE_CUDA
	/* cufft 1D limited to 8M elements */
	while (n2 > 8 << 20) {
		n1 *= 2;
		n2 /= 2;
	}
#endif
	STARPU_ASSERT(n == n1*n2);

	/* TODO: flags? Automatically set FFTW_MEASURE on calibration? */
	STARPU_ASSERT(flags == 0);

	STARPUFFT(plan) plan = malloc(sizeof(*plan));
	memset(plan, 0, sizeof(*plan));

	plan->number = STARPU_ATOMIC_ADD(&starpufft_last_plan_number, 1) - 1;

	/* 4bit limitation in the tag space */
	STARPU_ASSERT(plan->number < 16);

	plan->dim = 1;
	plan->n = malloc(plan->dim * sizeof(*plan->n));
	plan->n[0] = n;

	check_dims(plan);

	plan->n1 = malloc(plan->dim * sizeof(*plan->n1));
	plan->n1[0] = n1;
	plan->n2 = malloc(plan->dim * sizeof(*plan->n2));
	plan->n2[0] = n2;
	plan->totsize = n;
	plan->totsize1 = n1;
	plan->totsize2 = n2;
	plan->type = C2C;
	plan->sign = sign;

	compute_roots(plan);

	pthread_mutex_init(&plan->mutex, NULL);
	pthread_cond_init(&plan->cond, NULL);

	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			plan->plans[workerid].in = _FFTW(malloc)(n2 * sizeof(_fftw_complex));
			plan->plans[workerid].out = _FFTW(malloc)(n2 * sizeof(_fftw_complex));
			plan->plans[workerid].plan_cpu = _FFTW(plan_dft_1d)(n2, plan->plans[workerid].in, plan->plans[workerid].out, sign, _FFTW_FLAGS);
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

	plan->split_in = STARPUFFT(malloc)(plan->totsize * sizeof(STARPUFFT(complex)));
	plan->split_out = STARPUFFT(malloc)(plan->totsize * sizeof(STARPUFFT(complex)));
	plan->output = STARPUFFT(malloc)(plan->totsize * sizeof(STARPUFFT(complex)));

#ifdef HAVE_FFTW
	plan->plan_gather = _FFTW(plan_many_dft)(plan->dim, plan->n1, plan->totsize2,
			/* input */ plan->split_out, NULL, n2, 1,
			/* output */ plan->output, NULL, n2, 1,
			sign, _FFTW_FLAGS);
	STARPU_ASSERT(plan->plan_gather);
#else
#warning libstarpufft can not work correctly without libfftw3
#endif

	plan->in_handle = malloc(plan->totsize1 * sizeof(*plan->in_handle));
	plan->out_handle = malloc(plan->totsize1 * sizeof(*plan->out_handle));
	plan->tasks = malloc(plan->totsize1 * sizeof(*plan->tasks));
	plan->args = malloc(plan->totsize1 * sizeof(*plan->args));
	for (i = 0; i < plan->totsize1; i++) {
		/* Register data */
		starpu_register_vector_data(&plan->in_handle[i], 0, (uintptr_t) &plan->split_in[i*plan->totsize2], plan->totsize2, sizeof(*plan->split_in));
		starpu_register_vector_data(&plan->out_handle[i], 0, (uintptr_t) &plan->split_out[i*plan->totsize2], plan->totsize2, sizeof(*plan->split_out));

		/* We'll need it on the CPU only anyway */
		starpu_data_set_wb_mask(plan->out_handle[i], 1<<0);

		/* Create task */
		plan->tasks[i] = task = starpu_task_create();
		task->cl = &STARPUFFT(dft_1d_codelet;)
		task->buffers[0].handle = plan->in_handle[i];
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = plan->out_handle[i];
		task->buffers[1].mode = STARPU_W;
		task->buffers[2].handle = plan->roots_handle[0];
		task->buffers[2].mode = STARPU_R;
		plan->args[i].plan = plan;
		plan->args[i].i = i;
		task->cl_arg = &plan->args[i];
		task->callback_func = STARPUFFT(callback);
		task->callback_arg = plan;

		task->cleanup = 0;
	}

	return plan;
}

STARPUFFT(plan)
STARPUFFT(plan_dft_2d)(int n, int m, int sign, unsigned flags)
{
	int workerid;
	int n1 = DIV_2D;
	int n2 = n / n1;
	int m1 = DIV_2D;
	int m2 = m / m1;
	int i;
	struct starpu_task *task;

#ifdef USE_CUDA
	/* cufft 2D-3D limited to [2,16384] */
	while (n2 > 16384) {
		n1 *= 2;
		n2 /= 2;
	}
#endif
	STARPU_ASSERT(n == n1*n2);

#ifdef USE_CUDA
	/* cufft 2D-3D limited to [2,16384] */
	while (m2 > 16384) {
		m1 *= 2;
		m2 /= 2;
	}
#endif
	STARPU_ASSERT(m == m1*m2);

	/* TODO: flags? Automatically set FFTW_MEASURE on calibration? */
	STARPU_ASSERT(flags == 0);

	STARPUFFT(plan) plan = malloc(sizeof(*plan));
	memset(plan, 0, sizeof(*plan));

	plan->dim = 2;
	plan->n = malloc(plan->dim * sizeof(*plan->n));
	plan->n[0] = n;
	plan->n[1] = m;

	check_dims(plan);

	plan->n1 = malloc(plan->dim * sizeof(*plan->n1));
	plan->n1[0] = n1;
	plan->n1[1] = m1;
	plan->n2 = malloc(plan->dim * sizeof(*plan->n2));
	plan->n2[0] = n2;
	plan->n2[1] = m2;
	plan->totsize = n * m;
	plan->totsize1 = n1 * m1;
	plan->totsize2 = n2 * m2;
	plan->type = C2C;
	plan->sign = sign;

	compute_roots(plan);

	pthread_mutex_init(&plan->mutex, NULL);
	pthread_cond_init(&plan->cond, NULL);

	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			plan->plans[workerid].in = _FFTW(malloc)(plan->totsize2 * sizeof(_fftw_complex));
			plan->plans[workerid].out = _FFTW(malloc)(plan->totsize2 * sizeof(_fftw_complex));
			plan->plans[workerid].plan_cpu = _FFTW(plan_dft_2d)(n2, m2, plan->plans[workerid].in, plan->plans[workerid].out, sign, _FFTW_FLAGS);
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

	plan->split_in = STARPUFFT(malloc)(plan->totsize * sizeof(STARPUFFT(complex)));
	plan->split_out = STARPUFFT(malloc)(plan->totsize * sizeof(STARPUFFT(complex)));
	plan->output = STARPUFFT(malloc)(plan->totsize * sizeof(STARPUFFT(complex)));

#ifdef HAVE_FFTW
	plan->plan_gather = _FFTW(plan_many_dft)(plan->dim, plan->n1, plan->totsize2,
			/* input */ plan->split_out, 0, n2*m2, 1,
			/* output */ plan->output, 0, n2*m2, 1,
			sign, _FFTW_FLAGS);
	STARPU_ASSERT(plan->plan_gather);
#else
#warning libstarpufft can not work correctly without libfftw3
#endif

	plan->in_handle = malloc(plan->totsize1 * sizeof(*plan->in_handle));
	plan->out_handle = malloc(plan->totsize1 * sizeof(*plan->out_handle));
	plan->tasks = malloc(plan->totsize1 * sizeof(*plan->tasks));
	plan->args = malloc(plan->totsize1 * sizeof(*plan->args));
	for (i = 0; i < plan->totsize1; i++) {
		/* Register data */
		starpu_register_vector_data(&plan->in_handle[i], 0, (uintptr_t) &plan->split_in[i*plan->totsize2], plan->totsize2, sizeof(*plan->split_in));
		starpu_register_vector_data(&plan->out_handle[i], 0, (uintptr_t) &plan->split_out[i*plan->totsize2], plan->totsize2, sizeof(*plan->split_out));

		/* We'll need it on the CPU only anyway */
		starpu_data_set_wb_mask(plan->out_handle[i], 1<<0);

		/* Create task */
		plan->tasks[i] = task = starpu_task_create();
		task->cl = &STARPUFFT(dft_2d_codelet;)
		task->buffers[0].handle = plan->in_handle[i];
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = plan->out_handle[i];
		task->buffers[1].mode = STARPU_W;
		task->buffers[2].handle = plan->roots_handle[0];
		task->buffers[2].mode = STARPU_R;
		task->buffers[3].handle = plan->roots_handle[1];
		task->buffers[3].mode = STARPU_R;
		plan->args[i].plan = plan;
		plan->args[i].i = i/m1;
		plan->args[i].j = i%m1;
		task->cl_arg = &plan->args[i];
		task->callback_func = STARPUFFT(callback);
		task->callback_arg = plan;

		task->cleanup = 0;
	}

	return plan;
}

void
STARPUFFT(execute)(STARPUFFT(plan) plan, void *_in, void *_out)
{
	gettimeofday(&start, NULL);
	memset(task_per_worker, 0, sizeof(task_per_worker));
	memset(samples_per_worker, 0, sizeof(task_per_worker));

	plan->in = _in;

	switch (plan->dim) {
		case 1: {
			switch (plan->type) {
			case C2C: {
				STARPUFFT(complex) *out = _out;
				starpu_data_handle *out_handle = plan->out_handle;
				struct starpu_task **tasks = plan->tasks;
				int i;

				pthread_mutex_lock(&plan->mutex);
				plan->todo = plan->totsize1;
				pthread_mutex_unlock(&plan->mutex);

				for (i=0; i < plan->totsize1; i++)
					starpu_submit_task(tasks[i]);

				gettimeofday(&submit_tasks, NULL);
				/* Wait for tasks */
				pthread_mutex_lock(&plan->mutex);
				while (plan->todo != 0)
					pthread_cond_wait(&plan->cond, &plan->mutex);
				pthread_mutex_unlock(&plan->mutex);
				gettimeofday(&do_tasks, NULL);

				for (i = 0; i < plan->totsize1; i++)
					/* Make sure output is here? */
					starpu_sync_data_with_mem(out_handle[i]);

				gettimeofday(&tasks_done, NULL);

#ifdef HAVE_FFTW
				/* Perform n2 n1-ffts */
				_FFTW(execute)(plan->plan_gather);
				gettimeofday(&gather, NULL);
#endif
				memcpy(out, plan->output, plan->totsize * sizeof(*out));
				break;
			}
			default:
				STARPU_ASSERT(0);
				break;
			}
			break;
		}
		case 2: {
			STARPU_ASSERT(plan->type == C2C);
			STARPUFFT(complex) *out = _out;
			STARPUFFT(complex) *output = plan->output;
			int n1 = plan->n1[0], n2 = plan->n2[0];
			int m1 = plan->n1[1], m2 = plan->n2[1];
			starpu_data_handle *out_handle = plan->out_handle;
			struct starpu_task **tasks = plan->tasks;
			int i,j,k,l;

			pthread_mutex_lock(&plan->mutex);
			plan->todo = plan->totsize1;
			pthread_mutex_unlock(&plan->mutex);

			for (i=0; i < plan->totsize1; i++)
				starpu_submit_task(tasks[i]);

			gettimeofday(&submit_tasks, NULL);
			/* Wait for tasks */
			pthread_mutex_lock(&plan->mutex);
			while (plan->todo != 0)
				pthread_cond_wait(&plan->cond, &plan->mutex);
			pthread_mutex_unlock(&plan->mutex);
			gettimeofday(&do_tasks, NULL);

			/* Unregister data */
			for (i = 0; i < plan->totsize1; i++)
				/* Make sure output is here? */
				starpu_sync_data_with_mem(out_handle[i]);
			gettimeofday(&tasks_done, NULL);

#ifdef HAVE_FFTW
			/* Perform n2*m2 n1*m1-ffts */
			_FFTW(execute)(plan->plan_gather);
#endif
			gettimeofday(&gather, NULL);

			for (i = 0; i < n1; i++)
				for (j = 0; j < m1; j++)
					for (k = 0; k < n2; k++)
						for (l = 0; l < m2; l++)
							out[i*m1*n2*m2+j*m2+k*m2*m1+l] = output[i*m1*n2*m2+j*n2*m2+k*m2+l];

			break;
		}
		default:
			STARPU_ASSERT(0);
			break;
	}

	gettimeofday(&end, NULL);
}

void
STARPUFFT(destroy_plan)(STARPUFFT(plan) plan)
{
	int workerid, dim, i;

	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			_FFTW(free)(plan->plans[workerid].in);
			_FFTW(free)(plan->plans[workerid].out);
			_FFTW(destroy_plan)(plan->plans[workerid].plan_cpu);
#endif
			break;
		case STARPU_CUDA_WORKER:
#ifdef USE_CUDA
			/* FIXME: Can't deallocate */
#endif
			break;
		default:
			STARPU_ASSERT(0);
			break;
		}
	}
	for (i = 0; i < plan->totsize1; i++) {
		starpu_delete_data(plan->in_handle[i]);
		starpu_delete_data(plan->out_handle[i]);
	}
	free(plan->in_handle);
	free(plan->out_handle);
	free(plan->tasks);
	free(plan->args);
	for (dim = 0; dim < plan->dim; dim++) {
		starpu_delete_data(plan->roots_handle[dim]);
		free(plan->roots[dim]);
	}
	free(plan->n);
	free(plan->n1);
	free(plan->n2);
	STARPUFFT(free)(plan->split_in);
	STARPUFFT(free)(plan->split_out);
	STARPUFFT(free)(plan->output);
#ifdef HAVE_FFTW
	_FFTW(destroy_plan)(plan->plan_gather);
#endif
	free(plan);
}

void *
STARPUFFT(malloc)(size_t n)
{
#ifdef USE_CUDA
	void *res;
	starpu_malloc_pinned_if_possible(&res, n);
	return res;
#else
#  ifdef HAVE_FFTW
	return _FFTW(malloc)(n);
#  else
	return malloc(n);
#  endif
#endif
}

void
STARPUFFT(free)(void *p)
{
#ifdef USE_CUDA
	// TODO: FIXME
#else
#  ifdef HAVE_FFTW
	_FFTW(free)(p);
#  else
	free(p);
#  endif
#endif
}

void
STARPUFFT(showstats)(FILE *out)
{
	int worker;
	unsigned total;

#define TIMING(begin,end) (double)((end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec))
#define MSTIMING(begin,end) (TIMING(begin,end)/1000.)
	double paratiming = TIMING(start,do_tasks);
	fprintf(out, "Tasks submission took %2.2f ms\n", MSTIMING(start,submit_tasks));
	fprintf(out, "Tasks termination took %2.2f ms\n", MSTIMING(submit_tasks,do_tasks));
	fprintf(out, "Tasks cleanup took %2.2f ms\n", MSTIMING(do_tasks,tasks_done));
	fprintf(out, "Gather took %2.2f ms\n", MSTIMING(tasks_done,gather));
	fprintf(out, "Finalization took %2.2f ms\n", MSTIMING(gather,end));

	fprintf(out, "Total %2.2f ms\n", MSTIMING(start,end));

	for (worker = 0, total = 0; worker < STARPU_NMAXWORKERS; worker++)
		total += task_per_worker[worker];

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (task_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);

			unsigned long bytes = sizeof(STARPUFFT(complex))*samples_per_worker[worker];

			fprintf(stderr, "\t%s -> %2.2f MB\t%2.2f\tMB/s\t%u %2.2f %%\n", name, (1.0*bytes)/(1024*1024), bytes/paratiming, task_per_worker[worker], (100.0*task_per_worker[worker])/total);
		}
	}
}
