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

#define DIV_2D 8

#define STEP_TAG_2D(plan, step, i, j) (((starpu_tag_t) plan->number << 60) | ((starpu_tag_t)(step) << 56) | ((starpu_tag_t) i << 28) | (starpu_tag_t) j)

#ifdef USE_CUDA

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

struct starpu_perfmodel_t STARPUFFT(dft_2d_model) = {
	.type = HISTORY_BASED,
	.symbol = TYPE"dft_2d"
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
STARPUFFT(execute2dC2C)(STARPUFFT(plan) plan, void *_in, void *_out)
{
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
}
