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
 *
 * Sequential version
 *
 */

#ifdef __STARPU_USE_CUDA
/* Perform one fft of size n,m */
static void
STARPUFFT(fft_3d_plan_gpu)(void *args)
{
	STARPUFFT(plan) plan = args;
	cufftResult cures;
	int n = plan->n[0];
	int m = plan->n[1];
	int p = plan->n[2];
	int workerid = starpu_worker_get_id_check();

	cures = cufftPlan3d(&plan->plans[workerid].plan_cuda, n, m, p, _CUFFT_C2C);
	if (cures != CUFFT_SUCCESS)
		STARPU_CUFFT_REPORT_ERROR(cures);
	cufftSetStream(plan->plans[workerid].plan_cuda, starpu_cuda_get_local_stream());
	if (cures != CUFFT_SUCCESS)
		STARPU_CUFFT_REPORT_ERROR(cures);
}

static void
STARPUFFT(fft_3d_kernel_gpu)(void *descr[], void *args)
{
	STARPUFFT(plan) plan = args;
	cufftResult cures;

	_cufftComplex * restrict in = (_cufftComplex *)STARPU_VECTOR_GET_PTR(descr[0]);
	_cufftComplex * restrict out = (_cufftComplex *)STARPU_VECTOR_GET_PTR(descr[1]);

	int workerid = starpu_worker_get_id_check();

	task_per_worker[workerid]++;

	cures = _cufftExecC2C(plan->plans[workerid].plan_cuda, in, out, plan->sign == -1 ? CUFFT_FORWARD : CUFFT_INVERSE);
	if (cures != CUFFT_SUCCESS)
		STARPU_CUFFT_REPORT_ERROR(cures);

	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_HAVE_FFTW
/* Perform one fft of size n,m */
static void
STARPUFFT(fft_3d_kernel_cpu)(void *descr[], void *_args)
{
	STARPUFFT(plan) plan = _args;
	int workerid = starpu_worker_get_id_check();

	task_per_worker[workerid]++;

	STARPUFFT(complex) * restrict in = (STARPUFFT(complex) *)STARPU_VECTOR_GET_PTR(descr[0]);
	STARPUFFT(complex) * restrict out = (STARPUFFT(complex) *)STARPU_VECTOR_GET_PTR(descr[1]);

	_FFTW(execute_dft)(plan->plans[workerid].plan_cpu, in, out);
}
#endif

static struct starpu_perfmodel STARPUFFT(fft_3d_model) = {
	.type = STARPU_HISTORY_BASED,
	.symbol = TYPE"fft_3d"
};

static struct starpu_codelet STARPUFFT(fft_3d_codelet) = {
	.where =
#ifdef __STARPU_USE_CUDA
		STARPU_CUDA|
#endif
#ifdef STARPU_HAVE_FFTW
		STARPU_CPU|
#endif
		0,
#ifdef __STARPU_USE_CUDA
	.cuda_funcs = {STARPUFFT(fft_3d_kernel_gpu)},
#endif
#ifdef STARPU_HAVE_FFTW
	.cpu_funcs = {STARPUFFT(fft_3d_kernel_cpu)},
#endif
	CAN_EXECUTE
	.model = &STARPUFFT(fft_3d_model),
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_W},
	.name = "fft_3d_codelet"
};

STARPUFFT(plan)
STARPUFFT(plan_dft_3d)(int n, int m, int p, int sign, unsigned flags)
{
	unsigned workerid;

if (PARALLEL) {
	/* TODO */
	STARPU_ASSERT(0);
}

	/* TODO: flags? Automatically set FFTW_MEASURE on calibration? */
	STARPU_ASSERT(flags == 0);

	STARPUFFT(plan) plan = malloc(sizeof(*plan));
	memset(plan, 0, sizeof(*plan));

	plan->dim = 3;
	plan->n = malloc(plan->dim * sizeof(*plan->n));
	plan->n[0] = n;
	plan->n[1] = m;
	plan->n[2] = p;

	plan->totsize = n * m;

	plan->type = C2C;
	plan->sign = sign;


	/* Initialize per-worker working set */
	for (workerid = 0; workerid < starpu_worker_get_count(); workerid++) {
		switch (starpu_worker_get_type(workerid)) {
		case STARPU_CPU_WORKER:
#ifdef STARPU_HAVE_FFTW
			/* fft plan: one fft of size n, m. */
			plan->plans[workerid].plan_cpu = _FFTW(plan_dft_3d)(n, m, p, NULL, (void*) 1, sign, _FFTW_FLAGS);
			STARPU_ASSERT(plan->plans[workerid].plan_cpu);
#else
/* #warning libstarpufft can not work correctly if libfftw3 is not installed */
#endif
			break;
		case STARPU_CUDA_WORKER:
			break;
		default:
			/* Do not care, we won't be executing anything there. */
			break;
		}
	}
#ifdef __STARPU_USE_CUDA
	starpu_execute_on_each_worker(STARPUFFT(fft_3d_plan_gpu), plan, STARPU_CUDA);
#endif

	return plan;
}

/* Actually submit all the tasks. */
static struct starpu_task *
STARPUFFT(start3dC2C)(STARPUFFT(plan) plan, starpu_data_handle_t in, starpu_data_handle_t out)
{
	STARPU_ASSERT(plan->type == C2C);
	int ret;

if (PARALLEL) {
	/* TODO */
	STARPU_ASSERT(0);
} else /* !PARALLEL */ {
	struct starpu_task *task;

	/* Create FFT task */
	task = starpu_task_create();
	task->detach = 0;
	task->cl = &STARPUFFT(fft_3d_codelet);
	task->handles[0] = in;
	task->handles[1] = out;
	task->cl_arg = plan;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) return NULL;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return task;
}
}
