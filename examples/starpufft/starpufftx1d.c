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

#define DIV_1D 64

#define STEP_TAG_1D(plan, step, i) _STEP_TAG(plan, step, i)

#ifdef USE_CUDA
/* Twist the full vector into a n2 chunk */
static void
STARPUFFT(twist1_1d_kernel_gpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];

	_cufftComplex * restrict in = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[0]);
	_cufftComplex * restrict twisted1 = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[1]);
	
	cudaStream_t stream = STARPUFFT(get_local_stream)(plan, starpu_get_worker_id());

	STARPUFFT(cuda_twist1_1d_host)(in, twisted1, i, n1, n2, stream);

	cudaStreamSynchronize(stream);
}

/* Perform an n2 fft */
static void
STARPUFFT(fft1_1d_kernel_gpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int n2 = plan->n2[0];
	cufftResult cures;

	_cufftComplex * restrict in = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[0]);
	_cufftComplex * restrict out = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[1]);
	const _cufftComplex * restrict roots = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[2]);

	int workerid = starpu_get_worker_id();

	cudaStream_t stream;

	if (!plan->plans[workerid].initialized1) {
		cures = cufftPlan1d(&plan->plans[workerid].plan1_cuda, n2, _CUFFT_C2C, 1);

		stream = STARPUFFT(get_local_stream)(plan, workerid);
		cufftSetStream(plan->plans[workerid].plan1_cuda, stream);

		STARPU_ASSERT(cures == CUFFT_SUCCESS);
		plan->plans[workerid].initialized1 = 1;
	}

	stream = plan->plans[workerid].stream;

	cures = _cufftExecC2C(plan->plans[workerid].plan1_cuda, in, out, plan->sign == -1 ? CUFFT_FORWARD : CUFFT_INVERSE);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);

	STARPUFFT(cuda_twiddle_1d_host)(out, roots, n2, i, stream);

	cudaStreamSynchronize(plan->plans[workerid].stream);
}

static void
STARPUFFT(fft2_1d_kernel_gpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];
	int n3 = n2/DIV_1D;
	cufftResult cures;

	_cufftComplex * restrict in = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[0]);
	_cufftComplex * restrict out = (_cufftComplex *)STARPU_GET_VECTOR_PTR(descr[1]);

	int workerid = starpu_get_worker_id();

	if (!plan->plans[workerid].initialized2) {
		cures = cufftPlan1d(&plan->plans[workerid].plan2_cuda, n1, _CUFFT_C2C, n3);

		cudaStream_t stream = STARPUFFT(get_local_stream)(plan, workerid);
		cufftSetStream(plan->plans[workerid].plan2_cuda, stream);

		STARPU_ASSERT(cures == CUFFT_SUCCESS);
		plan->plans[workerid].initialized2 = 1;
	}

	/* NOTE using batch support */
	cures = _cufftExecC2C(plan->plans[workerid].plan2_cuda, in, out, plan->sign == -1 ? CUFFT_FORWARD : CUFFT_INVERSE);
	STARPU_ASSERT(cures == CUFFT_SUCCESS);

	cudaStreamSynchronize(plan->plans[workerid].stream);
}
#endif

/* Twist the full vector into a n2 chunk */
static void
STARPUFFT(twist1_1d_kernel_cpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int j;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];

	STARPUFFT(complex) * restrict in = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[0]);
	STARPUFFT(complex) * restrict twisted1 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[1]);

	//printf("twist1 %d %g\n", i, (double) cabs(plan->in[i]));

	for (j = 0; j < n2; j++)
		twisted1[j] = in[i+j*n1];
}

#ifdef HAVE_FFTW
/* Perform an n2 fft */
static void
STARPUFFT(fft1_1d_kernel_cpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int i = args->i;
	int j;
	int n2 = plan->n2[0];
	int workerid = starpu_get_worker_id();

	const STARPUFFT(complex) * restrict twisted1 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[0]);
	STARPUFFT(complex) * restrict fft1 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[1]);

	_fftw_complex * restrict worker_in1 = (STARPUFFT(complex) *)plan->plans[workerid].in1;
	_fftw_complex * restrict worker_out1 = (STARPUFFT(complex) *)plan->plans[workerid].out1;

	//printf("fft1 %d %g\n", i, (double) cabs(twisted1[0]));

	memcpy(worker_in1, twisted1, plan->totsize2 * sizeof(*worker_in1));
	_FFTW(execute)(plan->plans[workerid].plan1_cpu);

	for (j = 0; j < n2; j++)
		fft1[j] = worker_out1[j] * plan->roots[0][i*j];
}
#endif

/* Twist the full vector into a package of n2/DIV_1D (n1) chunks */
static void
STARPUFFT(twist2_1d_kernel_cpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int jj = args->jj;	/* between 0 and DIV_1D */
	int jjj;		/* beetween 0 and n3 */
	int i;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];
	int n3 = n2/DIV_1D;

	STARPUFFT(complex) * restrict twisted2 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[0]);

	//printf("twist2 %d %g\n", jj, (double) cabs(plan->fft1[jj]));

	for (jjj = 0; jjj < n3; jjj++) {
		int j = jj * n3 + jjj;
		for (i = 0; i < n1; i++)
			twisted2[jjj*n1+i] = plan->fft1[i*n2+j];
	}
}

#ifdef HAVE_FFTW
/* Perform n2/DIV_1D (n1) ffts */
static void
STARPUFFT(fft2_1d_kernel_cpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	//int jj = args->jj;
	int workerid = starpu_get_worker_id();

	const STARPUFFT(complex) * restrict twisted2 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[0]);
	STARPUFFT(complex) * restrict fft2 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[1]);

	//printf("fft2 %d %g\n", jj, (double) cabs(twisted2[plan->totsize4-1]));

	_fftw_complex * restrict worker_in2 = (STARPUFFT(complex) *)plan->plans[workerid].in2;
	_fftw_complex * restrict worker_out2 = (STARPUFFT(complex) *)plan->plans[workerid].out2;

	memcpy(worker_in2, twisted2, plan->totsize4 * sizeof(*worker_in2));
	_FFTW(execute)(plan->plans[workerid].plan2_cpu);
	/* no twiddle */
	memcpy(fft2, worker_out2, plan->totsize4 * sizeof(*worker_out2));
}
#endif

/* Spread the package of n2/DIV_1D (n1) chunks into the full vector */
static void
STARPUFFT(twist3_1d_kernel_cpu)(void *descr[], void *_args)
{
	struct STARPUFFT(args) *args = _args;
	STARPUFFT(plan) plan = args->plan;
	int jj = args->jj;	/* between 0 and DIV_1D */
	int jjj;		/* beetween 0 and n3 */
	int i;
	int n1 = plan->n1[0];
	int n2 = plan->n2[0];
	int n3 = n2/DIV_1D;

	const STARPUFFT(complex) * restrict fft2 = (STARPUFFT(complex) *)STARPU_GET_VECTOR_PTR(descr[0]);

	//printf("twist3 %d %g\n", jj, (double) cabs(fft2[0]));

	for (jjj = 0; jjj < n3; jjj++) {
		int j = jj * n3 + jjj;
		for (i = 0; i < n1; i++)
			plan->out[i*n2+j] = fft2[jjj*n1+i];
	}
}

static struct starpu_perfmodel_t STARPUFFT(twist1_1d_model) = {
	.type = STARPU_HISTORY_BASED,
	.symbol = TYPE"twist1_1d"
};

static struct starpu_perfmodel_t STARPUFFT(fft1_1d_model) = {
	.type = STARPU_HISTORY_BASED,
	.symbol = TYPE"fft1_1d"
};

static struct starpu_perfmodel_t STARPUFFT(twist2_1d_model) = {
	.type = STARPU_HISTORY_BASED,
	.symbol = TYPE"twist2_1d"
};

static struct starpu_perfmodel_t STARPUFFT(fft2_1d_model) = {
	.type = STARPU_HISTORY_BASED,
	.symbol = TYPE"fft2_1d"
};

static struct starpu_perfmodel_t STARPUFFT(twist3_1d_model) = {
	.type = STARPU_HISTORY_BASED,
	.symbol = TYPE"twist3_1d"
};

static starpu_codelet STARPUFFT(twist1_1d_codelet) = {
	.where =
#ifdef USE_CUDA
		STARPU_CUDA|
#endif
		STARPU_CORE,
#ifdef USE_CUDA
	.cuda_func = STARPUFFT(twist1_1d_kernel_gpu),
#endif
	.core_func = STARPUFFT(twist1_1d_kernel_cpu),
	.model = &STARPUFFT(twist1_1d_model),
	.nbuffers = 2
};

static starpu_codelet STARPUFFT(fft1_1d_codelet) = {
	.where =
#ifdef USE_CUDA
		STARPU_CUDA|
#endif
#ifdef HAVE_FFTW
		STARPU_CORE|
#endif
		0,
#ifdef USE_CUDA
	.cuda_func = STARPUFFT(fft1_1d_kernel_gpu),
#endif
#ifdef HAVE_FFTW
	.core_func = STARPUFFT(fft1_1d_kernel_cpu),
#endif
	.model = &STARPUFFT(fft1_1d_model),
	.nbuffers = 3
};

static starpu_codelet STARPUFFT(twist2_1d_codelet) = {
	.where = STARPU_CORE,
	.core_func = STARPUFFT(twist2_1d_kernel_cpu),
	.model = &STARPUFFT(twist2_1d_model),
	.nbuffers = 1
};

static starpu_codelet STARPUFFT(fft2_1d_codelet) = {
	.where =
#ifdef USE_CUDA
		STARPU_CUDA|
#endif
#ifdef HAVE_FFTW
		STARPU_CORE|
#endif
		0,
#ifdef USE_CUDA
	.cuda_func = STARPUFFT(fft2_1d_kernel_gpu),
#endif
#ifdef HAVE_FFTW
	.core_func = STARPUFFT(fft2_1d_kernel_cpu),
#endif
	.model = &STARPUFFT(fft2_1d_model),
	.nbuffers = 2
};

static starpu_codelet STARPUFFT(twist3_1d_codelet) = {
	.where = STARPU_CORE,
	.core_func = STARPUFFT(twist3_1d_kernel_cpu),
	.model = &STARPUFFT(twist3_1d_model),
	.nbuffers = 1
};

STARPUFFT(plan)
STARPUFFT(plan_dft_1d)(int n, int sign, unsigned flags)
{
	int workerid;
	int n1 = DIV_1D;
	int n2 = n / n1;
	int n3;
	int z;
	struct starpu_task *task;

	/*
	 * Simple strategy:
	 *
	 * - twist1: twist input in n1 (n2) chunks
	 * - fft1:   perform n1 (n2) ffts
	 * - twist2: twist into n2 (n1) chunks distributed in
	 *           DIV_1D groups
	 * - fft2:   perform DIV_1D times n3 (n1) ffts
	 * - twist3: twist back into output
	 */

#ifdef USE_CUDA
	/* cufft 1D limited to 8M elements */
	while (n2 > 8 << 20) {
		n1 *= 2;
		n2 /= 2;
	}
#endif
	STARPU_ASSERT(n == n1*n2);
	STARPU_ASSERT(n1 < (1ULL << I_BITS));

	/* distribute the n2 second ffts into DIV_1D packages */
	n3 = n2 / DIV_1D;
	STARPU_ASSERT(n2 == n3*DIV_1D);

	/* TODO: flags? Automatically set FFTW_MEASURE on calibration? */
	STARPU_ASSERT(flags == 0);

	STARPUFFT(plan) plan = malloc(sizeof(*plan));
	memset(plan, 0, sizeof(*plan));

	plan->number = STARPU_ATOMIC_ADD(&starpufft_last_plan_number, 1) - 1;

	/* 4bit limitation in the tag space */
	STARPU_ASSERT(plan->number < (1ULL << NUMBER_BITS));

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
	plan->totsize3 = DIV_1D;
	plan->totsize4 = plan->totsize / plan->totsize3;
	plan->type = C2C;
	plan->sign = sign;

	compute_roots(plan);

	/* Initialize per-worker working set */
	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			/* first fft plan: one n2 fft */
			plan->plans[workerid].in1 = _FFTW(malloc)(plan->totsize2 * sizeof(_fftw_complex));
			memset(plan->plans[workerid].in1, 0, plan->totsize2 * sizeof(_fftw_complex));
			plan->plans[workerid].out1 = _FFTW(malloc)(plan->totsize2 * sizeof(_fftw_complex));
			memset(plan->plans[workerid].out1, 0, plan->totsize2 * sizeof(_fftw_complex));
			plan->plans[workerid].plan1_cpu = _FFTW(plan_dft_1d)(n2, plan->plans[workerid].in1, plan->plans[workerid].out1, sign, _FFTW_FLAGS);
			STARPU_ASSERT(plan->plans[workerid].plan1_cpu);

			/* second fft plan: n3 n1 ffts */
			plan->plans[workerid].in2 = _FFTW(malloc)(plan->totsize4 * sizeof(_fftw_complex));
			memset(plan->plans[workerid].in2, 0, plan->totsize4 * sizeof(_fftw_complex));
			plan->plans[workerid].out2 = _FFTW(malloc)(plan->totsize4 * sizeof(_fftw_complex));
			memset(plan->plans[workerid].out2, 0, plan->totsize4 * sizeof(_fftw_complex));
			plan->plans[workerid].plan2_cpu = _FFTW(plan_many_dft)(plan->dim,
					plan->n1, n3,
					/* input */ plan->plans[workerid].in2, NULL, 1, plan->totsize1,
					/* output */ plan->plans[workerid].out2, NULL, 1, plan->totsize1,
					sign, _FFTW_FLAGS);
			STARPU_ASSERT(plan->plans[workerid].plan2_cpu);
#else
#warning libstarpufft can not work correctly without libfftw3
#endif
			break;
		case STARPU_CUDA_WORKER:
#ifdef USE_CUDA
			plan->plans[workerid].initialized1 = 0;
			plan->plans[workerid].initialized2 = 0;
#endif
			break;
		default:
			STARPU_ABORT();
			break;
		}
	}

	plan->twisted1 = STARPUFFT(malloc)(plan->totsize * sizeof(*plan->twisted1));
	memset(plan->twisted1, 0, plan->totsize * sizeof(*plan->twisted1));
	plan->fft1 = STARPUFFT(malloc)(plan->totsize * sizeof(*plan->fft1));
	memset(plan->fft1, 0, plan->totsize * sizeof(*plan->fft1));
	plan->twisted2 = STARPUFFT(malloc)(plan->totsize * sizeof(*plan->twisted2));
	memset(plan->twisted2, 0, plan->totsize * sizeof(*plan->twisted2));
	plan->fft2 = STARPUFFT(malloc)(plan->totsize * sizeof(*plan->fft2));
	memset(plan->fft2, 0, plan->totsize * sizeof(*plan->fft2));

	plan->twisted1_handle = malloc(plan->totsize1 * sizeof(*plan->twisted1_handle));
	plan->fft1_handle = malloc(plan->totsize1 * sizeof(*plan->fft1_handle));
	plan->twisted2_handle = malloc(plan->totsize3 * sizeof(*plan->twisted2_handle));
	plan->fft2_handle = malloc(plan->totsize3 * sizeof(*plan->fft2_handle));

	plan->twist1_tasks = malloc(plan->totsize1 * sizeof(*plan->twist1_tasks));
	plan->fft1_tasks = malloc(plan->totsize1 * sizeof(*plan->fft1_tasks));
	plan->twist2_tasks = malloc(plan->totsize3 * sizeof(*plan->twist2_tasks));
	plan->fft2_tasks = malloc(plan->totsize3 * sizeof(*plan->fft2_tasks));
	plan->twist3_tasks = malloc(plan->totsize3 * sizeof(*plan->twist3_tasks));

	plan->fft1_args = malloc(plan->totsize1 * sizeof(*plan->fft1_args));
	plan->fft2_args = malloc(plan->totsize3 * sizeof(*plan->fft2_args));

	/* Create first-round tasks */
	for (z = 0; z < plan->totsize1; z++) {
		int i = z;
#define STEP_TAG(step)	STEP_TAG_1D(plan, step, i)

		plan->fft1_args[z].plan = plan;
		plan->fft1_args[z].i = i;

		/* Register (n2) chunks */
		starpu_register_vector_data(&plan->twisted1_handle[z], 0, (uintptr_t) &plan->twisted1[z*plan->totsize2], plan->totsize2, sizeof(*plan->twisted1));
		starpu_register_vector_data(&plan->fft1_handle[z], 0, (uintptr_t) &plan->fft1[z*plan->totsize2], plan->totsize2, sizeof(*plan->fft1));

		/* We'll need it on the CPU for the second twist anyway */
		starpu_data_set_wb_mask(plan->fft1_handle[z], 1<<0);

		/* Create twist1 task */
		plan->twist1_tasks[z] = task = starpu_task_create();
		task->cl = &STARPUFFT(twist1_1d_codelet);
		//task->buffers[0].handle = to be filled at execution
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = plan->twisted1_handle[z];
		task->buffers[1].mode = STARPU_W;
		task->cl_arg = &plan->fft1_args[z];
		task->tag_id = STEP_TAG(TWIST1);
		task->use_tag = 1;
		task->detach = 1;
		task->destroy = 0;

		/* Tell that fft1 depends on twisted1 */
		starpu_tag_declare_deps(STEP_TAG(FFT1),
				1, STEP_TAG(TWIST1));

		/* Create FFT1 task */
		plan->fft1_tasks[z] = task = starpu_task_create();
		task->cl = &STARPUFFT(fft1_1d_codelet);
		task->buffers[0].handle = plan->twisted1_handle[z];
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = plan->fft1_handle[z];
		task->buffers[1].mode = STARPU_W;
		task->buffers[2].handle = plan->roots_handle[0];
		task->buffers[2].mode = STARPU_R;
		task->cl_arg = &plan->fft1_args[z];
		task->tag_id = STEP_TAG(FFT1);
		task->use_tag = 1;
		task->detach = 1;
		task->destroy = 0;

		/* Tell that to be done with first step we need to have
		 * finished this fft1 */
		starpu_tag_declare_deps(STEP_TAG_1D(plan, JOIN, 0),
				1, STEP_TAG(FFT1));
#undef STEP_TAG
	}

	/* Create join task */
	plan->join_task = task = starpu_task_create();
	task->cl = NULL;
	task->tag_id = STEP_TAG_1D(plan, JOIN, 0);
	task->use_tag = 1;
	task->detach = 1;
	task->destroy = 0;

	/* Create second-round tasks */
	for (z = 0; z < plan->totsize3; z++) {
		int jj = z;
#define STEP_TAG(step)	STEP_TAG_1D(plan, step, jj)

		plan->fft2_args[z].plan = plan;
		plan->fft2_args[z].jj = jj;

		/* Register n3 (n1) chunks */
		starpu_register_vector_data(&plan->twisted2_handle[z], 0, (uintptr_t) &plan->twisted2[z*plan->totsize4], plan->totsize4, sizeof(*plan->twisted2));
		starpu_register_vector_data(&plan->fft2_handle[z], 0, (uintptr_t) &plan->fft2[z*plan->totsize4], plan->totsize4, sizeof(*plan->fft2));

		/* We'll need it on the CPU for the last twist anyway */
		starpu_data_set_wb_mask(plan->fft2_handle[z], 1<<0);

		/* Tell that twisted2 depends on the whole first step to be
		 * done */
		starpu_tag_declare_deps(STEP_TAG(TWIST2),
				1, STEP_TAG_1D(plan, JOIN, 0));

		/* Create twist2 task */
		plan->twist2_tasks[z] = task = starpu_task_create();
		task->cl = &STARPUFFT(twist2_1d_codelet);
		task->buffers[0].handle = plan->twisted2_handle[z];
		task->buffers[0].mode = STARPU_W;
		task->cl_arg = &plan->fft2_args[z];
		task->tag_id = STEP_TAG(TWIST2);
		task->use_tag = 1;
		task->detach = 1;
		task->destroy = 0;

		/* Tell that fft2 depends on twisted2 */
		starpu_tag_declare_deps(STEP_TAG(FFT2),
				1, STEP_TAG(TWIST2));

		/* Create FFT2 task */
		plan->fft2_tasks[z] = task = starpu_task_create();
		task->cl = &STARPUFFT(fft2_1d_codelet);
		task->buffers[0].handle = plan->twisted2_handle[z];
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = plan->fft2_handle[z];
		task->buffers[1].mode = STARPU_W;
		task->cl_arg = &plan->fft2_args[z];
		task->tag_id = STEP_TAG(FFT2);
		task->use_tag = 1;
		task->detach = 1;
		task->destroy = 0;

		/* Tell that twist3 depends on fft2 */
		starpu_tag_declare_deps(STEP_TAG(TWIST3),
				1, STEP_TAG(FFT2));

		/* Create twist3 tasks */
		plan->twist3_tasks[z] = task = starpu_task_create();
		task->cl = &STARPUFFT(twist3_1d_codelet);
		task->buffers[0].handle = plan->fft2_handle[z];
		task->buffers[0].mode = STARPU_R;
		task->cl_arg = &plan->fft2_args[z];
		task->tag_id = STEP_TAG(TWIST3);
		task->use_tag = 1;
		task->detach = 1;
		task->destroy = 0;

		/* Tell that to be completely finished we need to have finished this twisted3 */
		starpu_tag_declare_deps(STEP_TAG_1D(plan, END, 0),
				1, STEP_TAG(TWIST3));
#undef STEP_TAG
	}

	/* Create end task */
	plan->end_task = task = starpu_task_create();
	task->cl = NULL;
	task->tag_id = STEP_TAG_1D(plan, END, 0);
	task->use_tag = 1;
	task->detach = 1;
	task->destroy = 0;

	return plan;
}

static starpu_tag_t
STARPUFFT(start1dC2C)(STARPUFFT(plan) plan)
{
	STARPU_ASSERT(plan->type == C2C);
	int z;

	for (z=0; z < plan->totsize1; z++) {
		starpu_submit_task(plan->twist1_tasks[z]);
		starpu_submit_task(plan->fft1_tasks[z]);
	}

	starpu_submit_task(plan->join_task);

	for (z=0; z < plan->totsize3; z++) {
		starpu_submit_task(plan->twist2_tasks[z]);
		starpu_submit_task(plan->fft2_tasks[z]);
		starpu_submit_task(plan->twist3_tasks[z]);
	}

	starpu_submit_task(plan->end_task);

	return STEP_TAG_1D(plan, END, 0);
}

static void
STARPUFFT(free_1d_tags)(STARPUFFT(plan) plan)
{
	unsigned i;
	int n1 = plan->n1[0];

	for (i = 0; i < n1; i++) {
		starpu_tag_remove(STEP_TAG_1D(plan, TWIST1, i));
		starpu_tag_remove(STEP_TAG_1D(plan, FFT1, i));
	}

	starpu_tag_remove(STEP_TAG_1D(plan, JOIN, 0));

	for (i = 0; i < DIV_1D; i++) {
		starpu_tag_remove(STEP_TAG_1D(plan, TWIST2, i));
		starpu_tag_remove(STEP_TAG_1D(plan, FFT2, i));
		starpu_tag_remove(STEP_TAG_1D(plan, TWIST3, i));
	}

	starpu_tag_remove(STEP_TAG_1D(plan, END, 0));
}
