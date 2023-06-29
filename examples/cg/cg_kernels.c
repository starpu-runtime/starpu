/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Standard BLAS kernels used by CG
 */

#include "cg.h"
#include <math.h>
#include <limits.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <starpu_cublas_v2.h>
static const TYPE gp1 = 1.0;
static const TYPE gm1 = -1.0;
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static unsigned nblocks = 8;

#ifdef STARPU_QUICK_CHECK
static int i_max = 5;
static int long long n = 2048;
#elif !defined(STARPU_LONG_CHECK)
static int long long n = 4096;
static int i_max = 100;
#else
static int long long n = 4096;
static int i_max = 1000;
#endif
static double eps = (10e-14);

int use_reduction = 1;
int display_result = 0;

HANDLE_TYPE_MATRIX A_handle;
HANDLE_TYPE_VECTOR b_handle;
HANDLE_TYPE_VECTOR x_handle;

HANDLE_TYPE_VECTOR r_handle;
HANDLE_TYPE_VECTOR d_handle;
HANDLE_TYPE_VECTOR q_handle;

starpu_data_handle_t dtq_handle;
starpu_data_handle_t rtr_handle;
TYPE dtq, rtr;

#if 0
static void print_vector_from_descr(unsigned nx, TYPE *v)
{
	unsigned i;
	for (i = 0; i < nx; i++)
	{
		fprintf(stderr, "%2.2e ", v[i]);
	}
	fprintf(stderr, "\n");
}


static void print_matrix_from_descr(unsigned nx, unsigned ny, unsigned ld, TYPE *mat)
{
	unsigned i, j;
	for (j = 0; j < nx; j++)
	{
		for (i = 0; i < ny; i++)
		{
			fprintf(stderr, "%2.2e ", mat[j+i*ld]);
		}
		fprintf(stderr, "\n");
	}
}
#endif

static int can_execute(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	enum starpu_worker_archtype type = starpu_worker_get_type(workerid);
	if (type == STARPU_CPU_WORKER || type == STARPU_OPENCL_WORKER)
		return 1;

#ifdef STARPU_USE_CUDA
#ifdef STARPU_SIMGRID
	/* We don't know, let's assume it can */
	return 1;
#else
	/* Cuda device */
	const struct cudaDeviceProp *props;
	props = starpu_cuda_get_device_properties(workerid);
	if (props->major >= 2 || props->minor >= 3)
		/* At least compute capability 1.3, supports doubles */
		return 1;
#endif
#endif
	/* Old card, does not support doubles */
	return 0;
}

/*
 *	Reduction accumulation methods
 */

#ifdef STARPU_USE_CUDA
static void accumulate_variable_cuda(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v_dst = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	cublasStatus_t status = cublasaxpy(starpu_cublas_get_local_handle(), 1, &gp1, v_src, 1, v_dst, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void accumulate_variable_cpu(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v_dst = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*v_dst = *v_dst + *v_src;
}

static struct starpu_perfmodel accumulate_variable_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "accumulate_variable"
};

struct starpu_codelet accumulate_variable_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {accumulate_variable_cpu},
	.cpu_funcs_name = {"accumulate_variable_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {accumulate_variable_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = {STARPU_RW|STARPU_COMMUTE, STARPU_R},
	.nbuffers = 2,
	.model = &accumulate_variable_model,
	.name = "accumulate_variable"
};

#ifdef STARPU_USE_CUDA
static void accumulate_vector_cuda(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v_dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	cublasStatus_t status = cublasaxpy(starpu_cublas_get_local_handle(), nx, &gp1, v_src, 1, v_dst, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void accumulate_vector_cpu(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v_dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	AXPY(nx, (TYPE)1.0, v_src, 1, v_dst, 1);
}

static struct starpu_perfmodel accumulate_vector_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "accumulate_vector"
};

struct starpu_codelet accumulate_vector_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {accumulate_vector_cpu},
	.cpu_funcs_name = {"accumulate_vector_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {accumulate_vector_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = {STARPU_RW|STARPU_COMMUTE, STARPU_R},
	.nbuffers = 2,
	.model = &accumulate_vector_model,
	.name = "accumulate_vector"
};

/*
 *	Reduction initialization methods
 */

#ifdef STARPU_USE_CUDA
extern void zero_vector(TYPE *x, unsigned nelems);

static void bzero_variable_cuda(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	size_t size = STARPU_VARIABLE_GET_ELEMSIZE(descr[0]);

	cudaMemsetAsync(v, 0, size, starpu_cuda_get_local_stream());
}
#endif

void bzero_variable_cpu(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*v = (TYPE)0.0;
}

static struct starpu_perfmodel bzero_variable_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "bzero_variable"
};

struct starpu_codelet bzero_variable_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {bzero_variable_cpu},
	.cpu_funcs_name = {"bzero_variable_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {bzero_variable_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = {STARPU_W},
	.nbuffers = 1,
	.model = &bzero_variable_model,
	.name = "bzero_variable"
};

#ifdef STARPU_USE_CUDA
static void bzero_vector_cuda(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(descr[0]);

	cudaMemsetAsync(v, 0, nx * elemsize, starpu_cuda_get_local_stream());
}
#endif

void bzero_vector_cpu(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *v = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	memset(v, 0, nx*sizeof(TYPE));
}

static struct starpu_perfmodel bzero_vector_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "bzero_vector"
};

struct starpu_codelet bzero_vector_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {bzero_vector_cpu},
	.cpu_funcs_name = {"bzero_vector_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {bzero_vector_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.modes = {STARPU_W},
	.nbuffers = 1,
	.model = &bzero_vector_model,
	.name = "bzero_vector"
};

/*
 *	DOT kernel : s = dot(v1, v2)
 */

#ifdef STARPU_USE_CUDA
static void dot_kernel_cuda(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *dot = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[1]);

	cublasHandle_t handle = starpu_cublas_get_local_handle();
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	cublasStatus_t status = cublasdot(handle,
		nx, v1, 1, v2, 1, dot);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}
#endif

void dot_kernel_cpu(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *dot = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[1]);

	TYPE local_dot;
	/* Note that we explicitly cast the result of the DOT kernel because
	 * some BLAS library will return a double for sdot for instance. */
	local_dot = (TYPE)DOT(nx, v1, 1, v2, 1);

	*dot = *dot + local_dot;
}

static struct starpu_perfmodel dot_kernel_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "dot_kernel"
};

static struct starpu_codelet dot_kernel_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {dot_kernel_cpu},
	.cpu_funcs_name = {"dot_kernel_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dot_kernel_cuda},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.model = &dot_kernel_model,
	.name = "dot_kernel"
};

int dot_kernel(HANDLE_TYPE_VECTOR v1,
	       HANDLE_TYPE_VECTOR v2,
	       starpu_data_handle_t s,
	       unsigned nb)
{
	int ret;

	/* Blank the accumulation variable */
	if (use_reduction)
		starpu_data_invalidate_submit(s);
	else
	{
		ret = TASK_INSERT(&bzero_variable_cl, STARPU_W, s, 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "TASK_INSERT");
	}

	unsigned block;
	for (block = 0; block < nb; block++)
	{
		ret = TASK_INSERT(&dot_kernel_cl,
					 use_reduction?STARPU_REDUX:STARPU_RW, s,
					 STARPU_R, GET_VECTOR_BLOCK(v1, block),
					 STARPU_R, GET_VECTOR_BLOCK(v2, block),
					 STARPU_TAG_ONLY, (starpu_tag_t) block,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "TASK_INSERT");
	}
	return 0;
}

/*
 *	SCAL kernel : v1 = p1 v1
 */

#ifdef STARPU_USE_CUDA
static void scal_kernel_cuda(void *descr[], void *cl_arg)
{
	TYPE p1;
	starpu_codelet_unpack_args(cl_arg, &p1);

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	/* v1 = p1 v1 */
	TYPE alpha = p1;
	cublasStatus_t status = cublasscal(starpu_cublas_get_local_handle(), nx, &alpha, v1, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void scal_kernel_cpu(void *descr[], void *cl_arg)
{
	TYPE alpha;
	starpu_codelet_unpack_args(cl_arg, &alpha);

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	/* v1 = alpha v1 */
	SCAL(nx, alpha, v1, 1);
}

static struct starpu_perfmodel scal_kernel_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "scal_kernel"
};

static struct starpu_codelet scal_kernel_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {scal_kernel_cpu},
	.cpu_funcs_name = {"scal_kernel_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {scal_kernel_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.nbuffers = 1,
	.model = &scal_kernel_model,
	.name = "scal_kernel"
};

/*
 *	GEMV kernel : v1 = p1 * v1 + p2 * M v2
 */

#ifdef STARPU_USE_CUDA
static void gemv_kernel_cuda(void *descr[], void *cl_arg)
{
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);
	TYPE *M = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[1]);

	TYPE alpha, beta;
	starpu_codelet_unpack_args(cl_arg, &beta, &alpha);

	/* Compute v1 = alpha M v2 + beta v1 */
	cublasStatus_t status = cublasgemv(starpu_cublas_get_local_handle(),
			CUBLAS_OP_N, nx, ny, &alpha, M, ld, v2, 1, &beta, v1, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void gemv_kernel_cpu(void *descr[], void *cl_arg)
{
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);
	TYPE *M = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[1]);

	TYPE alpha, beta;
	starpu_codelet_unpack_args(cl_arg, &beta, &alpha);

	int worker_size = starpu_combined_worker_get_size();

	if (worker_size > 1)
	{
		/* Parallel CPU task */
		unsigned i = starpu_combined_worker_get_rank();

		unsigned bs = (ny + worker_size - 1)/worker_size;
		unsigned new_nx = STARPU_MIN(nx, bs*(i+1)) - bs*i;

		nx = new_nx;
		v1 = &v1[bs*i];
		M = &M[bs*i];
	}

	/* Compute v1 = alpha M v2 + beta v1 */
	GEMV("N", nx, ny, alpha, M, ld, v2, 1, beta, v1, 1);
}

static struct starpu_perfmodel gemv_kernel_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "gemv_kernel"
};

static struct starpu_codelet gemv_kernel_cl =
{
	.can_execute = can_execute,
	.type = STARPU_SPMD,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {gemv_kernel_cpu},
	.cpu_funcs_name = {"gemv_kernel_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {gemv_kernel_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.nbuffers = 3,
	.model = &gemv_kernel_model,
	.name = "gemv_kernel"
};

int gemv_kernel(HANDLE_TYPE_VECTOR v1,
		HANDLE_TYPE_MATRIX matrix,
		HANDLE_TYPE_VECTOR v2,
		TYPE p1, TYPE p2,
		unsigned nb)
{
	unsigned b1, b2;
	int ret;

	for (b2 = 0; b2 < nb; b2++)
	{
		ret = TASK_INSERT(&scal_kernel_cl,
					 STARPU_RW, GET_VECTOR_BLOCK(v1, b2),
					 STARPU_VALUE, &p1, sizeof(p1),
					 STARPU_TAG_ONLY, (starpu_tag_t) b2,
					 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "TASK_INSERT");
	}

	for (b2 = 0; b2 < nb; b2++)
	{
		for (b1 = 0; b1 < nb; b1++)
		{
			TYPE one = 1.0;
			ret = TASK_INSERT(&gemv_kernel_cl,
						 use_reduction?STARPU_REDUX:STARPU_RW,	GET_VECTOR_BLOCK(v1, b2),
						 STARPU_R,	GET_MATRIX_BLOCK(matrix, b2, b1),
						 STARPU_R,	GET_VECTOR_BLOCK(v2, b1),
						 STARPU_VALUE,	&one,	sizeof(one),
						 STARPU_VALUE,	&p2,	sizeof(p2),
						 STARPU_TAG_ONLY, ((starpu_tag_t)b2) * nb + b1,
						 0);
			STARPU_CHECK_RETURN_VALUE(ret, "TASK_INSERT");
		}
	}
	return 0;
}

/*
 *	AXPY + SCAL kernel : v1 = p1 * v1 + p2 * v2
 */
#ifdef STARPU_USE_CUDA
static void scal_axpy_kernel_cuda(void *descr[], void *cl_arg)
{
	TYPE p1, p2;
	starpu_codelet_unpack_args(cl_arg, &p1, &p2);

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	/* Compute v1 = p1 * v1 + p2 * v2.
	 *	v1 = p1 v1
	 *	v1 = v1 + p2 v2
	 */
	cublasStatus_t status;
	status = cublasscal(starpu_cublas_get_local_handle(), nx, &p1, v1, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
	status = cublasaxpy(starpu_cublas_get_local_handle(), nx, &p2, v2, 1, v1, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void scal_axpy_kernel_cpu(void *descr[], void *cl_arg)
{
	TYPE p1, p2;
	starpu_codelet_unpack_args(cl_arg, &p1, &p2);

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	/* Compute v1 = p1 * v1 + p2 * v2.
	 *	v1 = p1 v1
	 *	v1 = v1 + p2 v2
	 */
	SCAL(nx, p1, v1, 1);
	AXPY(nx, p2, v2, 1, v1, 1);
}

static struct starpu_perfmodel scal_axpy_kernel_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "scal_axpy_kernel"
};

static struct starpu_codelet scal_axpy_kernel_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {scal_axpy_kernel_cpu},
	.cpu_funcs_name = {"scal_axpy_kernel_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {scal_axpy_kernel_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.nbuffers = 2,
	.model = &scal_axpy_kernel_model,
	.name = "scal_axpy_kernel"
};

int scal_axpy_kernel(HANDLE_TYPE_VECTOR v1, TYPE p1,
		     HANDLE_TYPE_VECTOR v2, TYPE p2,
		     unsigned nb)
{
	unsigned block;
	for (block = 0; block < nb; block++)
	{
		int ret;
		ret = TASK_INSERT(&scal_axpy_kernel_cl,
					 STARPU_RW, GET_VECTOR_BLOCK(v1, block),
					 STARPU_R,  GET_VECTOR_BLOCK(v2, block),
					 STARPU_VALUE, &p1, sizeof(p1),
					 STARPU_VALUE, &p2, sizeof(p2),
					 STARPU_TAG_ONLY, (starpu_tag_t) block,
					 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "TASK_INSERT");
	}
	return 0;
}


/*
 *	AXPY kernel : v1 = v1 + p1 * v2
 */
#ifdef STARPU_USE_CUDA
static void axpy_kernel_cuda(void *descr[], void *cl_arg)
{
	TYPE p1;
	starpu_codelet_unpack_args(cl_arg, &p1);

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	/* Compute v1 = v1 + p1 * v2.
	 */
	cublasStatus_t status = cublasaxpy(starpu_cublas_get_local_handle(),
			nx, &p1, v2, 1, v1, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

void axpy_kernel_cpu(void *descr[], void *cl_arg)
{
	TYPE p1;
	starpu_codelet_unpack_args(cl_arg, &p1);

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);

	/* Compute v1 = p1 * v1 + p2 * v2.
	 */
	AXPY(nx, p1, v2, 1, v1, 1);
}

static struct starpu_perfmodel axpy_kernel_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "axpy_kernel"
};

static struct starpu_codelet axpy_kernel_cl =
{
	.can_execute = can_execute,
	.cpu_funcs = {axpy_kernel_cpu},
	.cpu_funcs_name = {"axpy_kernel_cpu"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {axpy_kernel_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.nbuffers = 2,
	.model = &axpy_kernel_model,
	.name = "axpy_kernel"
};

int axpy_kernel(HANDLE_TYPE_VECTOR v1,
		HANDLE_TYPE_VECTOR v2, TYPE p1,
		unsigned nb)
{
	unsigned block;
	for (block = 0; block < nb; block++)
	{
		int ret;
		ret = TASK_INSERT(&axpy_kernel_cl,
					 STARPU_RW, GET_VECTOR_BLOCK(v1, block),
					 STARPU_R,  GET_VECTOR_BLOCK(v2, block),
					 STARPU_VALUE, &p1, sizeof(p1),
					 STARPU_TAG_ONLY, (starpu_tag_t) block,
					 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "TASK_INSERT");
	}
	return 0;
}


/*
 *	Main loop
 */
int cg(void)
{
	TYPE delta_new, delta_0, error, delta_old, alpha, beta;
	double start, end, timing;
	int i = 0, ret;

	/* r <- b */
	ret = copy_handle(r_handle, b_handle, nblocks);
	if (ret == -ENODEV) return ret;

	/* r <- r - A x */
	ret = gemv_kernel(r_handle, A_handle, x_handle, 1.0, -1.0, nblocks);
	if (ret == -ENODEV) return ret;

	/* d <- r */
	ret = copy_handle(d_handle, r_handle, nblocks);
	if (ret == -ENODEV) return ret;

	/* delta_new = dot(r,r) */
	ret = dot_kernel(r_handle, r_handle, rtr_handle, nblocks);
	if (ret == -ENODEV) return ret;

	GET_DATA_HANDLE(rtr_handle);
	starpu_data_acquire(rtr_handle, STARPU_R);
	delta_new = rtr;
	delta_0 = delta_new;
	starpu_data_release(rtr_handle);

	FPRINTF_SERVER(stderr, "Delta limit: %e\n", (double) (eps*eps*delta_0));

	FPRINTF_SERVER(stderr, "**************** INITIAL ****************\n");
	FPRINTF_SERVER(stderr, "Delta 0: %e\n", delta_new);

	BARRIER();
	start = starpu_timing_now();

	while ((i < i_max) && ((double)delta_new > (double)(eps*eps*delta_0)))
	{
		starpu_iteration_push(i);

		/* q <- A d */
		gemv_kernel(q_handle, A_handle, d_handle, 0.0, 1.0, nblocks);

		/* dtq <- dot(d,q) */
		dot_kernel(d_handle, q_handle, dtq_handle, nblocks);

		/* alpha = delta_new / dtq */
		GET_DATA_HANDLE(dtq_handle);
		starpu_data_acquire(dtq_handle, STARPU_R);
		alpha = delta_new / dtq;
		starpu_data_release(dtq_handle);

		/* x <- x + alpha d */
		axpy_kernel(x_handle, d_handle, alpha, nblocks);

		if ((i % 50) == 0)
		{
			/* r <- b */
			copy_handle(r_handle, b_handle, nblocks);

			/* r <- r - A x */
			gemv_kernel(r_handle, A_handle, x_handle, 1.0, -1.0, nblocks);
		}
		else
		{
			/* r <- r - alpha q */
			axpy_kernel(r_handle, q_handle, -alpha, nblocks);
		}

		/* delta_new = dot(r,r) */
		dot_kernel(r_handle, r_handle, rtr_handle, nblocks);

		GET_DATA_HANDLE(rtr_handle);
		starpu_data_acquire(rtr_handle, STARPU_R);
		delta_old = delta_new;
		delta_new = rtr;
		beta = delta_new / delta_old;
		starpu_data_release(rtr_handle);

		/* d <- beta d + r */
		scal_axpy_kernel(d_handle, beta, r_handle, 1.0, nblocks);

		if ((i % 10) == 0)
		{
			/* We here take the error as ||r||_2 / (n||b||_2) */
			error = sqrt(delta_new/delta_0)/(1.0*n);
			FPRINTF_SERVER(stderr, "*****************************************\n");
			FPRINTF_SERVER(stderr, "iter %d DELTA %e - %e\n", i, delta_new, error);
		}

		starpu_iteration_pop();
		i++;
	}

	BARRIER();
	end = starpu_timing_now();
	timing = end - start;

	error = sqrt(delta_new/delta_0)/(1.0*n);
	FPRINTF_SERVER(stderr, "*****************************************\n");
	FPRINTF_SERVER(stderr, "iter %d DELTA %e - %e\n", i, delta_new, error);
	FPRINTF_SERVER(stderr, "Total timing : %2.2f seconds\n", timing/1e6);
	FPRINTF_SERVER(stderr, "Seconds per iteration : %2.2e seconds\n", timing/1e6/i);
	FPRINTF_SERVER(stderr, "Number of iterations per second : %2.2e it/s\n", i/(timing/1e6));

	return 0;
}


void parse_common_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-n") == 0)
		{
			n = (int long long)atoi(argv[++i]);
			continue;
		}

		if (strcmp(argv[i], "-display-result") == 0)
		{
			display_result = 1;
			continue;
		}

		if (strcmp(argv[i], "-maxiter") == 0)
		{
			i_max = atoi(argv[++i]);
			if (i_max <= 0)
			{
				FPRINTF_SERVER(stderr, "the number of iterations must be positive, not %d\n", i_max);
				exit(EXIT_FAILURE);
			}
			continue;
		}

		if (strcmp(argv[i], "-nblocks") == 0)
		{
			nblocks = atoi(argv[++i]);
			continue;
		}

		if (strcmp(argv[i], "-no-reduction") == 0)
		{
			use_reduction = 0;
			continue;
		}
	}
}
