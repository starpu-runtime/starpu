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

#include "cg.h"
#include <math.h>

struct kernel_params {
	TYPE p1;
	TYPE p2;
};

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


/*
 *	Reduction accumulation methods
 */

#ifdef STARPU_USE_CUDA
static void accumulate_variable_cuda(void *descr[], void *cl_arg)
{
	TYPE *v_dst = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);
 
	cublasaxpy(1, (TYPE)1.0, v_src, 1, v_dst, 1);
	cudaError_t ret = cudaThreadSynchronize();
	if (ret)
		STARPU_CUDA_REPORT_ERROR(ret);
}
#endif

static void accumulate_variable_cpu(void *descr[], void *cl_arg)
{
	TYPE *v_dst = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);
 
	*v_dst = *v_dst + *v_src;
}

static struct starpu_perfmodel_t accumulate_variable_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "accumulate_variable"
};

starpu_codelet accumulate_variable_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = accumulate_variable_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = accumulate_variable_cuda,
#endif
	.nbuffers = 2,
	.model = &accumulate_variable_model
};

#ifdef STARPU_USE_CUDA
static void accumulate_vector_cuda(void *descr[], void *cl_arg)
{
	TYPE *v_dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	cublasaxpy(n, (TYPE)1.0, v_src, 1, v_dst, 1);
	cudaError_t ret = cudaThreadSynchronize();
	if (ret)
		STARPU_CUDA_REPORT_ERROR(ret);
}
#endif

static void accumulate_vector_cpu(void *descr[], void *cl_arg)
{
	TYPE *v_dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v_src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	AXPY(n, (TYPE)1.0, v_src, 1, v_dst, 1);
}

static struct starpu_perfmodel_t accumulate_vector_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "accumulate_vector"
};

starpu_codelet accumulate_vector_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = accumulate_vector_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = accumulate_vector_cuda,
#endif
	.nbuffers = 2,
	.model = &accumulate_vector_model
};

/*
 *	Reduction initialization methods
 */

#ifdef STARPU_USE_CUDA
static void bzero_variable_cuda(void *descr[], void *cl_arg)
{
	TYPE *v = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
 
	cublasscal (1, (TYPE)0.0, v, 1);
	cudaThreadSynchronize();

}
#endif

static void bzero_variable_cpu(void *descr[], void *cl_arg)
{
	TYPE *v = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*v = (TYPE)0.0;
}

static struct starpu_perfmodel_t bzero_variable_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "bzero_variable"
};

starpu_codelet bzero_variable_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = bzero_variable_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = bzero_variable_cuda,
#endif
	.nbuffers = 1,
	.model = &bzero_variable_model
};

#ifdef STARPU_USE_CUDA
static void bzero_vector_cuda(void *descr[], void *cl_arg)
{
	TYPE *v = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	cublasscal (n, (TYPE)0.0, v, 1);
	cudaError_t ret = cudaThreadSynchronize();
	if (ret)
		STARPU_CUDA_REPORT_ERROR(ret);
}
#endif

static void bzero_vector_cpu(void *descr[], void *cl_arg)
{
	TYPE *v = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	memset(v, 0, n*sizeof(TYPE));
}

static struct starpu_perfmodel_t bzero_vector_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "bzero_vector"
};

starpu_codelet bzero_vector_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = bzero_vector_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = bzero_vector_cuda,
#endif
	.nbuffers = 1,
	.model = &bzero_vector_model
};

/*
 *	DOT kernel : s = dot(v1, v2)
 */

#ifdef STARPU_USE_CUDA
static void dot_kernel_cuda(void *descr[], void *cl_arg)
{
	cudaError_t ret;

	TYPE *dot = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]); 
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[1]);
 
	/* Get current value */
	TYPE host_dot;
	cudaMemcpy(&host_dot, dot, sizeof(TYPE), cudaMemcpyDeviceToHost);
	ret = cudaThreadSynchronize();
	if (ret)
		STARPU_CUDA_REPORT_ERROR(ret);

	TYPE local_dot = cublasdot(n, v1, 1, v2, 1);
	host_dot += local_dot;
	ret = cudaThreadSynchronize();
	if (ret)
		STARPU_CUDA_REPORT_ERROR(ret);

	cudaMemcpy(dot, &host_dot, sizeof(TYPE), cudaMemcpyHostToDevice);
	ret = cudaThreadSynchronize();
	if (ret)
		STARPU_CUDA_REPORT_ERROR(ret);
}
#endif

static void dot_kernel_cpu(void *descr[], void *cl_arg)
{
	TYPE *dot = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]); 
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[1]);

	TYPE local_dot = 0.0;
	/* Note that we explicitely cast the result of the DOT kernel because
	 * some BLAS library will return a double for sdot for instance. */
	local_dot = (TYPE)DOT(n, v1, 1, v2, 1);

	*dot = *dot + local_dot;
}

static struct starpu_perfmodel_t dot_kernel_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "dot_kernel"
};

static starpu_codelet dot_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dot_kernel_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = dot_kernel_cuda,
#endif
	.nbuffers = 3,
	.model = &dot_kernel_model
};

void dot_kernel(starpu_data_handle v1,
		starpu_data_handle v2,
		starpu_data_handle s,
		unsigned nblocks,
		int use_reduction)
{
	int ret;
	struct starpu_task *task;

	/* Blank the accumulation variable */
	task = starpu_task_create();
	task->cl = &bzero_variable_cl;
	task->buffers[0].handle = s;
	task->buffers[0].mode = STARPU_W;
	ret = starpu_task_submit(task);
	assert(!ret);

	if (use_reduction)
		starpu_task_wait_for_all();

	unsigned b;
	for (b = 0; b < nblocks; b++)
	{
		task = starpu_task_create();

		task->cl = &dot_kernel_cl;
		task->buffers[0].handle = s;
		task->buffers[0].mode = use_reduction?STARPU_REDUX:STARPU_RW;
		task->buffers[1].handle = starpu_data_get_sub_data(v1, 1, b);
		task->buffers[1].mode = STARPU_R;
		task->buffers[2].handle = starpu_data_get_sub_data(v2, 1, b);
		task->buffers[2].mode = STARPU_R;

		ret = starpu_task_submit(task);
		assert(!ret);
	}

	if (use_reduction)
		starpu_task_wait_for_all();
}

/*
 *	SCAL kernel : v1 = p1 v1
 */

#ifdef STARPU_USE_CUDA
static void scal_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	/* v1 = p1 v1 */
	TYPE alpha = params->p1;
	cublasscal(n, alpha, v1, 1);
	cudaThreadSynchronize();

	free(params);
}
#endif

static void scal_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	/* v1 = p1 v1 */
	TYPE alpha = params->p1;

	SCAL(n, alpha, v1, 1);

	free(params);
}

static struct starpu_perfmodel_t scal_kernel_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "scal_kernel"
};

static starpu_codelet scal_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = scal_kernel_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = scal_kernel_cuda,
#endif
	.nbuffers = 1,
	.model = &scal_kernel_model
};

/*
 *	GEMV kernel : v1 = p1 * v1 + p2 * M v2 
 */

#ifdef STARPU_USE_CUDA
static void gemv_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);
	TYPE *M = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[1]);
 
	TYPE alpha = params->p2;
	TYPE beta = params->p1;

	/* Compute v1 = alpha M v2 + beta v1 */
	cublasgemv('N', nx, ny, alpha, M, ld, v2, 1, beta, v1, 1);
	cudaThreadSynchronize();

	free(params);
}
#endif

static void gemv_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);
	TYPE *M = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[1]);

	TYPE alpha = params->p2;
	TYPE beta = params->p1;

	/* Compute v1 = alpha M v2 + beta v1 */
	GEMV("N", nx, ny, alpha, M, ld, v2, 1, beta, v1, 1);

	free(params);
}

static struct starpu_perfmodel_t gemv_kernel_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "gemv_kernel"
};

static starpu_codelet gemv_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = gemv_kernel_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = gemv_kernel_cuda,
#endif
	.nbuffers = 3,
	.model = &gemv_kernel_model
};

void gemv_kernel(starpu_data_handle v1,
		starpu_data_handle matrix,
		starpu_data_handle v2,
		TYPE p1, TYPE p2,
		unsigned nblocks,
		int use_reduction)
{
	int ret;

	unsigned b1, b2;

	if (use_reduction)
		starpu_task_wait_for_all();

	for (b2 = 0; b2 < nblocks; b2++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &scal_kernel_cl;
		task->buffers[0].handle = starpu_data_get_sub_data(v1, 1, b2);
		task->buffers[0].mode = STARPU_RW;
		
		struct kernel_params *params = malloc(sizeof(struct kernel_params));
		params->p1 = p1;

		task->cl_arg = params;
		ret = starpu_task_submit(task);
		assert(!ret);
	}

	if (use_reduction)
		starpu_task_wait_for_all();

	for (b2 = 0; b2 < nblocks; b2++)
	{
		for (b1 = 0; b1 < nblocks; b1++)
		{
			struct starpu_task *task = starpu_task_create();
		
			task->cl = &gemv_kernel_cl;
			task->buffers[0].handle = starpu_data_get_sub_data(v1, 1, b2);
			task->buffers[0].mode = use_reduction?STARPU_REDUX:STARPU_RW;
			task->buffers[1].handle = starpu_data_get_sub_data(matrix, 2, b2, b1);
			task->buffers[1].mode = STARPU_R;
			task->buffers[2].handle = starpu_data_get_sub_data(v2, 1, b1);
			task->buffers[2].mode = STARPU_R;
		
			struct kernel_params *params = malloc(sizeof(struct kernel_params));
			assert(params);
			params->p1 = 1.0;
			params->p2 = p2;
		
			task->cl_arg = params;
		
			ret = starpu_task_submit(task);
			assert(!ret);
		}
	}

	if (use_reduction)
		starpu_task_wait_for_all();
}

/*
 *	AXPY + SCAL kernel : v1 = p1 * v1 + p2 * v2 
 */
#ifdef STARPU_USE_CUDA
static void scal_axpy_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	/* Compute v1 = p1 * v1 + p2 * v2.
	 *	v1 = p1 v1
	 *	v1 = v1 + p2 v2
	 */
	cublasscal(n, params->p1, v1, 1);
	cublasaxpy(n, params->p2, v2, 1, v1, 1);
	cudaThreadSynchronize();

	free(params);
}
#endif

static void scal_axpy_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
 
	/* Compute v1 = p1 * v1 + p2 * v2.
	 *	v1 = p1 v1
	 *	v1 = v1 + p2 v2
	 */
	SCAL(nx, params->p1, v1, 1);
	AXPY(nx, params->p2, v2, 1, v1, 1);

	free(params);
}

static struct starpu_perfmodel_t scal_axpy_kernel_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "scal_axpy_kernel"
};

static starpu_codelet scal_axpy_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = scal_axpy_kernel_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = scal_axpy_kernel_cuda,
#endif
	.nbuffers = 2,
	.model = &scal_axpy_kernel_model
};

void scal_axpy_kernel(starpu_data_handle v1, TYPE p1,
			starpu_data_handle v2, TYPE p2,
			unsigned nblocks)
{
	int ret;

	unsigned b;
	for (b = 0; b < nblocks; b++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &scal_axpy_kernel_cl;
		task->buffers[0].handle = starpu_data_get_sub_data(v1, 1, b);
		task->buffers[0].mode = STARPU_RW;
		task->buffers[1].handle = starpu_data_get_sub_data(v2, 1, b);
		task->buffers[1].mode = STARPU_R;
	
		struct kernel_params *params = malloc(sizeof(struct kernel_params));
		assert(params);
		params->p1 = p1;
		params->p2 = p2;
	
		task->cl_arg = params;
	
		ret = starpu_task_submit(task);
		assert(!ret);
	}
}


/*
 *	AXPY kernel : v1 = v1 + p1 * v2 
 */
#ifdef STARPU_USE_CUDA
static void axpy_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
 
	/* Compute v1 = v1 + p1 * v2.
	 */
	cublasaxpy(n, params->p1, v2, 1, v1, 1);
	cudaThreadSynchronize();

	free(params);
}
#endif

static void axpy_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
 
	/* Compute v1 = p1 * v1 + p2 * v2.
	 */
	AXPY(nx, params->p1, v2, 1, v1, 1);

	free(params);
}

static struct starpu_perfmodel_t axpy_kernel_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "axpy_kernel"
};

static starpu_codelet axpy_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = axpy_kernel_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = axpy_kernel_cuda,
#endif
	.nbuffers = 2,
	.model = &axpy_kernel_model
};

void axpy_kernel(starpu_data_handle v1,
		starpu_data_handle v2, TYPE p1,
		unsigned nblocks)
{
	int ret;
	unsigned b;

	for (b = 0; b < nblocks; b++)
	{
		struct starpu_task *task = starpu_task_create();
	
		task->cl = &axpy_kernel_cl;
		task->buffers[0].handle = starpu_data_get_sub_data(v1, 1, b);
		task->buffers[0].mode = STARPU_RW;
		task->buffers[1].handle = starpu_data_get_sub_data(v2, 1, b);
		task->buffers[1].mode = STARPU_R;
	
		struct kernel_params *params = malloc(sizeof(struct kernel_params));
		assert(params);
		params->p1 = p1;
	
		task->cl_arg = params;
	
		ret = starpu_task_submit(task);
		assert(!ret);
	}
}


/*
 *	COPY kernel : vector_dst <- vector_src
 */

static void copy_handle_cpu(void *descr[], void *cl_arg)
{
	TYPE *dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(descr[0]);

	memcpy(dst, src, nx*elemsize);
}

#ifdef STARPU_USE_CUDA
static void copy_handle_cuda(void *descr[], void *cl_arg)
{
	TYPE *dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(descr[0]);

	cudaMemcpy(dst, src, nx*elemsize, cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();
}
#endif

static struct starpu_perfmodel_t copy_handle_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "copy_handle"
};

static starpu_codelet copy_handle_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = copy_handle_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = copy_handle_cuda,
#endif
	.nbuffers = 2,
	.model = &copy_handle_model
};

void copy_handle(starpu_data_handle dst, starpu_data_handle src, unsigned nblocks)
{
	int ret;
	unsigned b;

	for (b = 0; b < nblocks; b++)
	{
		struct starpu_task *task = starpu_task_create();
	
		task->cl = &copy_handle_cl;
		task->buffers[0].handle = starpu_data_get_sub_data(dst, 1, b);
		task->buffers[0].mode = STARPU_W;
		task->buffers[1].handle = starpu_data_get_sub_data(src, 1, b);
		task->buffers[1].mode = STARPU_R;
	
		ret = starpu_task_submit(task);
		assert(!ret);
	}
} 
