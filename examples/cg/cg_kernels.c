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

/*
 *	DOT kernel : s = dot(v1, v2)
 */

static void dot_kernel_cuda(void *descr[], void *cl_arg)
{
	TYPE *dot = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]); 
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[1]);
 
	TYPE local_dot = cublasdot(n, v1, 1, v2, 1);
//	fprintf(stderr, "DOT -> %e\n", local_dot);
	cudaMemcpy(dot, &local_dot, sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

static void dot_kernel_cpu(void *descr[], void *cl_arg)
{
	TYPE *dot = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]); 
	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[1]);

//	fprintf(stderr, "SDOT n = %d v1 %p v2 %p\n", n, v1, v2);
//	fprintf(stderr, "v1:");
//	print_vector(descr[1]);
//	fprintf(stderr, "v2:");
//	print_vector(descr[2]);

	*dot = DOT(n, v1, 1, v2, 1);

//	*dot = 0.0;
//	TYPE local_dot = 0.0;
//
//	unsigned i;
//	for (i =0; i < n; i++)
//		local_dot += v1[i]*v2[i]; 
//
//	*dot = local_dot;
}

static starpu_codelet dot_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dot_kernel_cpu,
	.cuda_func = dot_kernel_cuda,
	.nbuffers = 3
};

void dot_kernel(starpu_data_handle v1,
		starpu_data_handle v2,
		starpu_data_handle s)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &dot_kernel_cl;
	task->buffers[0].handle = s;
	task->buffers[0].mode = STARPU_W;
	task->buffers[1].handle = v1;
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = v2;
	task->buffers[2].mode = STARPU_R;

	ret = starpu_task_submit(task);
	assert(!ret);
}

/*
 *	GEMV kernel : v1 = p1 * v1 + p2 * M v2 
 */

struct kernel_params {
	TYPE p1;
	TYPE p2;
};

static void gemv_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);
	TYPE *M = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[1]);
 
	/* Compute v1 = p1 * v1 + p2 * M v2 */
	cublasgemv('N', nx, ny, params->p2, M, ld, v2, 1, params->p1, v1, 1);
	cudaThreadSynchronize();

	free(params);
}

static void gemv_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[2]);
	TYPE *M = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[1]);

	/* Compute v1 = p1 * v1 + p2 * M v2 */
	GEMV("N", nx, ny, params->p2, M, ld, v2, 1, params->p1, v1, 1);

	free(params);
}

static starpu_codelet gemv_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = gemv_kernel_cpu,
	.cuda_func = gemv_kernel_cuda,
	.nbuffers = 3
};

void gemv_kernel(starpu_data_handle v1,
		starpu_data_handle matrix,
		starpu_data_handle v2,
		TYPE p1, TYPE p2)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &gemv_kernel_cl;
	task->buffers[0].handle = v1;
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = matrix;
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = v2;
	task->buffers[2].mode = STARPU_R;

	struct kernel_params *params = malloc(sizeof(struct kernel_params));
	assert(params);
	params->p1 = p1;
	params->p2 = p2;

	task->cl_arg = params;

	ret = starpu_task_submit(task);
	assert(!ret);
}

/*
 *	AXPY + SCAL kernel : v1 = p1 * v1 + p2 * v2 
 */
static void scal_axpy_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned n = STARPU_MATRIX_GET_NX(descr[0]);
 
	/* Compute v1 = p1 * v1 + p2 * v2.
	 *	v1 = p1 v1
	 *	v1 = v1 + p2 v2
	 */
	cublasscal(n, params->p1, v1, 1);
	cublasaxpy(n, params->p2, v2, 1, v1, 1);
	cudaThreadSynchronize();

	free(params);
}

static void scal_axpy_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_MATRIX_GET_NX(descr[0]);
 
	/* Compute v1 = p1 * v1 + p2 * v2.
	 *	v1 = p1 v1
	 *	v1 = v1 + p2 v2
	 */
	SCAL(nx, params->p1, v1, 1);
	AXPY(nx, params->p2, v2, 1, v1, 1);

	free(params);
}

static starpu_codelet scal_axpy_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = scal_axpy_kernel_cpu,
	.cuda_func = scal_axpy_kernel_cuda,
	.nbuffers = 2
};

void scal_axpy_kernel(starpu_data_handle v1, TYPE p1,
			starpu_data_handle v2, TYPE p2)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &scal_axpy_kernel_cl;
	task->buffers[0].handle = v1;
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = v2;
	task->buffers[1].mode = STARPU_R;

	struct kernel_params *params = malloc(sizeof(struct kernel_params));
	assert(params);
	params->p1 = p1;
	params->p2 = p2;

	task->cl_arg = params;

	ret = starpu_task_submit(task);
	assert(!ret);
}


/*
 *	AXPY kernel : v1 = v1 + p1 * v2 
 */
static void axpy_kernel_cuda(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned n = STARPU_MATRIX_GET_NX(descr[0]);
 
	/* Compute v1 = v1 + p1 * v2.
	 */
	cublasaxpy(n, params->p1, v2, 1, v1, 1);
	cudaThreadSynchronize();

	free(params);
}

static void axpy_kernel_cpu(void *descr[], void *cl_arg)
{
	struct kernel_params *params = cl_arg;

	TYPE *v1 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *v2 = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);

	unsigned nx = STARPU_MATRIX_GET_NX(descr[0]);
 
	/* Compute v1 = p1 * v1 + p2 * v2.
	 */
	AXPY(nx, params->p1, v2, 1, v1, 1);

	free(params);
}

static starpu_codelet axpy_kernel_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = axpy_kernel_cpu,
	.cuda_func = axpy_kernel_cuda,
	.nbuffers = 2
};

void axpy_kernel(starpu_data_handle v1,
		starpu_data_handle v2, TYPE p1)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &axpy_kernel_cl;
	task->buffers[0].handle = v1;
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = v2;
	task->buffers[1].mode = STARPU_R;

	struct kernel_params *params = malloc(sizeof(struct kernel_params));
	assert(params);
	params->p1 = p1;

	task->cl_arg = params;

	ret = starpu_task_submit(task);
	assert(!ret);
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

//	fprintf(stderr, "MEMCPY %p -> %p length %d src[0] = %f\n", src, dst, nx*elemsize, src[0]);
}

static void copy_handle_cuda(void *descr[], void *cl_arg)
{
	TYPE *dst = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	TYPE *src = (TYPE *)STARPU_VECTOR_GET_PTR(descr[1]);
	
	unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(descr[0]);

	cudaMemcpy(dst, src, nx*elemsize, cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();
}

static starpu_codelet copy_handle_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = copy_handle_cpu,
	.cuda_func = copy_handle_cuda,
	.nbuffers = 2
};

void copy_handle(starpu_data_handle dst, starpu_data_handle src)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &copy_handle_cl;
	task->buffers[0].handle = dst;
	task->buffers[0].mode = STARPU_W;
	task->buffers[1].handle = src;
	task->buffers[1].mode = STARPU_R;

	ret = starpu_task_submit(task);
	assert(!ret);
} 
