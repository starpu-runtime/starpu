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
#include <assert.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cublas.h>
#endif

static float *x;
static float *y;
static starpu_data_handle *x_handles;
static starpu_data_handle *y_handles;

static unsigned nblocks = 4096;
static unsigned entries_per_bock = 1024;

#define DOT_TYPE double

static DOT_TYPE dot = 0.0f;
static starpu_data_handle dot_handle;

/*
 *	Codelet to create a neutral element
 */

void init_cpu_func(void *descr[], void *cl_arg)
{
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dot = 0.0f;
}

#ifdef STARPU_USE_CUDA
void init_cuda_func(void *descr[], void *cl_arg)
{
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	cudaMemset(dot, 0, sizeof(DOT_TYPE));
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

/*
 *	Codelet to perform the reduction of two elements
 */

void redux_cpu_func(void *descr[], void *cl_arg)
{
	DOT_TYPE *dota = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	DOT_TYPE *dotb = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*dota = *dota + *dotb;
}

static struct starpu_codelet_t redux_codelet = {
	.where = STARPU_CPU,
	.cpu_func = redux_cpu_func,
	.nbuffers = 2
};

/*
 *	Dot product codelet
 */

void dot_cpu_func(void *descr[], void *cl_arg)
{
	float *local_x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	float *local_y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	DOT_TYPE local_dot = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		local_dot += (DOT_TYPE)local_x[i]*(DOT_TYPE)local_y[i];
	}

	*dot = *dot + local_dot;
}

#ifdef STARPU_USE_CUDA
void dot_cuda_func(void *descr[], void *cl_arg)
{
	DOT_TYPE current_dot;
	DOT_TYPE local_dot;

	float *local_x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	float *local_y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemcpy(&current_dot, dot, sizeof(DOT_TYPE), cudaMemcpyDeviceToHost);

	int ret = cudaThreadSynchronize();

	local_dot = (DOT_TYPE)cublasSdot(n, local_x, 1, local_y, 1);

	//fprintf(stderr, "current_dot %f local dot %f -> %f\n", current_dot, local_dot, current_dot + local_dot);
	current_dot += local_dot;

	cudaThreadSynchronize();

	cudaMemcpy(dot, &current_dot, sizeof(DOT_TYPE), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
}
#endif

static struct starpu_codelet_t dot_codelet = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dot_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func = dot_cuda_func,
#endif
	.nbuffers = 3
};

/*
 *	Tasks initialization
 */

extern void starpu_data_end_reduction_mode(starpu_data_handle handle);

int main(int argc, char **argv)
{
	starpu_init(NULL);

	starpu_helper_cublas_init();

	unsigned long nelems = nblocks*entries_per_bock;
	size_t size = nelems*sizeof(float);

	x = malloc(size);
	y = malloc(size);

	x_handles = calloc(nblocks, sizeof(starpu_data_handle));
	y_handles = calloc(nblocks, sizeof(starpu_data_handle));

	assert(x && y);

        starpu_srand48(0);
	
	DOT_TYPE reference_dot = 0.0;

	unsigned long i;
	for (i = 0; i < nelems; i++)
	{
		x[i] = (float)starpu_drand48();
		y[i] = (float)starpu_drand48();

		reference_dot += (DOT_TYPE)x[i]*(DOT_TYPE)y[i];
	} 
	
	unsigned block;
	for (block = 0; block < nblocks; block++)
	{
		starpu_vector_data_register(&x_handles[block], 0,
			(uintptr_t)&x[entries_per_bock*block], entries_per_bock, sizeof(float));
		starpu_vector_data_register(&y_handles[block], 0,
			(uintptr_t)&y[entries_per_bock*block], entries_per_bock, sizeof(float));
	}

	starpu_variable_data_register(&dot_handle, 0, (uintptr_t)&dot, sizeof(DOT_TYPE));

	/*
	 *	Compute dot product with StarPU
	 */
	starpu_data_set_reduction_methods(dot_handle, &redux_codelet, &init_codelet);

	for (block = 0; block < nblocks; block++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &dot_codelet;

		task->buffers[0].handle = x_handles[block];
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = y_handles[block];
		task->buffers[1].mode = STARPU_R;
		task->buffers[2].handle = dot_handle;
		task->buffers[2].mode = STARPU_REDUX;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_data_unregister(dot_handle);

	fprintf(stderr, "Reference : %e vs. %e (Delta %e)\n", reference_dot, dot, reference_dot - dot);

	starpu_helper_cublas_shutdown();

	starpu_shutdown();

	return 0;
}
