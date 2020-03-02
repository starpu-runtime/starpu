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
 * This computes the minimum and maximum values of a big vector, using data
 * reduction to optimize the computation.
 */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <starpu.h>

#ifdef STARPU_QUICK_CHECK
static unsigned _nblocks = 512;
static unsigned _entries_per_bock = 64;
#else
static unsigned _nblocks = 8192;
static unsigned _entries_per_bock = 1024;
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define TYPE		double
#define TYPE_MAX	DBL_MAX
#define TYPE_MIN	DBL_MIN

static TYPE *_x;
static starpu_data_handle_t *_x_handles;

/* The first element (resp. second) stores the min element (resp. max). */
static TYPE _minmax[2];
static starpu_data_handle_t _minmax_handle;

/*
 *	Codelet to create a neutral element
 */

void minmax_neutral_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *array = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* Initialize current min to the greatest possible value. */
	array[0] = TYPE_MAX;

	/* Initialize current max to the smallest possible value. */
	array[1] = TYPE_MIN;
}

static struct starpu_codelet minmax_init_codelet =
{
	.cpu_funcs = {minmax_neutral_cpu_func},
	.cpu_funcs_name = {"minmax_neutral_cpu_func"},
	.modes = {STARPU_W},
	.nbuffers = 1,
	.name = "init"
};

/*
 *	Codelet to perform the reduction of two elements
 */

void minmax_redux_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	TYPE *array_dst = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	TYPE *array_src = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	/* Compute the min value */
	TYPE min_dst = array_dst[0];
	TYPE min_src = array_src[0];
	array_dst[0] = STARPU_MIN(min_dst, min_src);

	/* Compute the max value */
	TYPE max_dst = array_dst[1];
	TYPE max_src = array_src[1];
	array_dst[1] = STARPU_MAX(max_dst, max_src);
}

static struct starpu_codelet minmax_redux_codelet =
{
	.cpu_funcs = {minmax_redux_cpu_func},
	.cpu_funcs_name = {"minmax_redux_cpu_func"},
	.modes = {STARPU_RW, STARPU_R},
	.nbuffers = 2,
	.name = "redux"
};

/*
 *	Compute max/min within a vector and update the min/max value
 */

void minmax_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	/* The array containing the values */
	TYPE *local_array = (TYPE *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	TYPE *minmax = (TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	TYPE local_min = minmax[0];
	TYPE local_max = minmax[1];

	/* Compute the min and the max elements in the array */
	unsigned i;
	for (i = 0; i < n; i++)
	{
		TYPE val = local_array[i];
		local_min = STARPU_MIN(local_min, val);
		local_max = STARPU_MAX(local_max, val);
	}

	minmax[0] = local_min;
	minmax[1] = local_max;
}

static struct starpu_codelet minmax_codelet =
{
	.cpu_funcs = {minmax_cpu_func},
	.cpu_funcs_name = {"minmax_cpu_func"},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_REDUX},
	.name = "minmax"
};

/*
 *	Tasks initialization
 */

int main(void)
{
	unsigned long i;
	int ret;

	/* Not supported yet */
	if (starpu_get_env_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return 77;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned long nelems = _nblocks*_entries_per_bock;
	size_t size = nelems*sizeof(TYPE);

	_x = (TYPE *) malloc(size);
	_x_handles = (starpu_data_handle_t *) calloc(_nblocks, sizeof(starpu_data_handle_t));

	assert(_x && _x_handles);

	/* Initialize the vector with random values */
        starpu_srand48(0);
	for (i = 0; i < nelems; i++)
		_x[i] = (TYPE)starpu_drand48();

	unsigned block;
	for (block = 0; block < _nblocks; block++)
	{
		uintptr_t block_start = (uintptr_t)&_x[_entries_per_bock*block];
		starpu_vector_data_register(&_x_handles[block], STARPU_MAIN_RAM, block_start,
					    _entries_per_bock, sizeof(TYPE));
	}

	/* Initialize current min */
	_minmax[0] = TYPE_MAX;

	/* Initialize current max */
	_minmax[1] = TYPE_MIN;

	starpu_variable_data_register(&_minmax_handle, STARPU_MAIN_RAM, (uintptr_t)_minmax, 2*sizeof(TYPE));

	/* Set the methods to define neutral elements and to perform the reduction operation */
	starpu_data_set_reduction_methods(_minmax_handle, &minmax_redux_codelet, &minmax_init_codelet);

	for (block = 0; block < _nblocks; block++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &minmax_codelet;

		task->handles[0] = _x_handles[block];
		task->handles[1] = _minmax_handle;

		ret = starpu_task_submit(task);
		if (ret)
		{
			STARPU_ASSERT(ret == -ENODEV);
			FPRINTF(stderr, "This test can only run on CPUs, but there are no CPU workers (this is not a bug).\n");
			return 77;
		}
	}

	for (block = 0; block < _nblocks; block++)
	{
		starpu_data_unregister(_x_handles[block]);
	}
	starpu_data_unregister(_minmax_handle);

	FPRINTF(stderr, "Min : %e\n", _minmax[0]);
	FPRINTF(stderr, "Max : %e\n", _minmax[1]);

	STARPU_ASSERT(_minmax[0] <= _minmax[1]);

	free(_x);
	free(_x_handles);
	starpu_shutdown();

	return 0;
}
