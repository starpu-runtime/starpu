/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This examplifies how to access the same matrix with different partitioned
 * views, doing the coherency through partition planning.
 *
 * We first run a kernel on the whole matrix to fill it, then check the value
 * in parallel from the whole handle, from the horizontal slices, and from the
 * vertical slices. Then we switch back to the whole matrix to check and scale
 * it. Then we check the result again from the whole handle, the horizontal
 * slices, and the vertical slices. Then we switch to read-write on the
 * horizontal slices to check and scale them. Then we check again from the
 * whole handle, the horizontal slices, and the vertical slices. Eventually we
 * switch back to the whole matrix to check and scale it.
 *
 * Please keep this in sync with fmultiple_submit_readonly.c
 */

#include <starpu.h>

#define NX    6
#define NY    6
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void matrix_fill(void *buffers[], void *cl_arg)
{
	unsigned i, j;
	(void)cl_arg;

	/* length of the matrix */
	unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
			val[(j*ld)+i] = i+100*j;
	}
}

struct starpu_codelet cl_fill =
{
	.cpu_funcs = {matrix_fill},
	.cpu_funcs_name = {"matrix_fill"},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "matrix_fill"
};

void fmultiple_check_scale(void *buffers[], void *cl_arg)
{
	int start, factor;
	unsigned i, j;

	/* length of the matrix */
	unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	starpu_codelet_unpack_args(cl_arg, &start, &factor);

	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
		{
			STARPU_ASSERT(val[(j*ld)+i] == start + factor*((int)(i+100*j)));
			val[(j*ld)+i] *= 2;
		}
	}
}

#ifdef STARPU_USE_CUDA
extern void fmultiple_check_scale_cuda(void *buffers[], void *cl_arg);
#endif
struct starpu_codelet cl_check_scale =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {fmultiple_check_scale_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.cpu_funcs = {fmultiple_check_scale},
	.cpu_funcs_name = {"fmultiple_check_scale"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "fmultiple_check_scale"
};

void fmultiple_check(void *buffers[], void *cl_arg)
{
	int start, factor;
	unsigned i, j;

	/* length of the matrix */
	unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	starpu_codelet_unpack_args(cl_arg, &start, &factor);

	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
		{
			STARPU_ASSERT(val[(j*ld)+i] == start + factor*((int)(i+100*j)));
		}
	}
}

#ifdef STARPU_USE_CUDA
extern void fmultiple_check_cuda(void *buffers[], void *cl_arg);
#endif
struct starpu_codelet cl_check =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {fmultiple_check_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.cpu_funcs = {fmultiple_check},
	.cpu_funcs_name = {"fmultiple_check"},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.name = "fmultiple_check"
};

int main(void)
{
	int start, factor;
	unsigned j, n=1;
	int matrix[NX][NY];
	int ret, i;

	/* We haven't taken care otherwise */
	STARPU_ASSERT((NX%PARTS) == 0);
	STARPU_ASSERT((NY%PARTS) == 0);

	starpu_data_handle_t handle;
	starpu_data_handle_t vert_handle[PARTS];
	starpu_data_handle_t horiz_handle[PARTS];

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Disable codelet on CPUs if we have a CUDA device, to force remote execution on the CUDA device */
	if (starpu_cuda_worker_get_count())
	{
		cl_check_scale.cpu_funcs[0] = NULL;
		cl_check_scale.cpu_funcs_name[0] = NULL;
		cl_check.cpu_funcs[0] = NULL;
		cl_check.cpu_funcs_name[0] = NULL;
	}

	/* Declare the whole matrix to StarPU */
	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX, NX, NY, sizeof(matrix[0][0]));

	/* Partition the matrix in PARTS vertical slices */
	struct starpu_data_filter f_vert =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = PARTS
	};
	starpu_data_partition_plan(handle, &f_vert, vert_handle);

	/* Partition the matrix in PARTS horizontal slices */
	struct starpu_data_filter f_horiz =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = PARTS
	};
	starpu_data_partition_plan(handle, &f_horiz, horiz_handle);

	/* Fill the matrix */
	ret = starpu_task_insert(&cl_fill, STARPU_W, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	factor = 1;

	/* Look at readonly vertical and horizontal view of the matrix */

	/* Check the values of the vertical slices */
	for (i = 0; i < PARTS; i++)
	{
		start = i*(NX/PARTS);
		ret = starpu_task_insert(&cl_check,
				STARPU_R, vert_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	/* Check the values of the horizontal slices */
	for (i = 0; i < PARTS; i++)
	{
		start = factor*100*i*(NY/PARTS);
		ret = starpu_task_insert(&cl_check,
				STARPU_R, horiz_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	/* And of the main matrix */
	start = 0;
	ret = starpu_task_insert(&cl_check,
			STARPU_R, handle,
			STARPU_VALUE, &start, sizeof(start),
			STARPU_VALUE, &factor, sizeof(factor),
			0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* Now look at the total view of the matrix, and modify it. StarPU has to unpartition everything */

	/* Check and scale it */
	start = 0;
	ret = starpu_task_insert(&cl_check_scale,
			STARPU_RW, handle,
			STARPU_VALUE, &start, sizeof(start),
			STARPU_VALUE, &factor, sizeof(factor),
			0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	factor = 2;

	/* Look again readonly vertical and horizontal slices */

	/* Check the values of the vertical slices */
	for (i = 0; i < PARTS; i++)
	{
		start = factor*i*(NX/PARTS);
		ret = starpu_task_insert(&cl_check,
				STARPU_R, vert_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	/* Check the values of the horizontal slices */
	for (i = 0; i < PARTS; i++)
	{
		start = factor*100*i*(NY/PARTS);
		ret = starpu_task_insert(&cl_check,
				STARPU_R, horiz_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	/* And of the main matrix */
	start = 0;
	ret = starpu_task_insert(&cl_check,
			STARPU_R, handle,
			STARPU_VALUE, &start, sizeof(start),
			STARPU_VALUE, &factor, sizeof(factor),
			0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* Now try to touch horizontal slices */

	/* Check and scale the values of the horizontal slices */
	for (i = 0; i < PARTS; i++)
	{
		start = factor*100*i*(NY/PARTS);
		ret = starpu_task_insert(&cl_check_scale,
				STARPU_RW, horiz_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	factor = 4;

	/* And come back to read-only */

	/* Check the values of the vertical slices */
	for (i = 0; i < PARTS; i++)
	{
		start = factor*i*(NX/PARTS);
		ret = starpu_task_insert(&cl_check,
				STARPU_R, vert_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	/* Check the values of the horizontal slices */
	for (i = 0; i < PARTS; i++)
	{
		start = factor*100*i*(NY/PARTS);
		ret = starpu_task_insert(&cl_check,
				STARPU_R, horiz_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	/* And of the main matrix */
	start = 0;
	ret = starpu_task_insert(&cl_check,
			STARPU_R, handle,
			STARPU_VALUE, &start, sizeof(start),
			STARPU_VALUE, &factor, sizeof(factor),
			0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* And access the whole matrix again */

	/* And check and scale the values of the whole matrix */
	start = 0;
	ret = starpu_task_insert(&cl_check_scale,
			STARPU_RW, handle,
			STARPU_VALUE, &start, sizeof(start),
			STARPU_VALUE, &factor, sizeof(factor),
			0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	factor = 8;

	/*
	 * Unregister data from StarPU and shutdown.
	 */
	starpu_data_partition_clean(handle, PARTS, vert_handle);
	starpu_data_partition_clean(handle, PARTS, horiz_handle);
	starpu_data_unregister(handle);
	starpu_shutdown();

	return ret;

enodev:
	starpu_shutdown();
	return 77;
}
