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
 * views, doing the coherency by hand.
 * We first run a kernel on the whole matrix to fill it, then run a kernel on
 * each vertical slice to check the value and multiply it by two, then run a
 * kernel on each horizontal slice to do the same.
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
#else
	/* Only enable it on CPUs if we don't have a CUDA device, to force remote execution on the CUDA device */
	.cpu_funcs = {fmultiple_check_scale},
	.cpu_funcs_name = {"fmultiple_check_scale"},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "fmultiple_check_scale"
};

void empty(void *buffers[], void *cl_arg)
{
	/* This doesn't need to do anything, it's simply used to make coherency
	 * between the two views, by simply running on the home node of the
	 * data, thus getting back all data pieces there.  */
	(void)buffers;
	(void)cl_arg;

	/* This check is just for testsuite */
	int node = starpu_task_get_current_data_node(0);
	unsigned i;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(starpu_task_get_current());
	STARPU_ASSERT(node >= 0);
	for (i = 1; i < nbuffers; i++)
		STARPU_ASSERT(starpu_task_get_current_data_node(i) == node);
}

struct starpu_codelet cl_switch =
{
	.cpu_funcs = {empty},
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
	.name = "switch"
};

int main(void)
{
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

	/* force to execute task on the home_node, here it is STARPU_MAIN_RAM */
	cl_switch.specific_nodes = 1;
	for (i = 0; i < STARPU_NMAXBUFS; i++)
		cl_switch.nodes[i] = STARPU_MAIN_RAM;

	/* Declare the whole matrix to StarPU */
	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX, NX, NY, sizeof(matrix[0][0]));

	/* Also declare the vertical slices to StarPU */
	for (i = 0; i < PARTS; i++)
	{
		starpu_matrix_data_register(&vert_handle[i], STARPU_MAIN_RAM, (uintptr_t)&matrix[0][i*(NX/PARTS)], NX, NX/PARTS, NY, sizeof(matrix[0][0]));
		/* But make it invalid for now, we'll access data through the whole matrix first */
		starpu_data_invalidate(vert_handle[i]);
	}
	/* And the horizontal slices to StarPU */
	for (i = 0; i < PARTS; i++)
	{
		starpu_matrix_data_register(&horiz_handle[i], STARPU_MAIN_RAM, (uintptr_t)&matrix[i*(NY/PARTS)][0], NX, NX, NY/PARTS, sizeof(matrix[0][0]));
		starpu_data_invalidate(horiz_handle[i]);
	}

	/* Fill the matrix */
	ret = starpu_task_insert(&cl_fill, STARPU_W, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* Now switch to vertical view of the matrix */
	struct starpu_data_descr vert_descr[PARTS];
	for (i = 0; i < PARTS; i++)
	{
		vert_descr[i].handle = vert_handle[i];
		vert_descr[i].mode = STARPU_W;
	}
	ret = starpu_task_insert(&cl_switch, STARPU_RW, handle, STARPU_DATA_MODE_ARRAY, vert_descr, PARTS, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	/* And make sure we don't accidentally access the matrix through the whole-matrix handle */
	starpu_data_invalidate_submit(handle);

	/* Check the values of the vertical slices */
	for (i = 0; i < PARTS; i++)
	{
		int factor = 1;
		int start = i*(NX/PARTS);
		ret = starpu_task_insert(&cl_check_scale,
				STARPU_RW, vert_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Now switch back to total view of the matrix */
	for (i = 0; i < PARTS; i++)
		vert_descr[i].mode = STARPU_RW;
	ret = starpu_task_insert(&cl_switch, STARPU_DATA_MODE_ARRAY, vert_descr, PARTS, STARPU_W, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	/* And make sure we don't accidentally access the matrix through the vertical slices */
	for (i = 0; i < PARTS; i++)
		starpu_data_invalidate_submit(vert_handle[i]);

	/* And switch to horizontal view of the matrix */
	struct starpu_data_descr horiz_descr[PARTS];
	for (i = 0; i < PARTS; i++)
	{
		horiz_descr[i].handle = horiz_handle[i];
		horiz_descr[i].mode = STARPU_W;
	}
	ret = starpu_task_insert(&cl_switch, STARPU_RW, handle, STARPU_DATA_MODE_ARRAY, horiz_descr, PARTS, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	/* And make sure we don't accidentally access the matrix through the whole-matrix handle */
	starpu_data_invalidate_submit(handle);

	/* Check the values of the horizontal slices */
	for (i = 0; i < PARTS; i++)
	{
		int factor = 2;
		int start = factor*100*i*(NY/PARTS);
		ret = starpu_task_insert(&cl_check_scale,
				STARPU_RW, horiz_handle[i],
				STARPU_VALUE, &start, sizeof(start),
				STARPU_VALUE, &factor, sizeof(factor),
				0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/*
	 * Unregister data from StarPU and shutdown It does not really matter
	 * which view is active at unregistration here, since all views cover
	 * the whole matrix, so it will be completely updated in the main memory.
	 */
	for (i = 0; i < PARTS; i++)
	{
		starpu_data_unregister(vert_handle[i]);
		starpu_data_unregister(horiz_handle[i]);
	}
	starpu_data_unregister(handle);
	starpu_shutdown();

	return ret;

enodev:
	starpu_shutdown();
	return 77;
}
