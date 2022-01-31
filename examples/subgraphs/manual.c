/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>

#define NX    6
#define NY    6
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern struct starpu_codelet cl_fill;
extern struct starpu_codelet cl_check_scale;

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
	.name = "switch",
};

int do_starpu_init()
{
	int ret, i;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* force to execute task on the home_node, here it is STARPU_MAIN_RAM */
	cl_switch.specific_nodes = 1;
	for(i = 0; i < STARPU_NMAXBUFS; i++)
		cl_switch.nodes[i] = STARPU_MAIN_RAM;

	return 0;
}

void do_init_sub_data(int matrix[NX][NY], starpu_data_handle_t handle, starpu_data_handle_t sub_handle[PARTS], void (*filter_func)(void *father_interface, void *child_interface, struct starpu_data_filter *, unsigned id, unsigned nparts), int x, int y, int nx, int ny, int ld)
{
	int i;
	for (i = 0; i < PARTS; i++)
	{
		starpu_matrix_data_register(&sub_handle[i], STARPU_MAIN_RAM, (uintptr_t)&matrix[i*x][i*y], nx, ny, ld, sizeof(matrix[0][0]));
		/* But make it invalid for now, we'll access data through the whole matrix first */
		starpu_data_invalidate(sub_handle[i]);
	}
}

int do_apply_sub_graph(starpu_data_handle_t handle, starpu_data_handle_t sub_handle[PARTS], void (*filter_func)(void *father_interface, void *child_interface, struct starpu_data_filter *, unsigned id, unsigned nparts), int factor, int start)
{
	int i, ret;

	/* Now switch to vertical view of the matrix */
	struct starpu_data_descr descr[PARTS];
	for (i = 0; i < PARTS; i++)
	{
		descr[i].handle = sub_handle[i];
		descr[i].mode = STARPU_W;
	}
	ret = starpu_task_insert(&cl_switch, STARPU_RW, handle, STARPU_DATA_MODE_ARRAY, descr, PARTS, 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	/* And make sure we don't accidentally access the matrix through the whole-matrix handle */
	starpu_data_invalidate_submit(handle);

	/* Check the values of the vertical slices */
	for (i = 0; i < PARTS; i++)
	{
		int xstart = i*start;
		ret = starpu_task_insert(&cl_check_scale,
					 STARPU_RW, sub_handle[i],
					 STARPU_VALUE, &xstart, sizeof(xstart),
					 STARPU_VALUE, &factor, sizeof(factor),
					 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	return 0;
}

int do_clean_sub_graph(starpu_data_handle_t handle, starpu_data_handle_t sub_handle[PARTS])
{
	int i, ret;
	struct starpu_data_descr descr[PARTS];

	/* Now switch back to total view of the matrix */
	for (i = 0; i < PARTS; i++)
	{
		descr[i].handle = sub_handle[i];
		descr[i].mode = STARPU_RW;
	}

	ret = starpu_task_insert(&cl_switch, STARPU_DATA_MODE_ARRAY, descr, PARTS, STARPU_W, handle, 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	/* And make sure we don't accidentally access the matrix through the sub slices */
	for (i = 0; i < PARTS; i++)
		starpu_data_invalidate_submit(sub_handle[i]);

	return 0;
}

void do_clean_sub_data(starpu_data_handle_t sub_handle[PARTS])
{
	int i;
	for (i = 0; i < PARTS; i++)
	{
		starpu_data_unregister(sub_handle[i]);
	}
}

#include "main.h"
