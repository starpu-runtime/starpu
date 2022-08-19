/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int do_starpu_init()
{
	int ret, i;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	return 0;
}

void do_init_sub_data(int matrix[NX][NY], starpu_data_handle_t handle, starpu_data_handle_t sub_handle[PARTS], void (*filter_func)(void *father_interface, void *child_interface, struct starpu_data_filter *, unsigned id, unsigned nparts), int x, int y, int nx, int ny, int ld)
{
	// nothing to do
}

int do_apply_sub_graph(starpu_data_handle_t handle, starpu_data_handle_t sub_handle[PARTS], void (*filter_func)(void *father_interface, void *child_interface, struct starpu_data_filter *, unsigned id, unsigned nparts), int factor, int start)
{
	int i, ret;

	struct starpu_data_filter f =
	{
		.filter_func = filter_func,
		.nchildren = PARTS
	};
	starpu_data_partition(handle, &f);

	/* Check the values of the slices */
	for (i = 0; i < PARTS; i++)
	{
		int xstart = i*start;
		ret = starpu_task_insert(&cl_check_scale,
					 STARPU_RW, starpu_data_get_sub_data(handle, 1, i),
					 STARPU_VALUE, &xstart, sizeof(xstart),
					 STARPU_VALUE, &factor, sizeof(factor),
					 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	return 0;
}

int do_clean_sub_graph(starpu_data_handle_t handle, starpu_data_handle_t sub_handle[PARTS])
{
	// nothing to do
	return 0;
}

void do_clean_sub_data(starpu_data_handle_t sub_handle[PARTS])
{
	// nothing to do
}

#include "main.h"
