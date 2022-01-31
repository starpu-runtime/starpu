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

int main(void)
{
	unsigned n=1;
	int matrix[NX][NY];
	int ret, i;

	STARPU_ASSERT((NX%PARTS) == 0);
	STARPU_ASSERT((NY%PARTS) == 0);

	starpu_data_handle_t handle;
	starpu_data_handle_t vert_handle[PARTS];
	starpu_data_handle_t horiz_handle[PARTS];

	ret = do_starpu_init();
	if (ret) return ret;

	/* Declare the whole matrix to StarPU */
	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX, NX, NY, sizeof(matrix[0][0]));

	do_init_sub_data(matrix, handle, vert_handle, starpu_matrix_filter_block, 0, NX/PARTS, NX, NX/PARTS, NY);
	do_init_sub_data(matrix, handle, horiz_handle, starpu_matrix_filter_vertical_block, NX/PARTS, 0, NX, NX, NY/PARTS);

	/* Fill the matrix */
	ret = starpu_task_insert(&cl_fill, STARPU_W, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = do_apply_sub_graph(handle, vert_handle, starpu_matrix_filter_block, 1, (NX/PARTS));
	if (ret == -ENODEV) goto enodev;

	ret = do_clean_sub_graph(handle, vert_handle);
	if (ret == -ENODEV) goto enodev;

	ret = do_apply_sub_graph(handle, horiz_handle, starpu_matrix_filter_vertical_block, 2, 2*100*(NY/PARTS));
	if (ret == -ENODEV) goto enodev;

	ret = do_clean_sub_graph(handle, horiz_handle);
	if (ret == -ENODEV) goto enodev;

	do_clean_sub_data(vert_handle);
	do_clean_sub_data(horiz_handle);
	starpu_data_unregister(handle);
	starpu_shutdown();

	return ret;

enodev:
	starpu_shutdown();
	return 77;
}
