/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>

#define NX    10
#define NY    21
#define PARTSX 2
#define PARTSY 3

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void variable_cpu_func(void *buffers[], void *cl_arg)
{
	int *factor = (int *) cl_arg;

	/* local copy of the variable pointer */
	int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);

	*val *= *factor;
}

#ifdef STARPU_USE_CUDA
extern void variable_cuda_func(void *buffers[], void *cl_arg);
#endif

extern void generate_matrix_data(int *matrix, int nx, int ny, unsigned ld);
extern void print_matrix_data(starpu_data_handle_t matrix_handle);

int main(void)
{
	int *matrix;
	int ret, i, j;
	int factor = 12;

	uint32_t pos[2];

	starpu_data_handle_t handle;

	struct starpu_codelet cl_r =
	{
		.cpu_funcs = {variable_cpu_func},
		.cpu_funcs_name = {"variable_cpu_func"},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {variable_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
	#endif
		.nbuffers = 1,
		.modes = {STARPU_R},
		.name = "matrix_pick_variable_scal_r"
	};

	struct starpu_codelet cl_rw =
	{
		.cpu_funcs = {variable_cpu_func},
		.cpu_funcs_name = {"variable_cpu_func"},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {variable_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
	#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "matrix_pick_variable_scal_rw"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&matrix, NX*NY*sizeof(int));
	generate_matrix_data(matrix, NX, NY, NX);

	/* Declare data to StarPU */
	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX, NX, NY, sizeof(matrix[0]));
	FPRINTF(stderr, "IN Matrix: \n");
	print_matrix_data(handle);

	/* Partition the matrix in PARTS sub-matrices */
	struct starpu_data_filter f_matrix_vert =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = PARTSX
	};

	struct starpu_data_filter f_matrix_horiz =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = PARTSY
	};

	starpu_data_map_filters(handle, 2, &f_matrix_vert, &f_matrix_horiz);

	starpu_data_handle_t sub_matrix_handle;

	int nn;
	for(nn=0; nn<=10; nn++)
	{
			int indxi = starpu_drand48()*(PARTSX);
			int indxj = starpu_drand48()*(PARTSY);
			sub_matrix_handle = starpu_data_get_sub_data(handle, 2, indxi, indxj);
			FPRINTF(stderr, "sub Matrix: \n");
			print_matrix_data(sub_matrix_handle);

			starpu_data_handle_t var_handle[1];

			pos[0] = starpu_drand48()*(NX/PARTSX);
			pos[1] = starpu_drand48()*(NY/PARTSY);

			/* Pick a variable in the matrix */
			struct starpu_data_filter f_var =
			{
				.filter_func = starpu_matrix_filter_pick_variable,
				.filter_arg_ptr = (void*)pos,
				.nchildren = 1,
				/* the children use a variable interface*/
				.get_child_ops = starpu_matrix_filter_pick_variable_child_ops
			};
			starpu_data_partition_plan(sub_matrix_handle, &f_var, var_handle);

			FPRINTF(stderr, "Sub Variable:\n");
			int *variable = (int *)starpu_variable_get_local_ptr(var_handle[0]);
			FPRINTF(stderr, "%5d ", *variable);
			FPRINTF(stderr,"\n");

			/* Submit the task */
			struct starpu_task *task = starpu_task_create();

			FPRINTF(stderr,"Dealing with sub-variable\n");
			task->handles[0] = var_handle[0];

			if(starpu_drand48()>=0.2)
				task->cl = &cl_r;
			else
				task->cl = &cl_rw;

			task->synchronous = 1;
			task->cl_arg = &factor;
			task->cl_arg_size = sizeof(factor);

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

			/* Print result variable */
			FPRINTF(stderr,"OUT Variable:\n");
			FPRINTF(stderr, "%5d ", *variable);
			FPRINTF(stderr,"\n");

			starpu_data_partition_clean(sub_matrix_handle, 1, var_handle);
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr,"OUT Matrix: \n");
	print_matrix_data(handle);
	starpu_data_unregister(handle);

	starpu_free_noflag(matrix, NX*NY*sizeof(int));
	starpu_shutdown();

	return ret;

enodev:
	starpu_shutdown();
	return 77;
}
