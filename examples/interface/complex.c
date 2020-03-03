/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "complex_interface.h"
#include "complex_codelet.h"

static int can_execute(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	if (starpu_worker_get_type(workerid) == STARPU_OPENCL_WORKER)
		return 1;

#ifdef STARPU_USE_CUDA
#ifdef STARPU_SIMGRID
	/* We don't know, let's assume it can */
	return 1;
#else
       /* Cuda device */
	const struct cudaDeviceProp *props;
	props = starpu_cuda_get_device_properties(workerid);
	if (props->major >= 2 || props->minor >= 3)
	{
		/* At least compute capability 1.3, supports doubles */
		return 1;
	}
	else
	{
		/* Old card does not support doubles */
		return 0;
	}
#endif
#else
	return 1;
#endif
}

#ifdef STARPU_USE_CUDA
extern void copy_complex_codelet_cuda(void *descr[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void copy_complex_codelet_opencl(void *buffers[], void *args);
#endif

struct starpu_codelet cl_copy =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {copy_complex_codelet_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {copy_complex_codelet_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_W},
	.can_execute = can_execute,
	.name = "cl_copy"
};

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(void)
{
	int ret = 0;
	starpu_data_handle_t handle1;
	starpu_data_handle_t handle2;
	starpu_data_handle_t handle3;

	double real = 45.0;
	double imaginary = 12.0;
	double copy_real = 78.0;
	double copy_imaginary = 78.0;

	int compare;
	int *compare_ptr = &compare;

	starpu_data_handle_t vectorh;
	struct starpu_vector_interface *vectori;
	double *vector;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("examples/interface/complex_kernels.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif
	starpu_complex_data_register(&handle1, STARPU_MAIN_RAM, &real, &imaginary, 1);
	starpu_complex_data_register(&handle2, STARPU_MAIN_RAM, &copy_real, &copy_imaginary, 1);

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle1", strlen("handle1")+1, STARPU_R, handle1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle2", strlen("handle2")+1, STARPU_R, handle2, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Compare two different complexs.  */
	ret = starpu_task_insert(&cl_compare,
				 STARPU_R, handle1,
				 STARPU_R, handle2,
				 STARPU_VALUE, &compare_ptr, sizeof(compare_ptr),
				 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	if (compare != 0)
	{
	     FPRINTF(stderr, "Complex numbers should NOT be similar\n");
	     goto end;
	}

	/* Copy one into the other.  */
	ret = starpu_task_insert(&cl_copy,
				 STARPU_R, handle1,
				 STARPU_W, handle2,
				 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle1", strlen("handle1")+1, STARPU_R, handle1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle2", strlen("handle2")+1, STARPU_R, handle2, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* And compare again.  */
	ret = starpu_task_insert(&cl_compare,
				 STARPU_R, handle1,
				 STARPU_R, handle2,
				 STARPU_VALUE, &compare_ptr, sizeof(compare_ptr),
				 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	if (compare != 1)
	{
	     FPRINTF(stderr, "Complex numbers should be similar\n");
	}

	/* Put another value again */
	starpu_data_acquire(handle2, STARPU_W);
	copy_real = 78.0;
	copy_imaginary = 77.0;
	starpu_data_release(handle2);

	/* Create a vector of two complexs.  */
	starpu_complex_data_register(&handle3, -1, 0, 0, 2);

	/* Split it in two pieces (thus one complex each).  */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_complex_filter_block,
		.nchildren = 2,
	};
	starpu_data_partition(handle3, &f);

	/* Copy the two complexs into each part */
	ret = starpu_task_insert(&cl_copy,
				 STARPU_R, handle1,
				 STARPU_W, starpu_data_get_sub_data(handle3, 1, 0),
				 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_insert(&cl_copy,
				 STARPU_R, handle2,
				 STARPU_W, starpu_data_get_sub_data(handle3, 1, 1),
				 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Gather the two pieces.  */
	starpu_data_unpartition(handle3, STARPU_MAIN_RAM);

	/* Show it.  */
	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle3", strlen("handle3")+1, STARPU_R, handle3, 0);

	/* Get the real and imaginary vectors.  */
	struct starpu_data_filter fcanon =
	{
		.filter_func = starpu_complex_filter_canonical,
		.nchildren = 2,
		.get_child_ops = starpu_complex_filter_canonical_child_ops,
	};
	starpu_data_partition(handle3, &fcanon);

	/* Check the corresponding data.  */
	vectorh = starpu_data_get_sub_data(handle3, 1, 0);
	starpu_data_acquire(vectorh, STARPU_R);
	vectori = starpu_data_get_interface_on_node(vectorh, STARPU_MAIN_RAM);
	vector = (double*) vectori->ptr;
	STARPU_ASSERT_MSG(vector[0] == 45., "Bogus value: %f instead of %f", vector[0], 45.);
	STARPU_ASSERT_MSG(vector[1] == 78., "Bogus value: %f instead of %f", vector[1], 78.);
	starpu_data_release(vectorh);

	vectorh = starpu_data_get_sub_data(handle3, 1, 1);
	starpu_data_acquire(vectorh, STARPU_R);
	vectori = starpu_data_get_interface_on_node(vectorh, STARPU_MAIN_RAM);
	vector = (double*) vectori->ptr;
	STARPU_ASSERT_MSG(vector[0] == 12., "Bogus value: %f instead of %f", vector[0], 12.);
	STARPU_ASSERT_MSG(vector[1] == 77., "Bogus value: %f instead of %f", vector[1], 77.);
	starpu_data_release(vectorh);

	starpu_data_unpartition(handle3, STARPU_MAIN_RAM);

end:
#ifdef STARPU_USE_OPENCL
	{
	     int ret2 = starpu_opencl_unload_opencl(&opencl_program);
	     STARPU_CHECK_RETURN_VALUE(ret2, "starpu_opencl_unload_opencl");
	}
#endif
	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_shutdown();
	if (ret == -ENODEV) return 77; else return !compare;
}
