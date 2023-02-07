/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void copy_complex_codelet_cpu(void *descr[], void *_args)
{
	(void)_args;
	int i;
	int nx = STARPU_COMPLEX_GET_NX(descr[0]);

	double *input = (double *)STARPU_COMPLEX_GET_PTR(descr[0]);
	double *output = (double *)STARPU_COMPLEX_GET_PTR(descr[1]);

	for(i=0 ; i<nx*2 ; i++)
	{
		output[i] = input[i];
	}

}

static int can_execute(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;

	if (starpu_worker_get_type(workerid) != STARPU_CUDA_WORKER)
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
	.cpu_funcs = {copy_complex_codelet_cpu},
//	.cpu_funcs_name = {"copy_complex_codelet_cpu"},
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
	starpu_data_handle_t handle4;

	double two_real_imaginary[4] = {1.0, 2.0, 3.0, 4.0};
	double real_imaginary[2] = {45.0, 12.0};
	double copy_real_imaginary[2] = {78.0, 78.0};

	int compare;
	int *compare_ptr = &compare;

	starpu_complex_data_register_ops();

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("examples/interface/complex_kernels.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	starpu_complex_data_register(&handle1, STARPU_MAIN_RAM, (uintptr_t)real_imaginary, 1);
	starpu_complex_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)copy_real_imaginary, 1);
	starpu_complex_data_register(&handle3, STARPU_MAIN_RAM, (uintptr_t)two_real_imaginary, 2);
	starpu_complex_data_register(&handle4, -1, (uintptr_t)NULL, 1);

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle1", strlen("handle1")+1, STARPU_R, handle1, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle2", strlen("handle2")+1, STARPU_R, handle2, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Compare two different complexs.  */
	ret = starpu_task_insert(&cl_compare, STARPU_R, handle1, STARPU_R, handle2, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	if (compare != 0)
	{
	     _FPRINTF(stderr, "Complex numbers should NOT be similar\n");
	     goto end;
	}

	/* Copy one into the other.  */
	ret = starpu_task_insert(&cl_copy, STARPU_R, handle1, STARPU_W, handle2, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle1", strlen("handle1")+1, STARPU_R, handle1, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle2", strlen("handle2")+1, STARPU_R, handle2, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* And compare again.  */
	ret = starpu_task_insert(&cl_compare, STARPU_R, handle1, STARPU_R, handle2, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	if (compare != 1)
	{
	     _FPRINTF(stderr, "Complex numbers should be similar\n");
	}

	/* Put another value again */
	starpu_data_acquire(handle2, STARPU_W);
	copy_real_imaginary[0] = 78.0;
	copy_real_imaginary[1] = 77.0;
	starpu_data_release(handle2);

	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle2", strlen("handle2")+1, STARPU_R, handle2, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	_FPRINTF(stderr, "\n");

	/* Display handle3.  */
	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle3", strlen("handle3")+1, STARPU_R, handle3, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Split it in two pieces (thus one complex each).  */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_complex_filter_block,
		.nchildren = 2,
	};
	starpu_data_partition(handle3, &f);

	/* Copy the two complexs into each part */
	ret = starpu_task_insert(&cl_copy, STARPU_R, handle1, STARPU_W, starpu_data_get_sub_data(handle3, 1, 0), STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_insert(&cl_copy, STARPU_R, handle2, STARPU_W, starpu_data_get_sub_data(handle3, 1, 1), STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Gather the two pieces.  */
	starpu_data_unpartition(handle3, STARPU_MAIN_RAM);

	/* Show it.  */
	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle3", strlen("handle3")+1, STARPU_R, handle3, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Use helper starpu_data_cpy */
	ret = starpu_data_cpy(handle4, handle1, 0, NULL, NULL);
	if (ret == -ENODEV) goto end;
	ret = starpu_task_insert(&cl_display, STARPU_VALUE, "handle4", strlen("handle4")+1, STARPU_R, handle4, STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Compare two different complexs.  */
	ret = starpu_task_insert(&cl_compare, STARPU_R, handle1, STARPU_R, handle4, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), STARPU_TASK_SYNCHRONOUS, 1, 0);
	if (ret == -ENODEV) goto end;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	if (compare != 1)
	{
	     _FPRINTF(stderr, "Complex numbers should be similar\n");
	     goto end;
	}

end:
#ifdef STARPU_USE_OPENCL
	{
	     int ret2 = starpu_opencl_unload_opencl(&opencl_program);
	     STARPU_CHECK_RETURN_VALUE(ret2, "starpu_opencl_unload_opencl");
	}
#endif
	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_data_unregister(handle3);
	starpu_data_unregister(handle4);
	starpu_shutdown();
	if (ret == -ENODEV) return 77; else return !compare;
}
