/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>

#ifdef STARPU_USE_GORDON
#include <gordon.h>
#endif

#define N	100
#define K	256
//#define N	1
//#define K	1

/*
 * In this test, we maintain a vector v = (a,b,c).
 *
 * Each iteration consists of:
 *  - increment a n times
 *  - sync v in ram
 *  - incrementer b
 *  - notify the modification of v
 *  - incrementer c n times
 *  - sync v 
 *
 * At the end, we have to make sure that if we did k iterations,
 *  v == (kn, k, kn)
 */

#ifdef STARPU_USE_CUDA
void cuda_codelet_incA(void *descr[], __attribute__ ((unused)) void *_args);
void cuda_codelet_incC(void *descr[], __attribute__ ((unused)) void *_args);
#endif

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
void opencl_codelet_incA(void *descr[], __attribute__ ((unused)) void *_args);
void opencl_codelet_incC(void *descr[], __attribute__ ((unused)) void *_args);
struct starpu_opencl_program opencl_code;
#endif

#define VECTORSIZE	16

starpu_data_handle v_handle;
static unsigned v[VECTORSIZE] __attribute__((aligned(128))) = {0, 0, 0, 0};

void cpu_codelet_incA(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *val = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	val[0]++;
}

void cpu_codelet_incC(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *val = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	val[2]++;
}

/* increment a = v[0] */
static starpu_codelet cl_inc_a = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL|STARPU_GORDON,
	.cpu_func = cpu_codelet_incA,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_codelet_incA,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func = opencl_codelet_incA,
#endif
#ifdef STARPU_USE_GORDON
	.gordon_func = kernel_incA_id,
#endif
	.nbuffers = 1
};

/* increment c = v[2] */
starpu_codelet cl_inc_c = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL|STARPU_GORDON,
	.cpu_func = cpu_codelet_incC,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_codelet_incC,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func = opencl_codelet_incC,
#endif
#ifdef STARPU_USE_GORDON
	.gordon_func = kernel_incC_id,
#endif
	.nbuffers = 1
};

int main(int argc, char **argv)
{
	starpu_init(NULL);

#ifdef STARPU_USE_GORDON
	unsigned elf_id = gordon_register_elf_plugin("./datawizard/sync_and_notify_data_gordon_kernels.spuelf");
	gordon_load_plugin_on_all_spu(elf_id);

	unsigned kernel_incA_id = gordon_register_kernel(elf_id, "incA");
	gordon_load_kernel_on_all_spu(kernel_incA_id);

	unsigned kernel_incC_id = gordon_register_kernel(elf_id, "incC");
	gordon_load_kernel_on_all_spu(kernel_incC_id);

	fprintf(stderr, "kernel incA %d incC %d elf %d\n", kernel_incA_id, kernel_incC_id, elf_id);
#endif

#ifdef STARPU_USE_OPENCL
        starpu_opencl_load_opencl_from_file("tests/datawizard/sync_and_notify_data_opencl_codelet.cl", &opencl_code);
#endif

        starpu_vector_data_register(&v_handle, 0, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned iter;
	for (iter = 0; iter < K; iter++)
	{
		int ret;
		unsigned ind;
		for (ind = 0; ind < N; ind++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_a;

			task->buffers[0].handle = v_handle;
			task->buffers[0].mode = STARPU_RW;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV)
				goto enodev;
		}

		/* synchronize v in RAM */
		starpu_data_acquire(v_handle, STARPU_RW);

		/* increment b */
		v[1]++;

		starpu_data_release(v_handle);

		for (ind = 0; ind < N; ind++)
		{
			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_c;

			task->buffers[0].handle = v_handle;
			task->buffers[0].mode = STARPU_RW;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV)
				goto enodev;
		}

	}

	starpu_data_acquire(v_handle, STARPU_RW);

	fprintf(stderr, "V = { %d, %d, %d }\n", v[0], v[1], v[2]);

	starpu_data_release(v_handle);

	starpu_shutdown();

	if ((v[0] != N*K) || (v[1] != K) || (v[2] != N*K))
		return -1;

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
