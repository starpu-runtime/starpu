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

#ifdef USE_GORDON
#include <gordon.h>
#endif

#define N	100
#define K	256

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

#ifdef USE_CUDA
void cuda_codelet_incA(void *descr[], __attribute__ ((unused)) void *_args);
void cuda_codelet_incC(void *descr[], __attribute__ ((unused)) void *_args);
#endif

#define VECTORSIZE	16

starpu_data_handle v_handle;
static unsigned v[VECTORSIZE] __attribute__((aligned(128))) = {0, 0, 0, 0};

void core_codelet_incA(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *val = (unsigned *)STARPU_GET_VECTOR_PTR(descr[0]);
	val[0]++;
}

void core_codelet_incC(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *val = (unsigned *)STARPU_GET_VECTOR_PTR(descr[0]);
	val[2]++;
}

int main(int argc, char **argv)
{
	starpu_init(NULL);

#ifdef USE_GORDON
	unsigned elf_id = gordon_register_elf_plugin("./datawizard/sync_and_notify_data_gordon_kernels.spuelf");
	gordon_load_plugin_on_all_spu(elf_id);

	unsigned kernel_incA_id = gordon_register_kernel(elf_id, "incA");
	gordon_load_kernel_on_all_spu(kernel_incA_id);

	unsigned kernel_incC_id = gordon_register_kernel(elf_id, "incC");
	gordon_load_kernel_on_all_spu(kernel_incC_id);

	fprintf(stderr, "kernel incA %d incC %d elf %d\n", kernel_incA_id, kernel_incC_id, elf_id);
#endif
	
	starpu_register_vector_data(&v_handle, 0, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned iter;
	for (iter = 0; iter < K; iter++)
	{
		int ret;
		unsigned ind;
		for (ind = 0; ind < N; ind++)
		{
			/* increment a = v[0] */
			starpu_codelet cl_inc_a = {
				.where = CORE|CUDA|GORDON,
				.core_func = core_codelet_incA,
#ifdef USE_CUDA
				.cuda_func = cuda_codelet_incA,
#endif
#ifdef USE_GORDON
				.gordon_func = kernel_incA_id,
#endif
				.nbuffers = 1
			};

			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_a;

			task->buffers[0].handle = v_handle;
			task->buffers[0].mode = STARPU_RW;

			task->synchronous = 1;

			ret = starpu_submit_task(task);
			if (ret == -ENODEV)
				goto enodev;
		}

		/* synchronize v in RAM */
		starpu_sync_data_with_mem(v_handle, STARPU_RW);

		/* increment b */
		v[1]++;

		starpu_release_data_from_mem(v_handle);

		for (ind = 0; ind < N; ind++)
		{
			/* increment c = v[2] */
			starpu_codelet cl_inc_c = {
				.where = CORE|CUDA|GORDON,
				.core_func = core_codelet_incC,
#ifdef USE_CUDA
				.cuda_func = cuda_codelet_incC,
#endif
#ifdef USE_GORDON
				.gordon_func = kernel_incC_id,
#endif
				.nbuffers = 1
			};

			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_c;

			task->buffers[0].handle = v_handle;
			task->buffers[0].mode = STARPU_RW;

			task->synchronous = 1;

			ret = starpu_submit_task(task);
			if (ret == -ENODEV)
				goto enodev;
		}

	}

	starpu_sync_data_with_mem(v_handle, STARPU_RW);

	fprintf(stderr, "V = { %d, %d, %d }\n", v[0], v[1], v[2]);

	starpu_release_data_from_mem(v_handle);

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
