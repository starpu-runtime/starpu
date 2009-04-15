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
#include <starpu.h>

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
void cuda_codelet_incA(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args);
void cuda_codelet_incC(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args);
#endif

starpu_data_handle v_handle;
unsigned v[3] = {0, 0, 0};

void core_codelet_incA(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	unsigned *val = (unsigned *)buffers[0].vector.ptr;
	val[0]++;
}

void core_codelet_incC(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	unsigned *val = (unsigned *)buffers[0].vector.ptr;
	val[2]++;
}

int main(int argc, char **argv)
{
	starpu_init(NULL);

	starpu_monitor_vector_data(&v_handle, 0, (uintptr_t)v, 3, sizeof(unsigned));

	unsigned iter;
	for (iter = 0; iter < K; iter++)
	{
		int ret;
		unsigned ind;
		for (ind = 0; ind < N; ind++)
		{
			/* increment a = v[0] */
			starpu_codelet cl_inc_a = {
				.where = CORE|CUBLAS,
				.core_func = core_codelet_incA,
#ifdef USE_CUDA
				.cublas_func = cuda_codelet_incA,
#endif
				.nbuffers = 1
			};

			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_a;

			task->buffers[0].state = v_handle;
			task->buffers[0].mode = RW;

			task->synchronous = 1;

			ret = starpu_submit_task(task);
			if (ret == -ENODEV)
				goto enodev;
		}

		/* synchronize v in RAM */
		starpu_sync_data_with_mem(v_handle);

		/* increment b */
		v[1]++;

		/* inform StarPU that v was modified by the apps */
		starpu_notify_data_modification(v_handle, 0);

		for (ind = 0; ind < N; ind++)
		{
			/* increment c = v[2] */
			starpu_codelet cl_inc_c = {
				.where = CORE|CUBLAS,
				.core_func = core_codelet_incC,
#ifdef USE_CUDA
				.cublas_func = cuda_codelet_incC,
#endif
				.nbuffers = 1
			};

			struct starpu_task *task = starpu_task_create();
			task->cl = &cl_inc_c;

			task->buffers[0].state = v_handle;
			task->buffers[0].mode = RW;

			task->synchronous = 1;

			ret = starpu_submit_task(task);
			if (ret == -ENODEV)
				goto enodev;
		}

	}

	starpu_sync_data_with_mem(v_handle);

	fprintf(stderr, "V = { %d, %d, %d }\n", v[0], v[1], v[2]);

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
