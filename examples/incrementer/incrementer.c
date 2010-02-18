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

#include <starpu.h>
#include <pthread.h>

static unsigned niter = 50000;

#ifdef STARPU_USE_CUDA
extern void cuda_codelet(void *descr[], __attribute__ ((unused)) void *_args);
#endif

extern void cuda_codelet_host(float *tab);

void cpu_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	float *val = (float *)STARPU_GET_VECTOR_PTR(descr[0]);

	val[0] += 1.0f; val[1] += 1.0f;
}

int main(int argc, char **argv)
{
	starpu_init(NULL);

	if (argc == 2)
		niter = atoi(argv[1]);

	float float_array[3] __attribute__ ((aligned (16))) = { 0.0f, 0.0f, 0.0f}; 

	starpu_data_handle float_array_handle;
	starpu_register_vector_data(&float_array_handle, 0 /* home node */,
			(uintptr_t)&float_array, 3, sizeof(float));

	starpu_codelet cl =
	{
		/* CUBLAS stands for CUDA kernels controlled from the host */
		.where = STARPU_CPU|STARPU_CUDA,
		.cpu_func = cpu_codelet,
#ifdef STARPU_USE_CUDA
		.cuda_func = cuda_codelet,
#endif
		.nbuffers = 1
	};

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	unsigned i;
	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;
		
		task->callback_func = NULL;

		task->buffers[0].handle = float_array_handle;
		task->buffers[0].mode = STARPU_RW;

		int ret = starpu_submit_task(task);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			fprintf(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_wait_all_tasks();

	/* update the array in RAM */
	starpu_sync_data_with_mem(float_array_handle, STARPU_R);
	
	gettimeofday(&end, NULL);

	fprintf(stderr, "array -> %f, %f, %f\n", float_array[0], 
			float_array[1], float_array[2]);
	
	if (float_array[0] != float_array[1] + float_array[2])
		return 1;
	
	starpu_release_data_from_mem(float_array_handle);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 +
					(end.tv_usec - start.tv_usec));

	fprintf(stderr, "%d elems took %lf ms\n", niter, timing/1000);

	starpu_shutdown();

	return 0;
}
