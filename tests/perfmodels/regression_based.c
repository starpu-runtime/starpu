/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2011 (see AUTHORS file)
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

#ifdef STARPU_USE_CUDA
static void memset_cuda(void *descr[], void *arg)
{
	int *ptr = (int *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemset(ptr, 42, n);
	cudaThreadSynchronize();
}
#endif

static void memset_cpu(void *descr[], void *arg)
{
	int *ptr = (int *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	memset(ptr, 42, n);
}

static struct starpu_perfmodel_t model = {
	.type = STARPU_REGRESSION_BASED,
	.symbol = "memset_regression_based"
};

static struct starpu_perfmodel_t nl_model = {
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "non_linear_memset_regression_based"
};

static starpu_codelet memset_cl = 
{
	.where = STARPU_CUDA|STARPU_CPU,
#ifdef STARPU_USE_CUDA
	.cuda_func = memset_cuda,
#endif
	.cpu_func = memset_cpu,
	.model = &model,
	.nbuffers = 1
};

static starpu_codelet nl_memset_cl = 
{
	.where = STARPU_CUDA|STARPU_CPU,
#ifdef STARPU_USE_CUDA
	.cuda_func = memset_cuda,
#endif
	.cpu_func = memset_cpu,
	.model = &nl_model,
	.nbuffers = 1
};



static void test_memset(int nelems, starpu_codelet *codelet)
{
	int nloops = 20;
	int loop;

	starpu_data_handle *handle;
	handle = malloc(nloops*sizeof(starpu_data_handle));
	assert(handle);

	for (loop = 0; loop < nloops; loop++)
	{
		starpu_vector_data_register(&handle[loop], -1, (uintptr_t)NULL, nelems, sizeof(int));

		struct starpu_task *task = starpu_task_create();
	
		task->cl = codelet;
		task->buffers[0].handle = handle[loop];
		task->buffers[0].mode = STARPU_W;
	
		int ret = starpu_task_submit(task);
		assert(!ret);
	}

	for (loop = 0; loop < nloops; loop++)
		starpu_data_unregister(handle[loop]);

	free(handle);
}

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	starpu_conf_init(&conf);

	conf.sched_policy_name = "dm";
	conf.calibrate = 1;

	starpu_init(&conf);

	int size;
	for (size = 1024; size < 16777216; size *= 2)
	{
		/* Use a linear regression */
		test_memset(size, &memset_cl);

		/* Use a non-linear regression */
		test_memset(size, &nl_memset_cl);
	}

	starpu_task_wait_for_all();

	starpu_shutdown();

	return 0;
}
