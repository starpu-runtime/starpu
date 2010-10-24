/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <cuda.h>
#endif

#define INIT_VALUE	42
#define NTASKS		10000

static unsigned variable;
static starpu_data_handle variable_handle;

static uintptr_t per_worker[STARPU_NMAXWORKERS];
static starpu_data_handle per_worker_handle[STARPU_NMAXWORKERS];

/* Create per-worker handles */

static void initialize_per_worker_handle(void *arg __attribute__((unused)))
{
	int workerid = starpu_worker_get_id();
	
	/* Allocate memory on the worker, and initialize it to 0 */
	
	switch (starpu_worker_get_type(workerid)) {
		case STARPU_CPU_WORKER:
			per_worker[workerid] = (uintptr_t)calloc(1, sizeof(variable));
			break;
		case STARPU_OPENCL_WORKER:
			/* Not supported yet */
			STARPU_ABORT();
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_WORKER:
			cudaMalloc((void **)&per_worker[workerid], sizeof(variable));
			cudaMemset((void *)per_worker[workerid], 0, sizeof(variable));
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}

	STARPU_ASSERT(per_worker[workerid]);
}

/*
 *	Implement reduction method
 */

static void cpu_redux_func(void *descr[], void *cl_arg __attribute__((unused)))
{
	unsigned *a = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *b = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	fprintf(stderr, "%d = %d + %d\n", *a + *b, *a, *b);

	*a = *a + *b;
}

static struct starpu_codelet_t reduction_codelet = {
	.where = STARPU_CPU,
	.cpu_func = cpu_redux_func,
	.cuda_func = NULL,
	.nbuffers = 2,
	.model = NULL
};

/*
 *	Use per-worker local copy
 */

static void cpu_func_incr(void *descr[], void *cl_arg __attribute__((unused)))
{
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*val = *val + 1;
}

#ifdef STARPU_USE_CUDA
extern void cuda_codelet_unsigned_inc(void *descr[], void *cl_arg);
#endif

static struct starpu_codelet_t use_data_on_worker_codelet = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = cpu_func_incr,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_codelet_unsigned_inc,
#endif
	.nbuffers = 1,
	.model = NULL
};

int main(int argc, char **argv)
{
	unsigned worker;
	unsigned i;

	variable = INIT_VALUE;

        starpu_init(NULL);

	unsigned nworkers = starpu_worker_get_count();

	starpu_variable_data_register(&variable_handle, 0, (uintptr_t)&variable, sizeof(unsigned));

	/* Allocate a per-worker handle on each worker (and initialize it to 0) */
	starpu_execute_on_each_worker(initialize_per_worker_handle, NULL, STARPU_CPU|STARPU_CUDA);

	/* Register all per-worker handles */
	for (worker = 0; worker < nworkers; worker++)
	{
		STARPU_ASSERT(per_worker[worker]);

		unsigned memory_node = starpu_worker_get_memory_node(worker);
		starpu_variable_data_register(&per_worker_handle[worker], memory_node,
						per_worker[worker], sizeof(variable));
	}

	/* Submit NTASKS tasks to the different worker to simulate the usage of a data in reduction */
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &use_data_on_worker_codelet;

		int workerid = (i % nworkers);
		task->buffers[0].handle = per_worker_handle[workerid];
		task->buffers[0].mode = STARPU_RW;

		task->execute_on_a_specific_worker = 1;
		task->workerid = (unsigned)workerid;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	/* Perform the reduction of all per-worker handles into the variable_handle */
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &reduction_codelet;

		task->buffers[0].handle = variable_handle;
		task->buffers[0].mode = STARPU_RW;

		task->buffers[1].handle = per_worker_handle[worker];
		task->buffers[1].mode = STARPU_R;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_data_unregister(variable_handle);

	/* Destroy all per-worker handles */
	for (worker = 0; worker < nworkers; worker++)
		starpu_data_unregister_no_coherency(per_worker_handle[worker]);

	STARPU_ASSERT(variable == (INIT_VALUE + NTASKS));

	starpu_shutdown();

	return 0;
}
