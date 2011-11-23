/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
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
#include <limits.h>
#include <unistd.h>
#include "../common/helper.h"

#define N	1000
#define VECTORSIZE	1024

starpu_data_handle_t v_handle;
static unsigned *v;

static void codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
	int worker_size = starpu_combined_worker_get_size();
	assert(worker_size > 0);

//	fprintf(stderr, "WORKERSIZE : %d\n", worker_size);

	usleep(1000/worker_size);
#if 0
	int id = starpu_worker_get_id();
	int combined_id = starpu_combined_worker_get_id();
	int rank = starpu_combined_worker_get_rank();
	fprintf(stderr, "worker id %d - combined id %d - worker size %d - SPMD rank %d\n", id, combined_id, worker_size, rank);
#endif
}

static starpu_codelet cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.type = STARPU_SPMD,
	.max_parallelism = INT_MAX,
	.cpu_func = codelet_null,
	.cuda_func = codelet_null,
        .opencl_func = codelet_null,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	int ret;

        struct starpu_conf conf;
	starpu_conf_init(&conf);
        conf.sched_policy_name = "pgreedy";

	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_malloc((void **)&v, VECTORSIZE*sizeof(unsigned));
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
	starpu_vector_data_register(&v_handle, 0, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned iter;//, worker;
	for (iter = 0; iter < N; iter++)
	{
		/* execute a task on that worker */
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;

		task->buffers[0].handle = v_handle;
		task->buffers[0].mode = STARPU_R;

		int ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(v_handle);
	starpu_free(v);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(v_handle);
	starpu_free(v);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return 77;
}
