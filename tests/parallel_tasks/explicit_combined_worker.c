/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#define N	1000
#define VECTORSIZE	1024

//static pthread_mutex_t mutex;
//static pthread_cond_t cond;
//static unsigned finished = 0;

starpu_data_handle v_handle;
static unsigned *v;

static void codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
	int worker_size = starpu_combined_worker_get_size();
	assert(worker_size > 0);
	usleep(1000/worker_size);
#if 0
	int id = starpu_worker_get_id();
	int combined_id = starpu_combined_worker_get_id();
	fprintf(stderr, "worker id %d - combined id %d - worker size %d\n", id, combined_id, worker_size);
#endif
}

static starpu_codelet cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.type = STARPU_FORKJOIN,
	.max_parallelism = INT_MAX,
	.cpu_func = codelet_null,
	.cuda_func = codelet_null,
        .opencl_func = codelet_null,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
//        struct starpu_conf conf = {
//                .sched_policy_name = "pheft",
//                .ncpus = -1,
//                .ncuda = -1,
//                .nopencl = -1,
//                .nspus = -1,
//                .use_explicit_workers_bindid = 0,
//                .use_explicit_workers_cuda_gpuid = 0,
//                .use_explicit_workers_opencl_gpuid = 0,
//                .calibrate = -1
//        };

	starpu_init(NULL);

	starpu_malloc((void **)&v, VECTORSIZE*sizeof(unsigned));
	starpu_vector_data_register(&v_handle, 0, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned nworker = starpu_worker_get_count() + starpu_combined_worker_get_count();

	unsigned iter, worker;
	for (iter = 0; iter < N; iter++)
	{
		for (worker = 0; worker < nworker; worker++)
		{
			/* execute a task on that worker */
			struct starpu_task *task = starpu_task_create();
			task->cl = &cl;

			task->buffers[0].handle = v_handle;
			task->buffers[0].mode = STARPU_R;

			task->execute_on_a_specific_worker = 1;
			task->workerid = worker;

			int ret = starpu_task_submit(task);
			if (ret == -ENODEV)
				goto enodev;
		}
	}

	starpu_task_wait_for_all();

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 77;
}
