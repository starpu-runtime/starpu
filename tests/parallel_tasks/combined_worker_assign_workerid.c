/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013	    Thibaut Lambert
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
#include "../helper.h"

/*
 * Check that one create a combined worker by hand and run tasks on it.
 */

#ifndef STARPU_QUICK_CHECK
#define N	1000
#else
#define N	100
#endif
#define VECTORSIZE	1024

static int combined_workerid;
static int combined_ncpus;

void codelet_null(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;

	STARPU_SKIP_IF_VALGRIND;

	STARPU_ASSERT(starpu_combined_worker_get_id() == combined_workerid);
	int worker_size = starpu_combined_worker_get_size();
	STARPU_ASSERT(worker_size == combined_ncpus);
	starpu_usleep(1000./worker_size);
#if 1
	int id = starpu_worker_get_id();
	int combined_id = starpu_combined_worker_get_id();
	FPRINTF(stderr, "worker id %d - combined id %d - worker size %d\n", id, combined_id, worker_size);
#endif
}

static struct starpu_codelet cl =
{
	.type = STARPU_FORKJOIN,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.cuda_funcs = {codelet_null},
	.opencl_funcs = {codelet_null},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

int main(void)
{
	starpu_data_handle_t v_handle;
	unsigned *v;
	int ret;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_conf_init");
	conf.sched_policy_name = "pheft";
	conf.calibrate = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	combined_ncpus = starpu_cpu_worker_get_count();
	if (combined_ncpus < 4) goto shutdown;

	int *workerids = malloc(sizeof(int) * combined_ncpus);
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, workerids, combined_ncpus);
	combined_ncpus /= 2;

	unsigned ctx_id = starpu_sched_ctx_get_context();
	if (ctx_id == STARPU_NMAX_SCHED_CTXS)
		ctx_id = 0;

	combined_workerid = starpu_combined_worker_assign_workerid(combined_ncpus, workerids);
	STARPU_ASSERT(combined_workerid > 0);
	free(workerids);

	struct starpu_worker_collection* workers = starpu_sched_ctx_get_worker_collection(ctx_id);
	workers->add(workers, combined_workerid);

	starpu_malloc((void **)&v, VECTORSIZE*sizeof(unsigned));
	starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	/* Allow tasks only on this combined worker */
	int nuint32 = (combined_workerid + 31) / 32;
	uint32_t *forced_workerids = malloc(sizeof(uint32_t) * nuint32);
	memset(forced_workerids, 0, sizeof(uint32_t) * nuint32);
	forced_workerids[combined_workerid / 32] |= 1U << (combined_workerid%32);

	unsigned iter;
	for (iter = 0; iter < N; iter++)
	{
		/* execute a task on that worker */
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;

		task->handles[0] = v_handle;

		if (iter % 2)
		{
			task->workerids = forced_workerids;
			task->workerids_len = nuint32;
		}
		else
		{
			task->execute_on_a_specific_worker = 1;
			task->workerid = combined_workerid;
		}

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) { task->destroy = 0; starpu_task_destroy(task); goto enodev; }
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(v_handle);
	starpu_free_noflag(v, VECTORSIZE*sizeof(unsigned));
	free(forced_workerids);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(v_handle);
	starpu_free_noflag(v, VECTORSIZE*sizeof(unsigned));
	free(forced_workerids);
	fprintf(stderr, "WARNING: No one can execute the task on workerid %u\n", combined_workerid);
shutdown:
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
