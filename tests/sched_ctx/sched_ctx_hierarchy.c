/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

void free_codelet(void *arg)
{
	// The argument of the function is automatically freed by StarPU
	//	free(arg);
}

void func_cpu_bis(void *descr[], void *_args)
{
	(void)descr;
	char msg;
	char worker_name[256];
	int worker_id = starpu_worker_get_id_check();
	int worker_id_expected;
	int ntasks;

	starpu_worker_get_name(worker_id, worker_name, 256);
	starpu_codelet_unpack_args(_args, &msg, &ntasks, &worker_id_expected);

	STARPU_ASSERT(worker_id == worker_id_expected);

	FPRINTF(stderr, "[msg '%c'] [worker id %d] [worker name %s] [tasks %d]\n", msg, worker_id, worker_name, ntasks);
	if (ntasks > 0)
	{
		struct starpu_codelet *codelet = calloc(1,sizeof(*codelet));
		codelet->cpu_funcs[0] = func_cpu_bis;
		codelet->cpu_funcs_name[0] = "func_cpu_bis";
		int nntasks = ntasks - 1;
		starpu_task_insert(codelet,
				   STARPU_VALUE, &msg, sizeof(msg),
				   STARPU_VALUE, &nntasks, sizeof(ntasks),
				   STARPU_VALUE, &worker_id, sizeof(worker_id),
				   STARPU_CALLBACK_WITH_ARG, free_codelet, codelet,
				   0);
	}
}

void func_cpu(void *descr[], void *_args)
{
	(void)descr;
	char msg;
	char worker_name[256];
	int worker_id = starpu_worker_get_id_check();
	int worker_id_expected;
	int ntasks;
	unsigned sched_ctx_id;
	unsigned *sched_ctx_id_p;

	starpu_worker_get_name(worker_id, worker_name, 256);
	starpu_codelet_unpack_args(_args, &msg, &ntasks, &sched_ctx_id, &worker_id_expected, &sched_ctx_id_p);

	STARPU_ASSERT(worker_id == worker_id_expected);

	*sched_ctx_id_p = sched_ctx_id;
	starpu_sched_ctx_set_context(sched_ctx_id_p);

	FPRINTF(stderr, "[msg '%c'] [worker id %d] [worker name %s] [sched_ctx_id %u] [tasks %d] [buffer %p]\n", msg, worker_id, worker_name, sched_ctx_id, ntasks, sched_ctx_id_p);
	if (ntasks > 0)
	{
		struct starpu_codelet *codelet = calloc(1,sizeof(*codelet));
		codelet->cpu_funcs[0] = func_cpu_bis;
		codelet->cpu_funcs_name[0] = "func_cpu_bis";
		int nntasks = ntasks - 1;
		starpu_task_insert(codelet,
				   STARPU_VALUE, &msg, sizeof(msg),
				   STARPU_VALUE, &nntasks, sizeof(nntasks),
				   STARPU_VALUE, &worker_id, sizeof(worker_id),
				   STARPU_CALLBACK_WITH_ARG, free_codelet, codelet,
				   0);
	}
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.cpu_funcs_name = {"func_cpu"},
};

int main(void)
{
        int i, ret;
	int nprocs, nprocs_per_context=1;
        int procs[STARPU_NMAXWORKERS];
	int ntasks=10;
	char msg[2] = "ab";
	unsigned *buffer[2];

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

        nprocs = starpu_cpu_worker_get_count();
	if (nprocs < 2) goto enodev;

	nprocs_per_context = 1;
	FPRINTF(stderr, "# Workers = %d -> %d worker for each sched context\n", nprocs, nprocs_per_context);
        starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, nprocs);

	unsigned sched_ctx_0 = starpu_sched_ctx_create(procs, nprocs_per_context, "ctx_0", 0);
	unsigned sched_ctx_1 = starpu_sched_ctx_create(&procs[nprocs_per_context], nprocs_per_context, "ctx_1", 0);

	if (!getenv("STARPU_SSILENT"))
	{
		char name0[256];
		char name1[256];

		starpu_worker_get_name(procs[0], name0, 256);
		starpu_worker_get_name(procs[1], name1, 256);

		FPRINTF(stderr, "Creating first sched_ctx with %d worker [id %d name %s]\n", nprocs_per_context, procs[0], name0);
		FPRINTF(stderr, "Creating second sched_ctx with %d worker [id %d name %s]\n", nprocs_per_context, procs[1], name1);

		starpu_sched_ctx_display_workers(sched_ctx_0, stderr);
		starpu_sched_ctx_display_workers(sched_ctx_1, stderr);
	}

	buffer[0] = malloc(sizeof(unsigned));
	buffer[1] = malloc(sizeof(unsigned));
	FPRINTF(stderr, "allocating %p and %p\n", buffer[0], buffer[1]);

	ret = starpu_task_insert(&mycodelet, STARPU_SCHED_CTX, sched_ctx_0,
				 STARPU_VALUE, &msg[0], sizeof(msg[0]),
				 STARPU_VALUE, &ntasks, sizeof(ntasks),
				 STARPU_VALUE, &sched_ctx_0, sizeof(sched_ctx_0),
				 STARPU_VALUE, &procs[0], sizeof(procs[0]),
				 STARPU_VALUE, &buffer[0], sizeof(buffer[0]),
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_insert(&mycodelet, STARPU_SCHED_CTX, sched_ctx_1,
				 STARPU_VALUE, &msg[1], sizeof(msg[1]),
				 STARPU_VALUE, &ntasks, sizeof(ntasks),
				 STARPU_VALUE, &sched_ctx_1, sizeof(sched_ctx_1),
				 STARPU_VALUE, &procs[1], sizeof(procs[1]),
				 STARPU_VALUE, &buffer[1], sizeof(buffer[1]),
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();
	starpu_sched_ctx_delete(sched_ctx_0);
	starpu_sched_ctx_delete(sched_ctx_1);
	starpu_shutdown();
	free(buffer[0]);
	free(buffer[1]);
	return 0;

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
