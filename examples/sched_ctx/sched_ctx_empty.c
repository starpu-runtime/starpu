/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static void cpu_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	FPRINTF(stdout, "Hello world\n");
}

static struct starpu_codelet codelet =
{
	.cpu_funcs = {cpu_func},
	.nbuffers = 0,
	.name = "codelet"
};

int main(void)
{
	int ret;
	int nprocs = 0;
	int procs[STARPU_NMAXWORKERS];
	unsigned sched_ctx_id;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	nprocs = starpu_cpu_worker_get_count();
	// if there is no cpu, skip
	if (nprocs == 0) goto enodev;

	sched_ctx_id = starpu_sched_ctx_create(NULL, 0, "ctx", 0);
	starpu_sched_ctx_set_context(&sched_ctx_id);

	ret = starpu_task_insert(&codelet, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, nprocs);
	starpu_sched_ctx_add_workers(procs, nprocs, sched_ctx_id);
	starpu_task_wait_for_all();

	starpu_sched_ctx_delete(sched_ctx_id);

enodev:
	starpu_shutdown();
	return nprocs == 0 ? 77 : 0;
}
