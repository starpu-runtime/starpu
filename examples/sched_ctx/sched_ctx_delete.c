/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>

int main(void)
{
	int ret;
	int nprocs = 0;
	int procs[STARPU_NMAXWORKERS];
	unsigned sched_ctx1, sched_ctx2;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_worker_get_count_by_type(STARPU_CPU_WORKER) == 0)
	{
		// Needs at least 1 CPU worker
		starpu_shutdown();
		return 77;
	}

	nprocs = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, nprocs);

	sched_ctx1 = starpu_sched_ctx_create(procs, nprocs, "ctx1", 0);
	sched_ctx2 = starpu_sched_ctx_create(procs, nprocs, "ctx2", 0);

	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);

	starpu_sched_ctx_delete(sched_ctx1);
	starpu_sched_ctx_delete(sched_ctx2);

	starpu_shutdown();
	return 0;
}
