/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(void)
{
	int ret;
	int nprocs;
	int *procs;

	unsigned sched_ctx;
	unsigned main_sched_ctx;
	int *ptr;
	int *main_ptr;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	nprocs = starpu_cpu_worker_get_count();
	if (nprocs == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	procs = (int*)malloc(nprocs*sizeof(int));
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, nprocs);

	sched_ctx = starpu_sched_ctx_create(procs, nprocs, "my_context", 0);
	ptr = starpu_sched_ctx_get_policy_data(sched_ctx);
	STARPU_ASSERT_MSG(ptr == NULL, "The policy data for the sched ctx should be NULL\n");

	starpu_sched_ctx_set_policy_data(sched_ctx, procs);
	ptr = starpu_sched_ctx_get_policy_data(sched_ctx);
	FPRINTF(stderr, "sched_ctx %u : data %p (procs %p)\n", sched_ctx, ptr, procs);
	STARPU_ASSERT_MSG(ptr == procs, "The policy data for the sched ctx is incorrect\n");

	main_sched_ctx = starpu_sched_ctx_get_context();
	main_ptr = starpu_sched_ctx_get_policy_data(main_sched_ctx);
	STARPU_ASSERT_MSG(main_ptr == NULL, "The policy data for the sched ctx should be NULL\n");

	starpu_sched_ctx_set_policy_data(main_sched_ctx, procs);
	main_ptr = starpu_sched_ctx_get_policy_data(sched_ctx);
	FPRINTF(stderr, "sched_ctx %u : data %p (procs %p)\n", main_sched_ctx, main_ptr, procs);
	STARPU_ASSERT_MSG(main_ptr == procs, "The policy data for the sched ctx is incorrect\n");

	starpu_sched_ctx_delete(sched_ctx);
	free(procs);
	starpu_shutdown();

	return (ptr == procs) ? 0 : 1;
}
