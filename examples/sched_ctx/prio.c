/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(void)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned sched_ctx1 = starpu_sched_ctx_create(NULL, -1, "ctx1", STARPU_SCHED_CTX_POLICY_NAME, "prio", 0);

	FPRINTF(stderr, "min prio %d\n", starpu_sched_ctx_get_min_priority(sched_ctx1));
	FPRINTF(stderr, "max prio %d\n", starpu_sched_ctx_get_max_priority(sched_ctx1));

	unsigned sched_ctx2 = starpu_sched_ctx_create(NULL, -1, "ctx2",
						      STARPU_SCHED_CTX_POLICY_NAME, "prio",
						      STARPU_SCHED_CTX_POLICY_MIN_PRIO, -12,
						      STARPU_SCHED_CTX_POLICY_MAX_PRIO, 32,
						      0);

	FPRINTF(stderr, "min prio %d\n", starpu_sched_ctx_get_min_priority(sched_ctx2));
	FPRINTF(stderr, "max prio %d\n", starpu_sched_ctx_get_max_priority(sched_ctx2));

	if (starpu_sched_ctx_get_min_priority(sched_ctx2) != -12)
	{
		FPRINTF(stderr, "Error with min priority: %d != %d\n", starpu_sched_ctx_get_min_priority(sched_ctx2), -12);
		ret = 1;
	}
	if (starpu_sched_ctx_get_max_priority(sched_ctx2) != 32)
	{
		FPRINTF(stderr, "Error with max priority: %d != %d\n", starpu_sched_ctx_get_max_priority(sched_ctx2), 32);
		ret = 1;
	}

	starpu_sched_ctx_delete(sched_ctx1);
	starpu_sched_ctx_delete(sched_ctx2);

	starpu_shutdown();

	return ret;
}
