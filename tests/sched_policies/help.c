/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <unistd.h>
#include <starpu.h>
#include <starpu_scheduler.h>
#include "../helper.h"

int main(void)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_sched_policy *default_policy = starpu_sched_get_sched_policy();
	FPRINTF(stderr, "Policy name %s\n", default_policy->policy_name);

	FPRINTF(stderr, "Available policies\n");
	struct starpu_sched_policy **policy;
	for(policy=starpu_sched_get_predefined_policies() ; *policy!=NULL ; policy++)
	{
		struct starpu_sched_policy *p = *policy;
		FPRINTF(stderr, "%-30s\t-> %s\n", p->policy_name, p->policy_description);
	}
	FPRINTF(stderr, "\n");

	starpu_shutdown();
	return 0;
}
