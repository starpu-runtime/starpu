/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016  Inria
 * Copyright (C) 2017  CNRS
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

#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_scheduler.h>

#include <starpu_mpi_lb.h>
#include "policy/load_balancer_policy.h"

static struct load_balancer_policy *defined_policy = NULL;
static void (*saved_post_exec_hook)(struct starpu_task *task, unsigned sched_ctx_id) = NULL;

static void post_exec_hook_wrapper(struct starpu_task *task, unsigned sched_ctx_id)
{
	//fprintf(stderr,"I am called ! \n");
	if (defined_policy && defined_policy->finished_task_entry_point)
		defined_policy->finished_task_entry_point();
	if (saved_post_exec_hook)
		saved_post_exec_hook(task, sched_ctx_id);
}

static struct load_balancer_policy *predefined_policies[] =
{
	&load_heat_propagation_policy,
	NULL
};

void starpu_mpi_lb_init(struct starpu_mpi_lb_conf *itf)
{
	const char *policy_name = starpu_getenv("STARPU_MPI_LB");
	if (!policy_name && itf)
		policy_name = itf->name;

	if (!policy_name || (strcmp(policy_name, "help") == 0))
	{
		fprintf(stderr,"Warning : load balancing is disabled for this run.\n");
		fprintf(stderr,"Use the STARPU_MPI_LB = <name> environment variable to use a load balancer.\n");
		fprintf(stderr,"Available load balancers :\n");
		struct load_balancer_policy **policy;
		for(policy=predefined_policies ; *policy!=NULL ; policy++)
		{
			struct load_balancer_policy *p = *policy;
			fprintf(stderr," - %s\n", p->policy_name);
		}
		return;
	}

	if (policy_name)
	{
		struct load_balancer_policy **policy;
		for(policy=predefined_policies ; *policy!=NULL ; policy++)
		{
			struct load_balancer_policy *p = *policy;
			if (p->policy_name)
			{
				if (strcmp(policy_name, p->policy_name) == 0)
				{
					/* we found a policy with the requested name */
					defined_policy = p;
					break;
				}
			}
		}
	}

	if (!defined_policy)
	{
		fprintf(stderr,"Error : no load balancer with the name %s. Load balancing will be disabled for this run.\n", policy_name);
		return;
	}

	if (defined_policy->init(itf) != 0)
	{
		fprintf(stderr,"Error in load_balancer->init: invalid starpu_mpi_lb_conf. Load balancing will be disabled for this run.\n");
		return;
	}

	/* starpu_register_hook(submitted_task, defined_policy->submitted_task_entry_point); */
	if (defined_policy->submitted_task_entry_point)
		starpu_mpi_pre_submit_hook_register(defined_policy->submitted_task_entry_point);

	/* starpu_register_hook(finished_task, defined_policy->finished_task_entry_point); */
	if (defined_policy->finished_task_entry_point)
	{
		STARPU_ASSERT(saved_post_exec_hook == NULL);
		struct starpu_sched_policy **predefined_sched_policies = starpu_sched_get_predefined_policies();
		struct starpu_sched_policy **sched_policy;
		const char *sched_policy_name = starpu_getenv("STARPU_SCHED");

		if (!sched_policy_name)
			sched_policy_name = "eager";

		for(sched_policy=predefined_sched_policies ; *sched_policy!=NULL ; sched_policy++)
		{
			struct starpu_sched_policy *sched_p = *sched_policy;
			if (strcmp(sched_policy_name, sched_p->policy_name) == 0)
			{
				/* We found the scheduling policy with the requested name */
				saved_post_exec_hook = sched_p->post_exec_hook;
				break;
			}
		}
		starpu_sched_policy_set_post_exec_hook(post_exec_hook_wrapper, sched_policy_name);
	}
}

void starpu_mpi_lb_shutdown()
{
	if (!defined_policy)
		return;

	if (defined_policy && defined_policy->deinit())
		return;

	/* starpu_unregister_hook(submitted_task, defined_policy->submitted_task_entry_point); */
	if (defined_policy->submitted_task_entry_point)
		starpu_mpi_pre_submit_hook_unregister();

	/* starpu_unregister_hook(finished_task, defined_policy->finished_task_entry_point); */
	if (defined_policy->finished_task_entry_point && saved_post_exec_hook != NULL)
	{
		struct starpu_sched_policy **predefined_sched_policies = starpu_sched_get_predefined_policies();
		struct starpu_sched_policy **sched_policy;
		const char *sched_policy_name = starpu_getenv("STARPU_SCHED");

		if (!sched_policy_name)
			sched_policy_name = "eager";

		for(sched_policy=predefined_sched_policies ; *sched_policy!=NULL ; sched_policy++)
		{
			struct starpu_sched_policy *sched_p = *sched_policy;
			if (strcmp(sched_policy_name, sched_p->policy_name) == 0)
			{
				/* We found the scheduling policy with the requested name */
				sched_p->post_exec_hook = saved_post_exec_hook;
				saved_post_exec_hook = NULL;
				break;
			}
		}
	}
	STARPU_ASSERT(saved_post_exec_hook == NULL);
	defined_policy = NULL;
}
