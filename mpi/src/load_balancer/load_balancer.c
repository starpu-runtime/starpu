/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/utils.h>
#include <common/config.h>

#include <starpu_mpi_lb.h>
#include "policy/load_balancer_policy.h"

#if defined(STARPU_USE_MPI_MPI)

static struct load_balancer_policy *defined_policy = NULL;
typedef void (*_post_exec_hook_func_t)(struct starpu_task *task, unsigned sched_ctx_id);
static _post_exec_hook_func_t saved_post_exec_hook[STARPU_NMAX_SCHED_CTXS];

static void post_exec_hook_wrapper(struct starpu_task *task, unsigned sched_ctx_id)
{
	//fprintf(stderr,"I am called ! \n");
	if (defined_policy && defined_policy->finished_task_entry_point)
		defined_policy->finished_task_entry_point();
	if (saved_post_exec_hook[sched_ctx_id])
		saved_post_exec_hook[sched_ctx_id](task, sched_ctx_id);
}

static struct load_balancer_policy *predefined_policies[] =
{
	&load_heat_propagation_policy,
	NULL
};

void starpu_mpi_lb_init(const char *lb_policy_name, struct starpu_mpi_lb_conf *itf)
{
	int ret;

	const char *policy_name = starpu_getenv("STARPU_MPI_LB");
	if (!policy_name)
		policy_name = lb_policy_name;

	if (!policy_name || (strcmp(policy_name, "help") == 0))
	{
		_STARPU_MSG("Warning : load balancing is disabled for this run.\n");
		_STARPU_MSG("Use the STARPU_MPI_LB = <name> environment variable to use a load balancer.\n");
		_STARPU_MSG("Available load balancers :\n");
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
		_STARPU_MSG("Error : no load balancer with the name %s. Load balancing will be disabled for this run.\n", policy_name);
		return;
	}

	ret = defined_policy->init(itf);
	if (ret != 0)
	{
		_STARPU_MSG("Error (%d) in %s->init: invalid starpu_mpi_lb_conf. Load balancing will be disabled for this run.\n", ret, defined_policy->policy_name);
		return;
	}

	/* starpu_register_hook(submitted_task, defined_policy->submitted_task_entry_point); */
	if (defined_policy->submitted_task_entry_point)
		starpu_mpi_pre_submit_hook_register(defined_policy->submitted_task_entry_point);

	/* starpu_register_hook(finished_task, defined_policy->finished_task_entry_point); */
	if (defined_policy->finished_task_entry_point)
	{
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			struct starpu_sched_policy *sched_policy = starpu_sched_ctx_get_sched_policy(i);
			if (sched_policy)
			{
				_STARPU_DEBUG("Setting post_exec_hook for scheduling context %d %s (%d)\n", i, sched_policy->policy_name, STARPU_NMAX_SCHED_CTXS);
				saved_post_exec_hook[i] = sched_policy->post_exec_hook;
				sched_policy->post_exec_hook = post_exec_hook_wrapper;
			}
			else
				saved_post_exec_hook[i] = NULL;
		}
	}

	return;
}

void starpu_mpi_lb_shutdown()
{
	if (!defined_policy)
		return;

	int ret = defined_policy->deinit();
	if (ret != 0)
	{
		_STARPU_MSG("Error (%d) in %s->deinit\n", ret, defined_policy->policy_name);
		return;
	}

	/* starpu_unregister_hook(submitted_task, defined_policy->submitted_task_entry_point); */
	if (defined_policy->submitted_task_entry_point)
		starpu_mpi_pre_submit_hook_unregister();

	/* starpu_unregister_hook(finished_task, defined_policy->finished_task_entry_point); */
	if (defined_policy->finished_task_entry_point)
	{
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			if (saved_post_exec_hook[i])
			{
				struct starpu_sched_policy *sched_policy = starpu_sched_ctx_get_sched_policy(i);
				sched_policy->post_exec_hook = saved_post_exec_hook[i];
				saved_post_exec_hook[i] = NULL;
			}
		}
	}
	defined_policy = NULL;
}

#endif /* STARPU_USE_MPI_MPI */
