/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <pthread.h>

#include <starpu.h>
#include <common/config.h>
#include <core/mechanisms/queues.h>
#include <core/policies/sched_policy.h>
#include <core/policies/no-prio-policy.h>
#include <core/policies/eager-central-policy.h>
#include <core/policies/eager-central-priority-policy.h>
#include <core/policies/work-stealing-policy.h>
#include <core/policies/deque-modeling-policy.h>
#include <core/policies/random-policy.h>
#include <core/policies/deque-modeling-policy-data-aware.h>

static struct sched_policy_s policy;

/*
 *	Predefined policies
 */

#define NPREDEFINED_POLICIES	7

struct sched_policy_s *predefined_policies[NPREDEFINED_POLICIES] = {
	&sched_ws_policy,
	&sched_prio_policy,
	&sched_no_prio_policy,
	&sched_dm_policy,
	&sched_dmda_policy,
	&sched_random_policy,
	&sched_eager_policy
};

struct sched_policy_s *get_sched_policy(void)
{
	return &policy;
}

/*
 *	Methods to initialize the scheduling policy
 */

static void load_sched_policy(struct sched_policy_s *sched_policy)
{
	STARPU_ASSERT(sched_policy);

#ifdef VERBOSE
	if (sched_policy->policy_name)
	{
		fprintf(stderr, "Use %s scheduler", sched_policy->policy_name);

		if (sched_policy->policy_description)
		{
			fprintf(stderr, " (%s)", sched_policy->policy_description);
		}

		fprintf(stderr, "\n");
	}
#endif

	policy.init_sched = sched_policy->init_sched;
	policy.deinit_sched = sched_policy->deinit_sched;
	policy.get_local_queue = sched_policy->get_local_queue;

	pthread_cond_init(&policy.sched_activity_cond, NULL);
	pthread_mutex_init(&policy.sched_activity_mutex, NULL);
	pthread_key_create(&policy.local_queue_key, NULL);
}

static struct sched_policy_s *find_sched_policy_from_name(const char *policy_name)
{

	if (!policy_name)
		return NULL;

	unsigned i;
	for (i = 0; i < NPREDEFINED_POLICIES; i++)
	{
		struct sched_policy_s *p;
		p = predefined_policies[i];
		if (p->policy_name)
		{
			if (strcmp(policy_name, p->policy_name) == 0) {
				/* we found a policy with the requested name */
				return p;
			}
		}
	}

	/* nothing was found */
	return NULL;
}

static void display_sched_help_message(void)
{
	const char *sched_env = getenv("SCHED");
	if (sched_env && (strcmp(sched_env, "help") == 0)) {
		fprintf(stderr, "SCHED can be either of\n");

		/* display the description of all predefined policies */
		unsigned i;
		for (i = 0; i < NPREDEFINED_POLICIES; i++)
		{
			struct sched_policy_s *p;
			p = predefined_policies[i];
			fprintf(stderr, "%s\t-> %s\n", p->policy_name, p->policy_description);
		}
	 }
}

static struct sched_policy_s *select_sched_policy(struct machine_config_s *config)
{
	struct sched_policy_s *selected_policy = NULL;
	struct starpu_conf *user_conf = config->user_conf;

	/* First, we check whether the application explicitely gave a scheduling policy or not */
	if (user_conf && (user_conf->sched_policy))
		return user_conf->sched_policy;

	/* Otherwise, we look if the application specified the name of a policy to load */
	const char *sched_pol_name;
	if (user_conf && (user_conf->sched_policy_name))
	{
		sched_pol_name = user_conf->sched_policy_name;
	}
	else {
		sched_pol_name = getenv("SCHED");
	}

	if (sched_pol_name)
		selected_policy = find_sched_policy_from_name(sched_pol_name);

	/* Perhaps there was no policy that matched the name */
	if (selected_policy)
		return selected_policy;

	/* If no policy was specified, we use the greedy policy as a default */
	return &sched_eager_policy;
}

void init_sched_policy(struct machine_config_s *config)
{
	/* Perhaps we have to display some help */
	display_sched_help_message();

	struct sched_policy_s *selected_policy;
	selected_policy = select_sched_policy(config);

	load_sched_policy(selected_policy);

	policy.init_sched(config, &policy);
}

void deinit_sched_policy(struct machine_config_s *config)
{
	if (policy.deinit_sched)
		policy.deinit_sched(config, &policy);

	pthread_key_delete(policy.local_queue_key);
	pthread_mutex_destroy(&policy.sched_activity_mutex);
	pthread_cond_destroy(&policy.sched_activity_cond);
}

/* the generic interface that call the proper underlying implementation */
int push_task(job_t j)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	/* in case there is no codelet associated to the task (that's a control
	 * task), we directly execute its callback and enforce the
	 * corresponding dependencies */
	if (j->task->cl == NULL)
	{
		_starpu_handle_job_termination(j);
		return 0;
	}

	if (STARPU_UNLIKELY(j->task->execute_on_a_specific_worker))
	{
		struct worker_s *worker = _starpu_get_worker_struct(j->task->workerid);
		return _starpu_push_local_task(worker, j);
	}
	else {
		STARPU_ASSERT(queue->push_task);

		return queue->push_task(queue, j);
	}
}

struct job_s * pop_task_from_queue(struct jobq_s *queue)
{
	STARPU_ASSERT(queue->pop_task);

	struct job_s *j = queue->pop_task(queue);

	return j;
}

struct job_s * pop_task(void)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	return pop_task_from_queue(queue);
}

struct job_list_s * pop_every_task_from_queue(struct jobq_s *queue, uint32_t where)
{
	STARPU_ASSERT(queue->pop_every_task);

	struct job_list_s *list = queue->pop_every_task(queue, where);

	return list;
}

/* pop every task that can be executed on "where" (eg. GORDON) */
struct job_list_s *pop_every_task(uint32_t where)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	return pop_every_task_from_queue(queue, where);
}

void wait_on_sched_event(void)
{
	struct jobq_s *q = policy.get_local_queue(&policy);

	pthread_mutex_lock(&q->activity_mutex);

	handle_all_pending_node_data_requests(get_local_memory_node());

	if (_starpu_machine_is_running())
	{
#ifndef NON_BLOCKING_DRIVERS
		pthread_cond_wait(&q->activity_cond, &q->activity_mutex);
#endif
	}

	pthread_mutex_unlock(&q->activity_mutex);
}
