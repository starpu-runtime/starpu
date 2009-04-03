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
extern mem_node_descr descr;

struct sched_policy_s *get_sched_policy(void)
{
	return &policy;
}

void init_sched_policy(struct machine_config_s *config)
{
	/* eager policy is taken by default */
	char *sched_env;
	sched_env = getenv("SCHED");
	if (sched_env) {
		 if (strcmp(sched_env, "ws") == 0) {
#ifdef VERBOSE
		 	fprintf(stderr, "USE WS SCHEDULER !! \n");
#endif
			policy.init_sched = initialize_ws_policy;
			policy.get_local_queue = get_local_queue_ws;
		 }
		 else if (strcmp(sched_env, "prio") == 0) {
#ifdef VERBOSE
		 	fprintf(stderr, "USE PRIO EAGER SCHEDULER !! \n");
#endif
			policy.init_sched = initialize_eager_center_priority_policy;
			policy.get_local_queue = get_local_queue_eager_priority;
		 }
		 else if (strcmp(sched_env, "no-prio") == 0) {
#ifdef VERBOSE
		 	fprintf(stderr, "USE _NO_ PRIO EAGER SCHEDULER !! \n");
#endif
			policy.init_sched = initialize_no_prio_policy;
			policy.get_local_queue = get_local_queue_no_prio;
		 }
		 else if (strcmp(sched_env, "dm") == 0) {
#ifdef VERBOSE
		 	fprintf(stderr, "USE MODEL SCHEDULER !! \n");
#endif
			policy.init_sched = initialize_dm_policy;
			policy.get_local_queue = get_local_queue_dm;
		 }
		 else if (strcmp(sched_env, "dmda") == 0) {
#ifdef VERBOSE
		 	fprintf(stderr, "USE DATA AWARE MODEL SCHEDULER !! \n");
#endif
			policy.init_sched = initialize_dmda_policy;
			policy.get_local_queue = get_local_queue_dmda;
		 }
		 else if (strcmp(sched_env, "random") == 0) {
#ifdef VERBOSE
		 	fprintf(stderr, "USE RANDOM SCHEDULER !! \n");
#endif
			policy.init_sched = initialize_random_policy;
			policy.get_local_queue = get_local_queue_random;
		 }
		 else {
#ifdef VERBOSE
		 	fprintf(stderr, "USE EAGER SCHEDULER !! \n");
#endif
			/* default scheduler is the eager one */
			policy.init_sched = initialize_eager_center_policy;
			policy.get_local_queue = get_local_queue_eager;
		 }
	}
	else {
#ifdef VERBOSE
	 	fprintf(stderr, "USE EAGER SCHEDULER !! \n");
#endif
		/* default scheduler is the eager one */
		policy.init_sched = initialize_eager_center_policy;
		policy.get_local_queue = get_local_queue_eager;
	}

	pthread_cond_init(&policy.sched_activity_cond, NULL);
	pthread_mutex_init(&policy.sched_activity_mutex, NULL);
	pthread_key_create(&policy.local_queue_key, NULL);
	init_mutex(&descr.attached_queues_mutex);
	descr.total_queues_count = 0;

	policy.init_sched(config, &policy);
}

/* the generic interface that call the proper underlying implementation */
int push_task(job_t task)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	STARPU_ASSERT(queue->push_task);

	return queue->push_task(queue, task);
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

struct job_list_s * pop_every_task_from_queue(struct jobq_s *queue)
{
	STARPU_ASSERT(queue->pop_every_task);

	struct job_list_s *list = queue->pop_every_task(queue);

	return list;
}

struct job_list_s *pop_every_task(void)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	return pop_every_task_from_queue(queue);
}

void wait_on_sched_event(void)
{
	struct jobq_s *q = policy.get_local_queue(&policy);

	pthread_mutex_lock(&q->activity_mutex);

	if (machine_is_running())
		pthread_cond_wait(&q->activity_cond, &q->activity_mutex);

	pthread_mutex_unlock(&q->activity_mutex);
}
