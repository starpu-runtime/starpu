#include <pthread.h>

#include <core/mechanisms/queues.h>
#include <core/policies/sched_policy.h>
#include <core/policies/no-prio-policy.h>
#include <core/policies/eager-central-policy.h>
#include <core/policies/eager-central-priority-policy.h>
#include <core/policies/work-stealing-policy.h>
#include <core/policies/deque-modeling-policy.h>
#include <core/policies/random-policy.h>


static struct sched_policy_s policy;

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
		 	fprintf(stderr, "USE WS SCHEDULER !! \n");
			policy.init_sched = initialize_ws_policy;
			policy.get_local_queue = get_local_queue_ws;
		 }
		 else if (strcmp(sched_env, "prio") == 0) {
		 	fprintf(stderr, "USE PRIO EAGER SCHEDULER !! \n");
			policy.init_sched = initialize_eager_center_priority_policy;
			policy.get_local_queue = get_local_queue_eager_priority;
		 }
		 else if (strcmp(sched_env, "no-prio") == 0) {
		 	fprintf(stderr, "USE _NO_ PRIO EAGER SCHEDULER !! \n");
			policy.init_sched = initialize_no_prio_policy;
			policy.get_local_queue = get_local_queue_no_prio;
		 }
		 else if (strcmp(sched_env, "dm") == 0) {
		 	fprintf(stderr, "USE MODEL SCHEDULER !! \n");
			policy.init_sched = initialize_dm_policy;
			policy.get_local_queue = get_local_queue_dm;
		 }
		 else if (strcmp(sched_env, "random") == 0) {
		 	fprintf(stderr, "USE RANDOM SCHEDULER !! \n");
			policy.init_sched = initialize_random_policy;
			policy.get_local_queue = get_local_queue_random;
		 }
		 else {
		 	fprintf(stderr, "USE EAGER SCHEDULER !! \n");
			/* default scheduler is the eager one */
			policy.init_sched = initialize_eager_center_policy;
			policy.get_local_queue = get_local_queue_eager;
		 }
	}
	else {
		 	fprintf(stderr, "USE EAGER SCHEDULER !! \n");
		/* default scheduler is the eager one */
		policy.init_sched = initialize_eager_center_policy;
		policy.get_local_queue = get_local_queue_eager;
	}

	pthread_cond_init(&policy.sched_activity_cond, NULL);
	pthread_mutex_init(&policy.sched_activity_mutex, NULL);
	pthread_key_create(&policy.local_queue_key, NULL);

	policy.init_sched(config, &policy);
}

/* the generic interface that call the proper underlying implementation */
int push_task(job_t task)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	STARPU_ASSERT(queue->push_task);

	if (!worker_exists(task->where))
		return -ENODEV; 

	return queue->push_task(queue, task);
}

int push_prio_task(job_t task)
{
	task->priority = MAX_PRIO;
	
	return push_task(task);
}

/* note that this call is blocking, and will not make StarPU progress,
 * so it must only be called from the programmer thread, not by StarPU */
int push_task_sync(job_t task)
{
	int ret;

	task->synchronous = 1;
	sem_init(&task->sync_sem, 0, 0);

	ret = push_task(task);
	if (ret == -ENODEV)
	{	
		sem_destroy(&task->sync_sem);
		return ret;
	}

	sem_wait(&task->sync_sem);
	sem_destroy(&task->sync_sem);

	return 0;
}

struct job_s * pop_task(void)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	STARPU_ASSERT(queue->pop_task);

	struct job_s *j = queue->pop_task(queue);

	return j;
}

struct job_list_s *pop_every_task(void)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	STARPU_ASSERT(queue->pop_every_task);

	struct job_list_s *list = queue->pop_every_task(queue);

	return list;
}

void wait_on_sched_event(void)
{
	struct jobq_s *q = policy.get_local_queue(&policy);

	pthread_mutex_lock(&q->activity_mutex);
	pthread_cond_wait(&q->activity_cond, &q->activity_mutex);
	pthread_mutex_unlock(&q->activity_mutex);
}
