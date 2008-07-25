#include <core/mechanisms/queues.h>
#include <core/policies/sched_policy.h>
#include <core/policies/eager-central-policy.h>
#include <core/policies/eager-central-priority-policy.h>
#include <core/policies/work-stealing-policy.h>

static struct sched_policy_s policy;

void init_sched_policy(struct machine_config_s *config)
{
	/* first get the proper policy XXX */
	/* for now we hardcode the eager policy */
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


	pthread_key_create(&policy.local_queue_key, NULL);

	policy.init_sched(config, &policy);
}

void set_local_queue(struct jobq_s *jobq)
{
	//printf("set local queue %p \n", jobq);
	/* record the queue that was specified to the driver */
	pthread_setspecific(policy.local_queue_key, jobq);
}

/* the generic interface that call the proper underlying implementation */
void push_task(job_t task)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	ASSERT(queue->push_task);

	queue->push_task(queue, task);
}

void push_prio_task(job_t task)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	ASSERT(queue->push_prio_task);

	queue->push_prio_task(queue, task);
}

struct job_s * pop_task(void)
{
	struct jobq_s *queue = policy.get_local_queue(&policy);

	ASSERT(queue->pop_task);

	struct job_s *j = queue->pop_task(queue);

	return j;
}
