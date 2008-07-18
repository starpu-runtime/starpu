#include <core/mechanisms/queues.h>
#include <core/policies/sched_policy.h>
#include <core/policies/eager-central-policy.h>

static struct sched_policy_s policy;

void init_sched_policy(struct machine_config_s *config)
{
	/* first get the proper policy XXX */
	/* for now we hardcode the eager policy */
	policy.init_sched = initialize_eager_center_policy;
	policy.get_local_queue = get_local_queue_eager;
	policy.set_local_queue = set_local_queue_eager;

	policy.init_sched(config);
}

/* the generic interface that call the proper underlying implementation */
void push_task(job_t task)
{
	struct jobq_s *queue = policy.get_local_queue();

	ASSERT(queue->push_task);

	queue->push_task(queue, task);
}

void push_prio_task(job_t task)
{
	struct jobq_s *queue = policy.get_local_queue();

	ASSERT(queue->push_prio_task);

	queue->push_prio_task(queue, task);
}

struct job_s * pop_task(void)
{
	struct jobq_s *queue = policy.get_local_queue();

	ASSERT(queue->pop_task);

	struct job_s *j = queue->pop_task(queue);

	return j;
}
