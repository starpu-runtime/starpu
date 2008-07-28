#include <core/policies/eager-central-policy.h>

/*
 *	This is just the trivial policy where every worker use the same
 *	JOB QUEUE.
 */

/* the former is the actual queue, the latter some container */
static struct jobq_s *jobq;

static void init_central_queue_design(void)
{
	/* there is only a single queue in that trivial design */
	jobq = create_deque();

	init_deque_queues_mechanisms();

	jobq->push_task = deque_push_task;
	jobq->push_prio_task = deque_push_prio_task;
	jobq->pop_task = deque_pop_task;
}

static struct jobq_s *func_init_central_queue(void)
{
	/* once again, this is trivial */
	return jobq;
}

void initialize_eager_center_policy(struct machine_config_s *config, 
	   __attribute__ ((unused)) struct sched_policy_s *_policy) 
{
	setup_queues(init_central_queue_design, func_init_central_queue, config);
}

struct jobq_s *get_local_queue_eager(struct sched_policy_s *policy 
					__attribute__ ((unused)))
{
	/* this is trivial for that strategy :) */
	return jobq;
}


