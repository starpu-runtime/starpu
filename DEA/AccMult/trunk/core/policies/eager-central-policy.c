#include <core/policies/eager-central-policy.h>

/*
 *	This is just the trivial policy where every worker use the same
 *	job queue.
 */

/* the former is the actual queue, the latter some container */
static struct central_jobq_s central_jobq;
static struct jobq_s jobq;

void initialize_eager_center_policy(struct machine_config_s *config 
						__attribute__ ((unused)))
{
	jobq.queue = &central_jobq;

	init_central_jobq(&jobq);

	jobq.push_task = central_push_task;
	jobq.push_prio_task = central_push_prio_task;
	jobq.pop_task = central_pop_task;
}

void set_local_queue_eager(struct jobq_s *jobq __attribute__ ((unused)))
{
	/* this is not needed in that policy */
}

struct jobq_s *get_local_queue_eager(void)
{
	/* this is trivial for that strategy :) */
	return &jobq;
}
