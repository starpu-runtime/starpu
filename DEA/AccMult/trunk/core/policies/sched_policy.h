#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <core/mechanisms/queues.h>
#include <core/workers.h>

struct machine_config_s;

struct sched_policy_s {
	/* create all the queues */
	void (*init_sched)(struct machine_config_s *);

	/* anyone can request which queue it is associated to */
	struct jobq_s *(*get_local_queue)(void);
};

void init_sched_policy(struct machine_config_s *config);
void set_local_queue(struct jobq_s *jobq);

void push_task(job_t task);
void push_prio_task(job_t task);
struct job_s *pop_task(void);
void push_task(job_t task);

#endif // __SCHED_POLICY_H__
