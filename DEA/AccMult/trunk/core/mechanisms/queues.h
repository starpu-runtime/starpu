#ifndef __QUEUES_H__
#define __QUEUES_H__

#include <core/jobs.h>
#include <core/policies/sched_policy.h>

struct jobq_s {
	/* a pointer to some queue structure */
	void *queue; 

	/* some methods to manipulate the previous queue */
	void (*push_task)(struct jobq_s *, job_t);
	void (*push_prio_task)(struct jobq_s *, job_t);
	struct job_s* (*pop_task)(struct jobq_s *);

	/* what are the driver that may pop job from that queue ? */
	uint32_t who;
};

struct machine_config_s;

void setup_queues(void (*init_queue_design)(void),
                  struct jobq_s *(*func_init_queue)(void),
                  struct machine_config_s *config);

struct jobq_s *get_local_queue(void);
void set_local_queue(struct jobq_s *jobq);


#endif // __QUEUES_H__
