#ifndef __QUEUES_H__
#define __QUEUES_H__

#include <core/jobs.h>

struct jobq_s {
	/* a pointer to some queue structure */
	void *queue; 

	/* some methods to manipulate the previous queue */
	void (*push_task)(struct jobq_s *, job_t);
	void (*push_prio_task)(struct jobq_s *, job_t);
	struct job_s* (*pop_task)(struct jobq_s *);
};

#endif // __QUEUES_H__
