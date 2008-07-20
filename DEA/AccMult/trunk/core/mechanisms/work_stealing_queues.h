#ifndef __WORK_STEALING_QUEUES_H__
#define __WORK_STEALING_QUEUES_H__

#include <core/mechanisms/queues.h>

struct deque_jobq_s {
	/* the actual list */
	job_list_t jobq;

	/* the mutex to protect the list */
	thread_mutex_t workq_mutex;

	/* the number of tasks currently in the queue */
	unsigned njobs;
};

struct jobq_s *create_deque(void);
void ws_push_task(struct jobq_s *q, job_t task);
job_t ws_non_blocking_pop_task(struct jobq_s *q);

#endif // __WORK_STEALING_QUEUES_H__
