#ifndef __CENTRAL_QUEUES_H__
#define __CENTRAL_QUEUES_H__

#include <core/mechanisms/queues.h>

struct central_jobq_s {
	/* the actual list */
	job_list_t jobq;

	/* the mutex to protect the list */
	thread_mutex_t workq_mutex;

	/* possibly wait on a semaphore when the queue is empty */
	sem_t sem_jobq;
};

void init_central_jobq(struct jobq_s *queue);
void central_push_task(struct jobq_s *queue, job_t task);
void central_push_prio_task(struct jobq_s *queue, job_t task);
struct job_s *central_pop_task(struct jobq_s *queue);

#endif // __CENTRAL_QUEUES_H__
