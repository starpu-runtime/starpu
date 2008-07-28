#ifndef __DEQUE_QUEUES_H__
#define __DEQUE_QUEUES_H__

#include <core/mechanisms/queues.h>

struct deque_jobq_s {
	/* the actual list */
	job_list_t jobq;

	/* the mutex to protect the list */
	thread_mutex_t workq_mutex;

	/* the number of tasks currently in the queue */
	unsigned njobs;

	/* the number of tasks that were processed */
	unsigned nprocessed;

	/* possibly wait on a semaphore when the queue is empty */
	sem_t sem_jobq;
};

struct jobq_s *create_deque(void);

void deque_push_task(struct jobq_s *q, job_t task);

void deque_push_prio_task(struct jobq_s *q, job_t task);

job_t deque_pop_task(struct jobq_s *q);
job_t deque_non_blocking_pop_task(struct jobq_s *q);
job_t deque_non_blocking_pop_task_if_job_exists(struct jobq_s *q);

void init_deque_queues_mechanisms(void);


unsigned get_deque_njobs(struct jobq_s *q);
unsigned get_deque_nprocessed(struct jobq_s *q);


#endif // __DEQUE_QUEUES_H__
