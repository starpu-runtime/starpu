#ifndef __CENTRAL_QUEUES_PRIORITIES_H__
#define __CENTRAL_QUEUES_PRIORITIES_H__

#define MIN_PRIO	(-4)
#define MAX_PRIO	5

#define NPRIO_LEVELS	((MAX_PRIO) - (MIN_PRIO) + 1)

#include <core/mechanisms/queues.h>

struct central_priority_jobq_s {
	/* the actual lists 
	 *	jobq[p] is for priority [p - MIN_PRIO] */
	job_list_t jobq[NPRIO_LEVELS];
	unsigned njobs[NPRIO_LEVELS];

	/* the mutex to protect the list */
	thread_mutex_t workq_mutex;

	/* possibly wait on a semaphore when the queue is empty */
	sem_t sem_jobq;
};

void init_central_priority_jobq(struct jobq_s *queue);
void central_priority_push_task(struct jobq_s *queue, job_t task);
struct job_s *central_priority_pop_task(struct jobq_s *queue);

#endif // __CENTRAL_QUEUES_PRIORITIES_H__
