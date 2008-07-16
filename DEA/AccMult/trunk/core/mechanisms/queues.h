#ifndef __QUEUES_H__
#define __QUEUES_H__

#include <core/jobs.h>

struct jobq_s {
	/* a pointer to some queue structure */
	void *queue; 

	/* some methods to manipulate the previous queue */
	void (*push_task)(job_t);
	void (*push_prio_task)(job_t);
	job_t (*pop_task)(void);
};

struct central_jobq_s {
	/* the actual list */
	job_list_t jobq;

	/* the mutex to protect the list */
	thread_mutex_t workq_mutex;

	/* possibly wait on a semaphore when the queue is empty */
	sem_t sem_jobq;
};

#endif // __QUEUES_H__
