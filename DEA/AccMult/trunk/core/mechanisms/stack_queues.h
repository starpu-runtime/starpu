#ifndef __STACK_QUEUES_H__
#define __STACK_QUEUES_H__

#include <core/mechanisms/queues.h>

struct stack_jobq_s {
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

	/* only meaningful if the queue is only used by a single worker */
	double exp_start;
	double exp_end;
	double exp_len;
};

struct jobq_s *create_stack(void);

void stack_push_task(struct jobq_s *q, job_t task);

void stack_push_prio_task(struct jobq_s *q, job_t task);

job_t stack_pop_task(struct jobq_s *q);
job_t stack_non_blocking_pop_task(struct jobq_s *q);
job_t stack_non_blocking_pop_task_if_job_exists(struct jobq_s *q);

void init_stack_queues_mechanisms(void);


unsigned get_stack_njobs(struct jobq_s *q);
unsigned get_stack_nprocessed(struct jobq_s *q);


#endif // __STACK_QUEUES_H__
