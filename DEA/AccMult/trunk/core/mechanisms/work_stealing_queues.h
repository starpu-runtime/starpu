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

	/* the number of tasks that were processed */
	unsigned nprocessed;
};

void init_ws_queues_mechanisms(void);
struct jobq_s *create_deque(void);

void ws_push_task(struct jobq_s *q, job_t task);
void ws_push_prio_task(struct jobq_s *q, job_t task);
job_t ws_non_blocking_pop_task(struct jobq_s *q);
job_t ws_non_blocking_pop_task_if_job_exists(struct jobq_s *q);

unsigned get_queue_njobs_ws(struct jobq_s *q);
unsigned get_queue_nprocessed_ws(struct jobq_s *q);
unsigned get_total_njobs_ws(void);

#endif // __WORK_STEALING_QUEUES_H__
