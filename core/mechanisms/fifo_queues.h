#ifndef __FIFO_QUEUES_H__
#define __FIFO_QUEUES_H__

#include <core/mechanisms/queues.h>

struct fifo_jobq_s {
	/* the actual list */
	job_list_t jobq;

	/* the number of tasks currently in the queue */
	unsigned njobs;

	/* the number of tasks that were processed */
	unsigned nprocessed;

	/* only meaningful if the queue is only used by a single worker */
	double exp_start;
	double exp_end;
	double exp_len;
};

struct jobq_s *create_fifo(void);

int fifo_push_task(struct jobq_s *q, job_t task);
int fifo_push_prio_task(struct jobq_s *q, job_t task);

job_t fifo_pop_task(struct jobq_s *q);
struct job_list_s * fifo_pop_every_task(struct jobq_s *q);
job_t fifo_non_blocking_pop_task(struct jobq_s *q);
job_t fifo_non_blocking_pop_task_if_job_exists(struct jobq_s *q);

void init_fifo_queues_mechanisms(void);


unsigned get_fifo_njobs(struct jobq_s *q);
unsigned get_fifo_nprocessed(struct jobq_s *q);


#endif // __FIFO_QUEUES_H__
