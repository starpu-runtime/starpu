#include <core/mechanisms/priority_queues.h>

/*
 * Centralized queue with priorities 
 */

struct jobq_s *create_priority_jobq(void)
{
	struct jobq_s *q;

	q = malloc(sizeof(struct jobq_s));

	struct priority_jobq_s *central_queue;
	
	central_queue = malloc(sizeof(struct priority_jobq_s));

	q->queue = central_queue;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		central_queue->jobq[prio] = job_list_new();
		central_queue->njobs[prio] = 0;
	}

	thread_mutex_init(&central_queue->workq_mutex, NULL);
	sem_init(&central_queue->sem_jobq, 0, 0);

	return q;
}

void priority_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct priority_jobq_s *queue = q->queue;

	thread_mutex_lock(&queue->workq_mutex);

	TRACE_JOB_PUSH(task, 1);
	
	unsigned priolevel = task->priority - MIN_PRIO;

	job_list_push_front(queue->jobq[priolevel], task);
	queue->njobs[priolevel]++;

	sem_post(&queue->sem_jobq);

	thread_mutex_unlock(&queue->workq_mutex);
}

job_t priority_pop_task(struct jobq_s *q)
{
	job_t j = NULL;

	ASSERT(q);
	struct priority_jobq_s *queue = q->queue;

	/* block until a job is found */
	sem_wait(&queue->sem_jobq);

	thread_mutex_lock(&queue->workq_mutex);

	unsigned priolevel = NPRIO_LEVELS - 1;
	do {
		if (queue->njobs[priolevel] > 0) {
			/* there is some task that we can grab */
			j = job_list_pop_back(queue->jobq[priolevel]);
			queue->njobs[priolevel]--;
			TRACE_JOB_POP(j, 0);
		}
	} while (!j && priolevel-- > 0);

	thread_mutex_unlock(&queue->workq_mutex);

	return j;
}
