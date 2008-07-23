#include <core/mechanisms/central_queues_priorities.h>

/*
 * Centralized queue with priorities 
 */

void init_central_priority_jobq(struct jobq_s *q)
{
	ASSERT(q);

	struct central_priority_jobq_s *central_queue;
	
	central_queue = malloc(sizeof(struct central_priority_jobq_s));

	q->queue = central_queue;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		central_queue->jobq[prio] = job_list_new();
		central_queue->njobs[prio] = 0;
	}

	thread_mutex_init(&central_queue->workq_mutex, NULL);
	sem_init(&central_queue->sem_jobq, 0, 0);
}

void central_priority_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct central_priority_jobq_s *central_queue = q->queue;

	thread_mutex_lock(&central_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 1);
	
	unsigned priolevel = task->priority - MIN_PRIO;

	job_list_push_back(central_queue->jobq[priolevel], task);
	central_queue->njobs[priolevel]++;

	sem_post(&central_queue->sem_jobq);

	thread_mutex_unlock(&central_queue->workq_mutex);
}

job_t central_priority_pop_task(struct jobq_s *q)
{
	job_t j = NULL;

	ASSERT(q);
	struct central_priority_jobq_s *central_queue = q->queue;

	sem_wait(&central_queue->sem_jobq);

	thread_mutex_lock(&central_queue->workq_mutex);

	unsigned priolevel = NPRIO_LEVELS - 1;
	do {
		if (central_queue->njobs[priolevel] > 0) {
			/* there is some task that we can grab */
			j = job_list_pop_back(central_queue->jobq[priolevel]);
			central_queue->njobs[priolevel]--;
			TRACE_JOB_POP(j, 0);
		}
	} while (!j && priolevel-- > 0);

	thread_mutex_unlock(&central_queue->workq_mutex);

	return j;
}
