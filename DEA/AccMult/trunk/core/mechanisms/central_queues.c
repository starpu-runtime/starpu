#include <core/mechanisms/central_queues.h>

/*
 * Centralized queue 
 */

void init_central_jobq(struct jobq_s *q)
{
	ASSERT(q);

	struct central_jobq_s *central_queue;
	
	central_queue = malloc(sizeof(struct central_jobq_s));

	q->queue = central_queue;

	thread_mutex_init(&central_queue->workq_mutex, NULL);
	central_queue->jobq = job_list_new();
	sem_init(&central_queue->sem_jobq, 0, 0);
}

void central_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct central_jobq_s *central_queue = q->queue;

	thread_mutex_lock(&central_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(central_queue->jobq, task);
	sem_post(&central_queue->sem_jobq);

	thread_mutex_unlock(&central_queue->workq_mutex);
}

void central_push_prio_task(struct jobq_s *q, job_t task)
{
#ifndef NO_PRIO
	ASSERT(q);
	struct central_jobq_s *central_queue = q->queue;

	thread_mutex_lock(&central_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 1);
	job_list_push_back(central_queue->jobq, task);
	sem_post(&central_queue->sem_jobq);

	thread_mutex_unlock(&central_queue->workq_mutex);
#else
	central_push_task(q, task);
#endif
}

job_t central_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct central_jobq_s *central_queue = q->queue;

	sem_wait(&central_queue->sem_jobq);

	thread_mutex_lock(&central_queue->workq_mutex);

	if (job_list_empty(central_queue->jobq)) {
		thread_mutex_unlock(&central_queue->workq_mutex);
		return NULL;
	}

	j = job_list_pop_back(central_queue->jobq);
	TRACE_JOB_POP(j, 0);

	thread_mutex_unlock(&central_queue->workq_mutex);
	return j;
}
