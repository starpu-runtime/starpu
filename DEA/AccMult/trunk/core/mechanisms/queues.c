#include "queues.h"

/*
 * There can be various queue designs
 * 	- trivial single list
 * 	- cilk-like 
 * 	- hierarchical (marcel-like)
 */

/*
 * We need some "common" queue so that anyone (even some thread unknown by 
 * our runtime) may submit tasks
 */

/*  get_local_queue -> returns local queue or that generic queue 
 *  					if there is no local one
 */


/*
 * Centralized queue 
 */

void init_central_jobq(struct central_jobq_s *queue)
{
	ASSERT(queue);

	thread_mutex_init(&queue->workq_mutex, NULL);
	queue->jobq = job_list_new();
	sem_init(&queue->sem_jobq, 0, 0);
}

void central_push_task(struct central_jobq_s *queue, job_t task)
{
	thread_mutex_lock(&queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(queue->jobq, task);
	sem_post(&queue->sem_jobq);

	thread_mutex_unlock(&queue->workq_mutex);
}

void central_push_prio_task(struct central_jobq_s *queue, job_t task)
{
#ifndef NO_PRIO
	thread_mutex_lock(&queue->workq_mutex);

	TRACE_JOB_PUSH(task, 1);
	job_list_push_back(queue->jobq, task);
	sem_post(&queue->sem_jobq);

	thread_mutex_unlock(&queue->workq_mutex);
#else
	central_push_task(queue, task);
#endif
}

job_t central_pop_task(struct central_jobq_s *queue)
{
	job_t j;

	sem_wait(&queue->sem_jobq);

	thread_mutex_lock(&queue->workq_mutex);

	if (job_list_empty(queue->jobq)) {
		thread_mutex_unlock(&queue->workq_mutex);
		return NULL;
	}

	j = job_list_pop_back(queue->jobq);
	TRACE_JOB_POP(j, 0);

	thread_mutex_unlock(&queue->workq_mutex);
	return j;
}
