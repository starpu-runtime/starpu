#include <core/mechanisms/work_stealing_queues.h>

/*
 * "Work stealing" queues
 */

struct jobq_s *create_deque(void)
{
	struct jobq_s *jobq;
	jobq = malloc(sizeof(struct jobq_s));

	struct deque_jobq_s *deque;
	deque = malloc(sizeof(struct deque_jobq_s));

	thread_mutex_init(&deque->workq_mutex, NULL);
	deque->jobq = job_list_new();
	deque->njobs = 0;

	jobq->queue = deque;

	return jobq;
}

void ws_push_prio_task(struct jobq_s *q, job_t task)
{
#ifndef NO_PRIO
	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	thread_mutex_lock(&deque_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(deque_queue->jobq, task);
	deque_queue->njobs++;

	thread_mutex_unlock(&deque_queue->workq_mutex);
#else
	ws_push_task(q, task);
#endif
}



void ws_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	thread_mutex_lock(&deque_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(deque_queue->jobq, task);
	deque_queue->njobs++;

	thread_mutex_unlock(&deque_queue->workq_mutex);
}

//job_t ws_pop_task(struct jobq_s *q)
//{
//	job_t j;
//
//	ASSERT(q);
//	struct central_jobq_s *central_queue = q->queue;
//
//	sem_wait(&central_queue->sem_jobq);
//
//	thread_mutex_lock(&central_queue->workq_mutex);
//
//	if (job_list_empty(central_queue->jobq)) {
//		thread_mutex_unlock(&central_queue->workq_mutex);
//		return NULL;
//	}
//
//	j = job_list_pop_back(central_queue->jobq);
//	TRACE_JOB_POP(j, 0);
//
//	thread_mutex_unlock(&central_queue->workq_mutex);
//	return j;
//}

job_t ws_non_blocking_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	thread_mutex_lock(&deque_queue->workq_mutex);

	if (deque_queue->njobs == 0) {
		ASSERT (job_list_empty(deque_queue->jobq));
		thread_mutex_unlock(&deque_queue->workq_mutex);
		return NULL;
	}

	/* there was some task */
	deque_queue->njobs--;
	
	j = job_list_pop_back(deque_queue->jobq);
	TRACE_JOB_POP(j, 0);

	thread_mutex_unlock(&deque_queue->workq_mutex);
	return j;
}
