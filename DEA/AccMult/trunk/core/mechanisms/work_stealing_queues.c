#include <core/mechanisms/work_stealing_queues.h>

/*
 * "Work stealing" queues
 */

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static unsigned total_number_of_jobs;
static sem_t total_jobs_sem_t;

void init_ws_queues_mechanisms(void)
{
	total_number_of_jobs = 0;

	sem_init(&total_jobs_sem_t, 0, 0U);
}

struct jobq_s *create_deque(void)
{
	struct jobq_s *jobq;
	jobq = malloc(sizeof(struct jobq_s));

	struct deque_jobq_s *deque;
	deque = malloc(sizeof(struct deque_jobq_s));

	thread_mutex_init(&deque->workq_mutex, NULL);
	deque->jobq = job_list_new();
	deque->njobs = 0;
	deque->nprocessed = 0;

	jobq->queue = deque;

	return jobq;
}

unsigned get_total_njobs_ws(void)
{
	return total_number_of_jobs;
}

unsigned get_queue_njobs_ws(struct jobq_s *q)
{
	ASSERT(q);

	struct deque_jobq_s *deque_queue = q->queue;

	return deque_queue->njobs;
}

unsigned get_queue_nprocessed_ws(struct jobq_s *q)
{
	ASSERT(q);

	struct deque_jobq_s *deque_queue = q->queue;

	return deque_queue->nprocessed;
}

void ws_push_prio_task(struct jobq_s *q, job_t task)
{
#ifndef NO_PRIO
	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&deque_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(deque_queue->jobq, task);

	deque_queue->njobs++;
	deque_queue->nprocessed++;

	thread_mutex_unlock(&deque_queue->workq_mutex);
#else
	ws_push_task(q, task);
#endif
}



void ws_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&deque_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(deque_queue->jobq, task);
	deque_queue->njobs++;
	deque_queue->nprocessed++;

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

	j = job_list_pop_back(deque_queue->jobq);

	/* there was some task */
	deque_queue->njobs--;
	
	TRACE_JOB_POP(j, 0);

	/* we are sure that we got it now, so at worst, some people thought 
	 * there remained some work and will soon discover it is not true */
	(void)ATOMIC_ADD(&total_number_of_jobs, -1);

	thread_mutex_unlock(&deque_queue->workq_mutex);
	return j;
}

job_t ws_non_blocking_pop_task_if_job_exists(struct jobq_s *q)
{
	job_t j;

	j = ws_non_blocking_pop_task(q);

	if (!j && (ATOMIC_ADD(&total_number_of_jobs, 0) == 0)) {
		/* there is no job at all in the entire system : go to sleep ! */
		sem_wait(&total_jobs_sem_t);
	}

	return j;
}
