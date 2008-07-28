#include <core/mechanisms/deque_queues.h>
#include <errno.h>

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static unsigned total_number_of_jobs;

/* warning : this semaphore does not indicate the exact number of jobs, but it 
 * helps forcing some useless worker to sleep even if it may loop a little until
 * it realizes there is no work to be done */
static sem_t total_jobs_sem_t;

void init_deque_queues_mechanisms(void)
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

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	thread_mutex_init(&deque->workq_mutex, NULL);
	deque->jobq = job_list_new();
	deque->njobs = 0;
	deque->nprocessed = 0;
	sem_init(&deque->sem_jobq, 0, 0);

	jobq->queue = deque;

	return jobq;
}

unsigned get_total_njobs_deques(void)
{
	return total_number_of_jobs;
}

unsigned get_deque_njobs(struct jobq_s *q)
{
	ASSERT(q);

	struct deque_jobq_s *deque_queue = q->queue;

	return deque_queue->njobs;
}

unsigned get_deque_nprocessed(struct jobq_s *q)
{
	ASSERT(q);

	struct deque_jobq_s *deque_queue = q->queue;

	return deque_queue->nprocessed;
}

void deque_push_prio_task(struct jobq_s *q, job_t task)
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

	/* semaphore for the local queue */
	sem_post(&deque_queue->sem_jobq);

	thread_mutex_unlock(&deque_queue->workq_mutex);
#else
	ws_push_task(q, task);
#endif
}

void deque_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	/* semaphore for the entire system */
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&deque_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(deque_queue->jobq, task);
	deque_queue->njobs++;
	deque_queue->nprocessed++;

	/* semaphore for the local queue */
	sem_post(&deque_queue->sem_jobq);

	thread_mutex_unlock(&deque_queue->workq_mutex);
}

job_t deque_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	/* block until some task is available in that queue */
	sem_wait(&deque_queue->sem_jobq);

	thread_mutex_lock(&deque_queue->workq_mutex);

	j = job_list_pop_back(deque_queue->jobq);

	/* there was some task */
	ASSERT(j);
	deque_queue->njobs--;
	
	TRACE_JOB_POP(j, 0);

	/* we are sure that we got it now, so at worst, some people thought 
	 * there remained some work and will soon discover it is not true */
	(void)ATOMIC_ADD(&total_number_of_jobs, -1);

	thread_mutex_unlock(&deque_queue->workq_mutex);

	return j;

}

job_t deque_non_blocking_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct deque_jobq_s *deque_queue = q->queue;

	thread_mutex_lock(&deque_queue->workq_mutex);

	int fail;
	fail = sem_trywait(&deque_queue->sem_jobq);

	if (fail == -1 && errno == EAGAIN) {
		/* there is nothing in that queue */
		ASSERT (job_list_empty(deque_queue->jobq));
		ASSERT(deque_queue->njobs == 0);

		thread_mutex_unlock(&deque_queue->workq_mutex);

		j = NULL;
	}
	else {
		ASSERT(fail == 0);		
		j = job_list_pop_back(deque_queue->jobq);
		/* there was some task */
		deque_queue->njobs--;
	
		TRACE_JOB_POP(j, 0);

		/* we are sure that we got it now, so at worst, some people thought 
		 * there remained some work and will soon discover it is not true */
		(void)ATOMIC_ADD(&total_number_of_jobs, -1);

		thread_mutex_unlock(&deque_queue->workq_mutex);
	}

	return j;
}

job_t deque_non_blocking_pop_task_if_job_exists(struct jobq_s *q)
{
	job_t j;

	j = deque_non_blocking_pop_task(q);

	if (!j && (ATOMIC_ADD(&total_number_of_jobs, 0) == 0)) {
		/* there is no job at all in the entire system : go to sleep ! */

		/* that wait is not an absolute sign that there is some work 
		 * if there is some, the thread should be awoken, but if there is none 
		 * at the moment it is awoken, it may simply poll a limited number of 
		 * times and just get back to sleep */
		sem_wait(&total_jobs_sem_t);
	}

	return j;
}
