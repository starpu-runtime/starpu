#include <core/mechanisms/fifo_queues.h>
#include <errno.h>

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static unsigned total_number_of_jobs;

/* warning : this semaphore does not indicate the exact number of jobs, but it 
 * helps forcing some useless worker to sleep even if it may loop a little until
 * it realizes there is no work to be done */
static sem_t total_jobs_sem_t;

void init_fifo_queues_mechanisms(void)
{
	total_number_of_jobs = 0;

	sem_init(&total_jobs_sem_t, 0, 0U);
}

struct jobq_s *create_fifo(void)
{
	struct jobq_s *jobq;
	jobq = malloc(sizeof(struct jobq_s));

	struct fifo_jobq_s *fifo;
	fifo = malloc(sizeof(struct fifo_jobq_s));

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	thread_mutex_init(&fifo->workq_mutex, NULL);
	fifo->jobq = job_list_new();
	fifo->njobs = 0;
	fifo->nprocessed = 0;

	fifo->exp_start = timing_now()/1000000;
	fifo->exp_len = 0.0;
	fifo->exp_end = fifo->exp_start;

	sem_init(&fifo->sem_jobq, 0, 0);

	jobq->queue = fifo;

	return jobq;
}

unsigned get_total_njobs_fifos(void)
{
	return total_number_of_jobs;
}

unsigned get_fifo_njobs(struct jobq_s *q)
{
	ASSERT(q);

	struct fifo_jobq_s *fifo_queue = q->queue;

	return fifo_queue->njobs;
}

unsigned get_fifo_nprocessed(struct jobq_s *q)
{
	ASSERT(q);

	struct fifo_jobq_s *fifo_queue = q->queue;

	return fifo_queue->nprocessed;
}

void fifo_push_prio_task(struct jobq_s *q, job_t task)
{
#ifndef NO_PRIO
	ASSERT(q);
	struct fifo_jobq_s *fifo_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&fifo_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_back(fifo_queue->jobq, task);

	fifo_queue->njobs++;
	fifo_queue->nprocessed++;

	/* semaphore for the local queue */
	sem_post(&fifo_queue->sem_jobq);

	thread_mutex_unlock(&fifo_queue->workq_mutex);
#else
	fifo_push_task(q, task);
#endif
}

void fifo_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct fifo_jobq_s *fifo_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	/* semaphore for the entire system */
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&fifo_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(fifo_queue->jobq, task);
	fifo_queue->njobs++;
	fifo_queue->nprocessed++;

	/* semaphore for the local queue */
	sem_post(&fifo_queue->sem_jobq);

	thread_mutex_unlock(&fifo_queue->workq_mutex);
}

job_t fifo_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct fifo_jobq_s *fifo_queue = q->queue;

	/* block until some task is available in that queue */
	sem_wait(&fifo_queue->sem_jobq);

	thread_mutex_lock(&fifo_queue->workq_mutex);

	j = job_list_pop_back(fifo_queue->jobq);

	/* there was some task */
	ASSERT(j);
	fifo_queue->njobs--;
	
	TRACE_JOB_POP(j, 0);

	/* we are sure that we got it now, so at worst, some people thought 
	 * there remained some work and will soon discover it is not true */
	(void)ATOMIC_ADD(&total_number_of_jobs, -1);

	thread_mutex_unlock(&fifo_queue->workq_mutex);

	return j;

}

/* for work stealing, typically */
job_t fifo_non_blocking_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct fifo_jobq_s *fifo_queue = q->queue;

	thread_mutex_lock(&fifo_queue->workq_mutex);

	int fail;
	fail = sem_trywait(&fifo_queue->sem_jobq);

	if (fail == -1 && errno == EAGAIN) {
		/* there is nothing in that queue */
		ASSERT (job_list_empty(fifo_queue->jobq));
		ASSERT(fifo_queue->njobs == 0);

		thread_mutex_unlock(&fifo_queue->workq_mutex);

		j = NULL;
	}
	else {
		ASSERT(fail == 0);		
		j = job_list_pop_back(fifo_queue->jobq);
		/* there was some task */
		fifo_queue->njobs--;
	
		TRACE_JOB_POP(j, 0);

		/* we are sure that we got it now, so at worst, some people thought 
		 * there remained some work and will soon discover it is not true */
		(void)ATOMIC_ADD(&total_number_of_jobs, -1);

		thread_mutex_unlock(&fifo_queue->workq_mutex);
	}

	return j;
}

job_t fifo_non_blocking_pop_task_if_job_exists(struct jobq_s *q)
{
	job_t j;

	j = fifo_non_blocking_pop_task(q);

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
