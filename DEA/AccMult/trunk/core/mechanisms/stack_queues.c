#include <core/mechanisms/stack_queues.h>
#include <errno.h>

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static unsigned total_number_of_jobs;

/* warning : this semaphore does not indicate the exact number of jobs, but it 
 * helps forcing some useless worker to sleep even if it may loop a little until
 * it realizes there is no work to be done */
static sem_t total_jobs_sem_t;

void init_stack_queues_mechanisms(void)
{
	total_number_of_jobs = 0;

	sem_init(&total_jobs_sem_t, 0, 0U);
}

struct jobq_s *create_stack(void)
{
	struct jobq_s *jobq;
	jobq = malloc(sizeof(struct jobq_s));

	struct stack_jobq_s *stack;
	stack = malloc(sizeof(struct stack_jobq_s));

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	thread_mutex_init(&stack->workq_mutex, NULL);
	stack->jobq = job_list_new();
	stack->njobs = 0;
	stack->nprocessed = 0;

	stack->exp_start = timing_now()/1000000;
	stack->exp_len = 0.0;
	stack->exp_end = stack->exp_start;

	sem_init(&stack->sem_jobq, 0, 0);

	jobq->queue = stack;

	return jobq;
}

unsigned get_total_njobs_stacks(void)
{
	return total_number_of_jobs;
}

unsigned get_stack_njobs(struct jobq_s *q)
{
	ASSERT(q);

	struct stack_jobq_s *stack_queue = q->queue;

	return stack_queue->njobs;
}

unsigned get_stack_nprocessed(struct jobq_s *q)
{
	ASSERT(q);

	struct stack_jobq_s *stack_queue = q->queue;

	return stack_queue->nprocessed;
}

void stack_push_prio_task(struct jobq_s *q, job_t task)
{
#ifndef NO_PRIO
	ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&stack_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_back(stack_queue->jobq, task);

	stack_queue->njobs++;
	stack_queue->nprocessed++;

	/* semaphore for the local queue */
	sem_post(&stack_queue->sem_jobq);

	thread_mutex_unlock(&stack_queue->workq_mutex);
#else
	stack_push_task(q, task);
#endif
}

void stack_push_task(struct jobq_s *q, job_t task)
{
	ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	/* do that early to avoid sleepy for no reason */
	(void)ATOMIC_ADD(&total_number_of_jobs, 1);
	/* semaphore for the entire system */
	sem_post(&total_jobs_sem_t);

	thread_mutex_lock(&stack_queue->workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(stack_queue->jobq, task);
	stack_queue->njobs++;
	stack_queue->nprocessed++;

	/* semaphore for the local queue */
	sem_post(&stack_queue->sem_jobq);

	thread_mutex_unlock(&stack_queue->workq_mutex);
}

job_t stack_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	/* block until some task is available in that queue */
	sem_wait(&stack_queue->sem_jobq);

	thread_mutex_lock(&stack_queue->workq_mutex);

	j = job_list_pop_back(stack_queue->jobq);

	/* there was some task */
	ASSERT(j);
	stack_queue->njobs--;
	
	TRACE_JOB_POP(j, 0);

	/* we are sure that we got it now, so at worst, some people thought 
	 * there remained some work and will soon discover it is not true */
	(void)ATOMIC_ADD(&total_number_of_jobs, -1);

	thread_mutex_unlock(&stack_queue->workq_mutex);

	return j;

}

/* for work stealing, typically */
job_t stack_non_blocking_pop_task(struct jobq_s *q)
{
	job_t j;

	ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	thread_mutex_lock(&stack_queue->workq_mutex);

	int fail;
	fail = sem_trywait(&stack_queue->sem_jobq);

	if (fail == -1 && errno == EAGAIN) {
		/* there is nothing in that queue */
		ASSERT (job_list_empty(stack_queue->jobq));
		ASSERT(stack_queue->njobs == 0);

		thread_mutex_unlock(&stack_queue->workq_mutex);

		j = NULL;
	}
	else {
		ASSERT(fail == 0);		
		j = job_list_pop_back(stack_queue->jobq);
		/* there was some task */
		stack_queue->njobs--;
	
		TRACE_JOB_POP(j, 0);

		/* we are sure that we got it now, so at worst, some people thought 
		 * there remained some work and will soon discover it is not true */
		(void)ATOMIC_ADD(&total_number_of_jobs, -1);

		thread_mutex_unlock(&stack_queue->workq_mutex);
	}

	return j;
}

job_t stack_non_blocking_pop_task_if_job_exists(struct jobq_s *q)
{
	job_t j;

	j = stack_non_blocking_pop_task(q);

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
