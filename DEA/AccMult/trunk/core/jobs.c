#include "jobs.h"

static job_list_t jobq;
static thread_mutex_t workq_mutex;
static sem_t sem_jobq;

void init_work_queue(void)
{
	thread_mutex_init(&workq_mutex, NULL);
	jobq = job_list_new();
	sem_init(&sem_jobq, 0, 0);
}

void push_tasks(job_t *tasks, unsigned ntasks)
{
	unsigned i;
	thread_mutex_lock(&workq_mutex);

	for (i = 0; i < ntasks; i++)
	{
		TRACE_JOB_PUSH(tasks[i], 0);
		job_list_push_front(jobq, tasks[i]);
		sem_post(&sem_jobq);
	}

	thread_mutex_unlock(&workq_mutex);
}

void push_task(job_t task)
{
	thread_mutex_lock(&workq_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(jobq, task);
	sem_post(&sem_jobq);

	thread_mutex_unlock(&workq_mutex);
}

void push_prio_task(job_t task)
{
#ifndef NO_PRIO
	thread_mutex_lock(&workq_mutex);

	TRACE_JOB_PUSH(task, 1);
	job_list_push_back(jobq, task);
	sem_post(&sem_jobq);

	thread_mutex_unlock(&workq_mutex);
#else
	push_task(task);
#endif
}



job_t pop_task(void)
{
	job_t j;

	sem_wait(&sem_jobq);

	thread_mutex_lock(&workq_mutex);

	if (job_list_empty(jobq)) {
		thread_mutex_unlock(&workq_mutex);
		return NULL;
	}

	j = job_list_pop_back(jobq);
	TRACE_JOB_POP(j, 0);

	thread_mutex_unlock(&workq_mutex);
	return j;
}

inline struct tag_s *get_job_tag(struct job_s *j)
{
	return j->tag;
}
