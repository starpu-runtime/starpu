#include <pthread.h>
#include "jobs.h"

static job_list_t jobq;
static pthread_mutex_t workq_mutex;

void init_work_queue(void)
{
	pthread_mutex_init(&workq_mutex, NULL);
	jobq = job_list_new();
}

void push_tasks(job_t *tasks, unsigned ntasks)
{
	int i;
	pthread_mutex_lock(&workq_mutex);

	for (i = 0; i < ntasks; i++)
		job_list_push_front(jobq, tasks[i]);

	pthread_mutex_unlock(&workq_mutex);
}

void push_task(job_t task)
{
	pthread_mutex_lock(&workq_mutex);

	job_list_push_front(jobq, task);

	pthread_mutex_unlock(&workq_mutex);
}

job_t pop_task(void)
{
	job_t j;

	pthread_mutex_lock(&workq_mutex);

	if (job_list_empty(jobq)) {
		pthread_mutex_unlock(&workq_mutex);
		return NULL;
	}

	j = job_list_pop_back(jobq);

	pthread_mutex_unlock(&workq_mutex);

	return j;
}
