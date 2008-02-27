#include "threads.h"
#include "jobs.h"

static job_list_t jobq;
#ifdef USE_MARCEL
static marcel_mutex_t workq_mutex;
#else
static pthread_mutex_t workq_mutex;
#endif

void init_work_queue(void)
{
#ifdef USE_MARCEL
	marcel_mutex_init(&workq_mutex, NULL);
#else
	pthread_mutex_init(&workq_mutex, NULL);
#endif
	jobq = job_list_new();
}

void push_tasks(job_t *tasks, unsigned ntasks)
{
	int i;
#ifdef USE_MARCEL
	marcel_mutex_lock(&workq_mutex);
#else
	pthread_mutex_lock(&workq_mutex);
#endif

	for (i = 0; i < ntasks; i++)
		job_list_push_front(jobq, tasks[i]);

#ifdef USE_MARCEL
	marcel_mutex_unlock(&workq_mutex);
#else
	pthread_mutex_unlock(&workq_mutex);
#endif
}

void push_task(job_t task)
{
#ifdef USE_MARCEL
	marcel_mutex_lock(&workq_mutex);
#else
	pthread_mutex_lock(&workq_mutex);
#endif

	job_list_push_front(jobq, task);

#ifdef USE_MARCEL
	marcel_mutex_unlock(&workq_mutex);
#else
	pthread_mutex_unlock(&workq_mutex);
#endif
}

job_t pop_task(void)
{
	job_t j;

#ifdef USE_MARCEL
	marcel_mutex_lock(&workq_mutex);
#else
	pthread_mutex_lock(&workq_mutex);
#endif

	if (job_list_empty(jobq)) {
#ifdef USE_MARCEL
		marcel_mutex_unlock(&workq_mutex);
#else
		pthread_mutex_unlock(&workq_mutex);
#endif
		return NULL;
	}

	j = job_list_pop_back(jobq);

#ifdef USE_MARCEL
	marcel_mutex_unlock(&workq_mutex);
#else
	pthread_mutex_unlock(&workq_mutex);
#endif

	return j;
}
